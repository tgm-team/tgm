import argparse
import time
from collections import deque
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from tqdm import tqdm

from tgm import DGBatch, DGData, DGraph
from tgm.hooks import HookManager, NegativeEdgeSamplerHook, RecencyNeighborHook
from tgm.hooks.base import StatefulHook
from tgm.loader import DGDataLoader
from tgm.nn import Time2Vec
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='GraphMixer Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--bsize', type=int, default=600, help='batch size')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--dropout', type=str, default=0.1, help='dropout rate')
parser.add_argument('--time-dim', type=int, default=100, help='time encoding dimension')
parser.add_argument('--embed-dim', type=int, default=128, help='embedding dimension')
parser.add_argument('--n-nbrs', type=int, default=10, help='num sampled nbrs')
parser.add_argument(
    '--node-dim', type=int, default=100, help='node feat dimension if not provided'
)
parser.add_argument(
    '--time-gap', type=int, default=2000, help='graphmixer time slot size'
)
parser.add_argument(
    '--token-dim-expansion',
    type=float,
    default=0.5,
    help='token dimension expansion factor in MLP sub-blocks',
)
parser.add_argument(
    '--channel-dim-expansion',
    type=float,
    default=0.5,
    help='channel dimension expansion factor in MLP sub-blocks',
)


class GraphMixer(nn.Module):
    def __init__(
        self,
        time_dim: int,
        embed_dim: int,
        num_tokens: int,
        node_dim: int,
        edge_dim: int,
        num_layers: int = 2,
        token_dim_expansion: float = 0.5,
        channel_dim_expansion: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.link_predictor = LinkPredictor(dim=embed_dim)

        # GraphMixer time encoding function is not trainable
        self.time_encoder = Time2Vec(time_dim=time_dim)
        for param in self.time_encoder.parameters():
            param.requires_grad = False

        self.projection_layer = nn.Linear(edge_dim + time_dim, edge_dim)
        self.mlp_mixers = nn.ModuleList(
            [
                MLPMixer(
                    num_tokens=num_tokens,
                    num_channels=edge_dim,
                    token_dim_expansion=token_dim_expansion,
                    channel_dim_expansion=channel_dim_expansion,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_layer = nn.Linear(
            in_features=edge_dim + node_dim, out_features=embed_dim
        )

    def forward(
        self, batch: DGBatch, node_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Link Encoder
        hop = 0
        edge_feat = batch.nbr_feats[hop]
        nbr_time_feat = self.time_encoder(
            batch.time.unsqueeze(dim=1).repeat(3, 1) - batch.nbr_times[hop]
        )
        z_link = torch.cat([edge_feat, nbr_time_feat], dim=-1)
        z_link = self.projection_layer(z_link)
        for mixer in self.mlp_mixers:
            z_link = mixer(x=z_link)
        z_link = torch.mean(z_link, dim=1)

        # Node Encoder
        time_gap_node_feat = node_feat[batch.time_gap_node_nids]
        masked_feat = time_gap_node_feat.masked_fill(
            batch.time_gap_node_mask.unsqueeze(-1) == 0, -1e10
        )
        scores = torch.softmax(masked_feat, dim=1)

        # Handle rows with no valid neighbors
        scores[batch.time_gap_node_mask.sum(dim=1) == 0] = 0
        z_node = torch.sum(time_gap_node_feat * scores, dim=1)
        z_node += node_feat[torch.cat([batch.src, batch.dst, batch.neg])]

        # Link Decoder
        z = self.output_layer(torch.cat([z_link, z_node], dim=1))
        z_src = z[batch.global_to_local(batch.src)]
        z_dst = z[batch.global_to_local(batch.dst)]
        z_neg = z[batch.global_to_local(batch.neg)]
        pos_out = self.link_predictor(z_src, z_dst)
        neg_out = self.link_predictor(z_src, z_neg)
        return pos_out, neg_out


class FeedForwardNet(nn.Module):
    def __init__(
        self, input_dim: int, dim_expansion_factor: float, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(
                in_features=input_dim,
                out_features=int(dim_expansion_factor * input_dim),
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                in_features=int(dim_expansion_factor * input_dim),
                out_features=input_dim,
            ),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class MLPMixer(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        num_channels: int,
        token_dim_expansion: float = 0.5,
        channel_dim_expansion: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(num_tokens)
        self.token_ff = FeedForwardNet(
            input_dim=num_tokens,
            dim_expansion_factor=token_dim_expansion,
            dropout=dropout,
        )
        self.channel_norm = nn.LayerNorm(num_channels)
        self.channel_ff = FeedForwardNet(
            input_dim=num_channels,
            dim_expansion_factor=channel_dim_expansion,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mix tokens
        z = self.token_norm(x.permute(0, 2, 1))  # (B, num_channels, num_tokens)
        z = self.token_ff(z).permute(0, 2, 1)  # (B, num_tokens, num_channels)
        out = z + x

        # mix channels
        z = self.channel_norm(out)
        z = self.channel_ff(z)
        out = z + out
        return out


class LinkPredictor(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2 * dim, dim)
        self.fc2 = nn.Linear(dim, 1)

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        h = self.fc1(torch.cat([z_src, z_dst], dim=1))
        h = h.relu()
        return self.fc2(h).sigmoid().view(-1)


def train(
    loader: DGDataLoader,
    static_node_feats: torch.Tensor,
    model: nn.Module,
    opt: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0
    for batch in tqdm(loader):
        opt.zero_grad()
        pos_out, neg_out = model(batch, static_node_feats)
        loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        loss += F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
        loss.backward()
        opt.step()
        total_loss += float(loss)
    return total_loss


@torch.no_grad()
def eval(
    loader: DGDataLoader,
    static_node_feats: torch.Tensor,
    model: nn.Module,
    metrics: Metric,
) -> dict:
    model.eval()
    for batch in tqdm(loader):
        pos_out, neg_out = model(batch, static_node_feats)
        y_pred = torch.cat([pos_out, neg_out], dim=0).float()
        y_true = (
            torch.cat(
                [torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0
            )
            .long()
            .to(y_pred.device)
        )
        indexes = torch.zeros(y_pred.size(0), dtype=torch.long, device=y_pred.device)
        metrics(y_pred, y_true, indexes=indexes)
    return metrics.compute()


args = parser.parse_args()
seed_everything(args.seed)

train_data, val_data, test_data = DGData.from_tgb(args.dataset).split()

train_dg = DGraph(train_data, device=args.device)
val_dg = DGraph(val_data, device=args.device)
test_dg = DGraph(test_data, device=args.device)


# Neighbor Sampler and GraphMixer Hook is shared across loaders
nbr_hook = RecencyNeighborHook(
    num_nbrs=args.n_nbrs,
    num_nodes=test_dg.num_nodes,  # Assuming node ids at test set > train/val set
    edge_feats_dim=test_dg.edge_feats_dim,
)


class GraphMixerHook(StatefulHook):
    r"""Custom hook that gets 1-hop neighbors in a specific window.

    If N(v_i, t_s, t_e) = nbrs of v_i from [t_s, t_e], then we materialize
    N(node_ids, t - TIME_GAP, t) for all seed nodes in a given batch.
    """

    requires = {'neg'}
    produces = {'time_gap_node_nids', 'time_gap_node_mask'}

    def __init__(self, time_gap: int, num_nodes: int) -> None:
        self._time_gap = time_gap
        self._nbrs = {}
        for node in range(num_nodes):
            self._nbrs[node] = deque(maxlen=self._time_gap)

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        batch.neg = batch.neg.to(dg.device)
        seed_nodes = torch.cat([batch.src, batch.dst, batch.neg])  # type: ignore

        unique, inverse_indices = seed_nodes.unique(return_inverse=True)

        batch_size = len(seed_nodes)
        nbr_nids = torch.zeros(
            batch_size, self._time_gap, dtype=torch.long, device=dg.device
        )
        nbr_mask = torch.zeros(
            batch_size, self._time_gap, dtype=torch.long, device=dg.device
        )
        for i, node in enumerate(unique.tolist()):
            if nn := len(self._nbrs[node]):
                mask = inverse_indices == i
                nbr_nids[mask, :nn] = torch.tensor(
                    self._nbrs[node], device=dg.device, dtype=torch.long
                )
                nbr_mask[mask, :nn] = nn >= self._time_gap

        batch.time_gap_node_nids = nbr_nids  # type: ignore
        batch.time_gap_node_mask = nbr_mask  # type: ignore

        self._update(batch)
        return batch

    def reset(self) -> None:
        for node in self._nbrs:
            self._nbrs[node].clear()

    def _update(self, batch: DGBatch) -> None:
        for i in range(batch.src.size(0)):
            src_nbr = int(batch.src[i].item())
            dst_nbr = int(batch.dst[i].item())
            self._nbrs[src_nbr].append(dst_nbr)
            self._nbrs[dst_nbr].append(src_nbr)


graph_mixer_hook = GraphMixerHook(args.time_gap, num_nodes=test_dg.num_nodes)

_, train_dst, _ = train_dg.edges
_, val_dst, _ = val_dg.edges
_, test_dst, _ = test_dg.edges
train_neg_hook = NegativeEdgeSamplerHook(
    low=int(train_dst.min()), high=int(train_dst.max())
)
val_neg_hook = NegativeEdgeSamplerHook(low=int(val_dst.min()), high=int(val_dst.max()))
test_neg_hook = NegativeEdgeSamplerHook(
    low=int(test_dst.min()), high=int(test_dst.max())
)

hm = HookManager(args.device)
hm.register('train', train_neg_hook)
hm.register('val', val_neg_hook)
hm.register('test', test_neg_hook)
hm.register_shared(nbr_hook)
hm.register_shared(graph_mixer_hook)

train_loader = DGDataLoader(train_dg, args.bsize, hook_manager=hm)
val_loader = DGDataLoader(val_dg, args.bsize, hook_manager=hm)
test_loader = DGDataLoader(test_dg, args.bsize, hook_manager=hm)


if train_dg.static_node_feats is not None:
    static_node_feats = train_dg.static_node_feats
else:
    static_node_feats = torch.randn(
        (test_dg.num_nodes, args.node_dim), device=args.device
    )

model = GraphMixer(
    embed_dim=args.embed_dim,
    time_dim=args.time_dim,
    num_tokens=args.n_nbrs,
    token_dim_expansion=float(args.token_dim_expansion),
    channel_dim_expansion=float(args.channel_dim_expansion),
    dropout=float(args.dropout),
    node_dim=static_node_feats.shape[1],
    edge_dim=train_dg.edge_feats_dim | args.embed_dim,
).to(args.device)
opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

metrics = [BinaryAveragePrecision(), BinaryAUROC()]
val_metrics = MetricCollection(metrics, prefix='Validation')
test_metrics = MetricCollection(metrics, prefix='Test')

for epoch in range(1, args.epochs + 1):
    with hm.activate('train'):
        start_time = time.perf_counter()
        loss = train(train_loader, static_node_feats, model, opt)
        end_time = time.perf_counter()
        latency = end_time - start_time

    with hm.activate('val'):
        val_results = eval(val_loader, static_node_feats, model, val_metrics)
        val_metrics.reset()
        print(
            f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f} '
            + ' '.join(f'{k}={v.item():.4f}' for k, v in val_results.items())
        )

    if epoch < args.epochs:  # Reset hooks after each epoch, except last epoch
        hm.reset_state()

with hm.activate('test'):
    test_results = eval(test_loader, static_node_feats, model, test_metrics)
    print(' '.join(f'{k}={v.item():.4f}' for k, v in test_results.items()))
