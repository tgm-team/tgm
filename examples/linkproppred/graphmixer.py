import argparse
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.linkproppred.evaluate import Evaluator
from tqdm import tqdm

from tgm import DGBatch, DGData, DGraph, RecipeRegistry
from tgm.constants import (
    METRIC_TGB_LINKPROPPRED,
    PADDED_NODE_ID,
    RECIPE_TGB_LINK_PRED,
)
from tgm.hooks import RecencyNeighborHook, StatefulHook
from tgm.loader import DGDataLoader
from tgm.nn import Time2Vec
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='GraphMixer LinkPropPred Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--dropout', type=str, default=0.1, help='dropout rate')
parser.add_argument('--n-nbrs', type=int, default=20, help='num sampled nbrs')
parser.add_argument('--time-dim', type=int, default=100, help='time encoding dimension')
parser.add_argument('--embed-dim', type=int, default=128, help='attention dimension')
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
    default=4.0,
    help='channel dimension expansion factor in MLP sub-blocks',
)


class GraphMixerEncoder(nn.Module):
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

    def forward(self, batch: DGBatch, node_feat: torch.Tensor) -> torch.Tensor:
        # Link Encoder
        edge_feat = batch.nbr_feats[0]
        nbr_time_feat = self.time_encoder(batch.times[0][:, None] - batch.nbr_times[0])
        nbr_time_feat[batch.nbr_nids[0] == PADDED_NODE_ID] = 0.0
        z_link = self.projection_layer(torch.cat([edge_feat, nbr_time_feat], dim=-1))
        for mixer in self.mlp_mixers:
            z_link = mixer(x=z_link)
        z_link = torch.mean(z_link, dim=1)

        # Node Encoder
        time_gap_node_feats = node_feat[batch.time_gap_node_nids]
        mask = (batch.time_gap_node_nids != PADDED_NODE_ID).float()
        masked_feats = time_gap_node_feats * mask.unsqueeze(-1)
        nbr_count = mask.sum(dim=1, keepdim=True).clamp(min=1)  # Mean over valid nbrs
        agg_feats = masked_feats.sum(dim=1) / nbr_count
        z_node = agg_feats + node_feat[torch.cat([batch.src, batch.dst, batch.neg])]

        z = self.output_layer(torch.cat([z_link, z_node], dim=1))
        return z


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
        return self.fc2(h).view(-1)


def train(
    loader: DGDataLoader,
    static_node_feats: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    opt: torch.optim.Optimizer,
) -> float:
    encoder.train()
    decoder.train()
    total_loss = 0

    for batch in tqdm(loader):
        opt.zero_grad()

        z = encoder(batch, static_node_feats)
        z_src, z_dst, z_neg = torch.chunk(z, 3)

        pos_out = decoder(z_src, z_dst)
        neg_out = decoder(z_src, z_neg)

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
    encoder: nn.Module,
    decoder: nn.Module,
    evaluator: Evaluator,
) -> float:
    encoder.eval()
    decoder.eval()
    perf_list = []

    for batch in tqdm(loader):
        z = encoder(batch, static_node_feats)
        id_map = {nid.item(): i for i, nid in enumerate(batch.nids[0])}
        for idx, neg_batch in enumerate(batch.neg_batch_list):
            dst_ids = torch.cat([batch.dst[idx].unsqueeze(0), neg_batch])
            src_ids = batch.src[idx].repeat(len(dst_ids))

            src_idx = torch.tensor([id_map[n.item()] for n in src_ids], device=z.device)
            dst_idx = torch.tensor([id_map[n.item()] for n in dst_ids], device=z.device)
            z_src = z[src_idx]
            z_dst = z[dst_idx]
            y_pred = decoder(z_src, z_dst).sigmoid()

            input_dict = {
                'y_pred_pos': y_pred[0],
                'y_pred_neg': y_pred[1:],
                'eval_metric': [METRIC_TGB_LINKPROPPRED],
            }
            perf_list.append(evaluator.eval(input_dict)[METRIC_TGB_LINKPROPPRED])

    return float(np.mean(perf_list))


args = parser.parse_args()
seed_everything(args.seed)
evaluator = Evaluator(name=args.dataset)

train_data, val_data, test_data = DGData.from_tgb(args.dataset).split()
train_dg = DGraph(train_data, device=args.device)
val_dg = DGraph(val_data, device=args.device)
test_dg = DGraph(test_data, device=args.device)


class GraphMixerHook(StatefulHook):
    r"""Custom hook that gets 1-hop neighbors in a specific window.

    If N(v_i, t_s, t_e) = nbrs of v_i from [t_s, t_e], then we materialize
    N(node_ids, t - TIME_GAP, t) for all seed nodes in a given batch.
    """

    requires = {'neg'}
    produces = {'time_gap_node_nids'}

    def __init__(self, time_gap: int) -> None:
        self._num_nbrs = time_gap
        self._history = deque(maxlen=self._num_nbrs)
        self._device = torch.device('cpu')

    def reset_state(self) -> None:
        self._history.clear()

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        device = dg.device
        self._move_queues_to_device_if_needed(device)  # No-op after first batch

        batch.neg = batch.neg.to(device)

        seed_nodes = torch.cat([batch.src, batch.dst, batch.neg])
        seed_times = torch.cat([batch.time.repeat(2), batch.neg_time])

        batch.time_gap_node_nids = self._get_recency_neighbors(
            seed_nodes, seed_times, self._num_nbrs
        )

        self._update(batch)
        return batch

    def _get_recency_neighbors(
        self, node_ids: torch.Tensor, query_times: torch.Tensor, k: int
    ) -> torch.Tensor:
        num_nodes = node_ids.size(0)
        device = node_ids.device
        nbr_nids = torch.full(
            (num_nodes, k), PADDED_NODE_ID, dtype=torch.long, device=device
        )

        for i in range(num_nodes):
            nid, qtime = int(node_ids[i]), int(query_times[i])
            nbrs = []
            for u, v, t in reversed(self._history):  # most recent first
                if t < qtime:
                    if u == nid:
                        nbrs.append(v)
                    elif v == nid:
                        nbrs.append(u)
                    if len(nbrs) == k:
                        break

            if nbrs:
                nbr_nids[i, -len(nbrs) :] = torch.tensor(nbrs, device=device)
        return nbr_nids

    def _update(self, batch: DGBatch) -> None:
        src, dst, time = batch.src.tolist(), batch.dst.tolist(), batch.time.tolist()
        for s, d, t in zip(src, dst, time):
            self._history.append((s, d, t))
            self._history.append((d, s, t))  # undirected

    def _move_queues_to_device_if_needed(self, device: torch.device) -> None:
        if device != self._device:
            self._device = device


hm = RecipeRegistry.build(
    RECIPE_TGB_LINK_PRED, dataset_name=args.dataset, train_dg=train_dg
)
train_key, val_key, test_key = hm.keys
hm.register_shared(GraphMixerHook(args.time_gap))
hm.register_shared(
    RecencyNeighborHook(
        num_nbrs=[args.n_nbrs],
        num_nodes=test_dg.num_nodes,
        edge_feats_dim=test_dg.edge_feats_dim,
    )
)

train_loader = DGDataLoader(train_dg, args.bsize, hook_manager=hm)
val_loader = DGDataLoader(val_dg, args.bsize, hook_manager=hm)
test_loader = DGDataLoader(test_dg, args.bsize, hook_manager=hm)


if train_dg.static_node_feats is not None:
    static_node_feats = train_dg.static_node_feats
else:
    static_node_feats = torch.randn(
        (test_dg.num_nodes, args.node_dim), device=args.device
    )

encoder = GraphMixerEncoder(
    embed_dim=args.embed_dim,
    time_dim=args.time_dim,
    num_tokens=args.n_nbrs,
    token_dim_expansion=float(args.token_dim_expansion),
    channel_dim_expansion=float(args.channel_dim_expansion),
    dropout=float(args.dropout),
    node_dim=static_node_feats.shape[1],
    edge_dim=train_dg.edge_feats_dim | args.embed_dim,
).to(args.device)
decoder = LinkPredictor(args.embed_dim).to(args.device)
opt = torch.optim.Adam(
    set(encoder.parameters()) | set(decoder.parameters()), lr=float(args.lr)
)

for epoch in range(1, args.epochs + 1):
    with hm.activate(train_key):
        start_time = time.perf_counter()
        loss = train(train_loader, static_node_feats, encoder, decoder, opt)
        end_time = time.perf_counter()
        latency = end_time - start_time

    with hm.activate(val_key):
        val_mrr = eval(val_loader, static_node_feats, encoder, decoder, evaluator)
    print(
        f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f} Validation {METRIC_TGB_LINKPROPPRED}={val_mrr:.4f}'
    )

    if epoch < args.epochs:  # Reset hooks after each epoch, except last epoch
        hm.reset_state()

with hm.activate('test'):
    test_mrr = eval(test_loader, static_node_feats, encoder, decoder, evaluator)
    print(f'Test {METRIC_TGB_LINKPROPPRED}={test_mrr:.4f}')
