import argparse
import time
from collections import deque
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from tqdm import tqdm

from opendg.graph import DGBatch, DGraph
from opendg.hooks import DGHook, NegativeEdgeSamplerHook, RecencyNeighborHook
from opendg.loader import DGDataLoader
from opendg.nn import Time2Vec
from opendg.timedelta import TimeDeltaDG
from opendg.util.seed import seed_everything

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
        self.mlp_mixer = MLPMixer(
            num_tokens=num_tokens,
            num_channels=edge_dim,
            token_dim_expansion=token_dim_expansion,
            channel_dim_expansion=channel_dim_expansion,
            dropout=dropout,
        )
        self.output_layer = nn.Linear(
            in_features=edge_dim + node_dim, out_features=embed_dim
        )

    def forward(
        self, batch: DGBatch, node_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Temporary
        device = batch.src.device

        # Link Encoder
        edge_feat = batch.nbr_feats[0].to(device)
        nbr_time_feat = self.time_encoder(
            batch.nbr_times[0].to(device)
            - batch.time.unsqueeze(dim=1).repeat(3, 1).to(device)
        )
        z_link = torch.cat([edge_feat, nbr_time_feat], dim=-1)
        print(z_link.shape)
        z_link = self.projection_layer(z_link)
        print(z_link.shape)
        z_link = self.mlp_mixer(x=z_link)
        print(z_link.shape)
        z_link = torch.mean(z_link, dim=1)
        print(z_link.shape)
        input()

        # Node Encoder
        time_gap_node_feat = node_feat[batch.time_gap_node_nids]
        valid_time_gap_node_feats = time_gap_node_feat
        valid_time_gap_node_feats[batch.time_gap_node_mask == 0] = -1e10
        scores = torch.softmax(valid_time_gap_node_feats, dim=1).to(device)
        z_node = torch.mean(time_gap_node_feat * scores, dim=1)
        z_node = z_node + node_feat[torch.cat([batch.src, batch.dst, batch.neg])]

        # Link Decoder
        z = self.output_layer(torch.cat([z_link, z_node], dim=1))
        z_src, z_dst, z_neg = z.chunk(3, dim=0)
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
        num_tokens: int,  # per_graph_size
        num_channels: int,  # dims
        token_dim_expansion: float = 0.5,
        channel_dim_expansion: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(num_channels)
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
        z = self.channel_norm(out)  # (B, num_tokens, num_channels)
        z = self.channel_ff(z)  # (B, num_tokens, num_channels)
        out = z + out
        return out


class LinkPredictor(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin_src = nn.Linear(dim, dim)
        self.lin_dst = nn.Linear(dim, dim)
        self.lin_out = nn.Linear(dim, 1)

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_out(h).sigmoid().view(-1)


def train(
    loader: DGDataLoader,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    node_feat: torch.Tensor,
) -> float:
    model.train()
    total_loss = 0
    for batch in tqdm(loader):
        opt.zero_grad()
        pos_out, neg_out = model(batch, node_feat)
        loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        loss += F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
        loss.backward()
        opt.step()
        total_loss += float(loss)
    return total_loss


@torch.no_grad()
def eval(
    loader: DGDataLoader,
    model: nn.Module,
    metrics: Metric,
    node_feat: torch.Tensor,
) -> dict:
    model.eval()
    for batch in tqdm(loader):
        pos_out, neg_out = model(batch, node_feat)
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

train_dg = DGraph(
    args.dataset, time_delta=TimeDeltaDG('r'), split='train', device=args.device
)
val_dg = DGraph(
    args.dataset, time_delta=TimeDeltaDG('r'), split='valid', device=args.device
)
test_dg = DGraph(
    args.dataset, time_delta=TimeDeltaDG('r'), split='test', device=args.device
)


def _init_hooks(dg: DGraph, time_gap: int) -> List[DGHook]:
    # Graphmixer always uses 1-hop recent neighbors
    nbr_hook = RecencyNeighborHook(num_nbrs=[args.n_nbrs], num_nodes=dg.num_nodes)

    # Always produce negative edge prior to neighbor sampling for link prediction
    neg_hook = NegativeEdgeSamplerHook(low=0, high=dg.num_nodes)

    class GraphMixerHook:
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
            seed_nodes = torch.cat([batch.src, batch.dst, batch.neg])

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

            batch.time_gap_node_nids = nbr_nids
            batch.time_gap_node_mask = nbr_mask

            self._update(batch)
            return batch

        def _update(self, batch: DGBatch) -> None:
            for i in range(batch.src.size(0)):
                src_nbr = int(batch.src[i].item())
                dst_nbr = int(batch.dst[i].item())
                self._nbrs[src_nbr].append(dst_nbr)
                self._nbrs[dst_nbr].append(src_nbr)

    graph_mixer_hook = GraphMixerHook(time_gap, num_nodes=dg.num_nodes)
    return [neg_hook, nbr_hook, graph_mixer_hook]


train_loader = DGDataLoader(
    train_dg, hook=_init_hooks(train_dg, args.time_gap), batch_size=args.bsize
)
val_loader = DGDataLoader(
    val_dg, hook=_init_hooks(val_dg, args.time_gap), batch_size=args.bsize
)
test_loader = DGDataLoader(
    test_dg, hook=_init_hooks(test_dg, args.time_gap), batch_size=args.bsize
)


if train_dg.dynamic_node_feats_dim is not None:
    raise ValueError(
        'node features are not supported yet, make sure to incorporate them in the model'
    )

# TODO: add static node features to DGraph
args.node_dim = args.embed_dim
static_node_feats = torch.randn((test_dg.num_nodes, args.node_dim), device=args.device)

model = GraphMixer(
    embed_dim=args.embed_dim,
    time_dim=args.time_dim,
    num_tokens=args.n_nbrs,
    token_dim_expansion=float(args.token_dim_expansion),
    channel_dim_expansion=float(args.channel_dim_expansion),
    dropout=float(args.dropout),
    node_dim=args.node_dim,
    edge_dim=train_dg.edge_feats_dim | args.embed_dim,
).to(args.device)
opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

metrics = [BinaryAveragePrecision(), BinaryAUROC()]
val_metrics = MetricCollection(metrics, prefix='Validation')
test_metrics = MetricCollection(metrics, prefix='Test')

for epoch in range(1, args.epochs + 1):
    start_time = time.perf_counter()
    loss = train(train_loader, model, opt, static_node_feats)
    end_time = time.perf_counter()
    latency = end_time - start_time

    val_results = eval(val_loader, model, val_metrics, static_node_feats)
    val_metrics.reset()
    print(
        f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f} '
        + ' '.join(f'{k}={v.item():.4f}' for k, v in val_results.items())
    )

test_results = eval(test_loader, model, test_metrics, static_node_feats)
print(' '.join(f'{k}={v.item():.4f}' for k, v in test_results.items()))


# The below is not integrated code from the original authors' reference  implemenation
# class FeatEncode(nn.Module):
#    def __init__(self, time_dims, feat_dims, out_dims):
#        super().__init__()
#
#        self.time_encoder = TimeEncode(time_dims)
#        self.feat_encoder = nn.Linear(time_dims + feat_dims, out_dims)
#        self.reset_parameters()
#
#    def reset_parameters(self):
#        self.time_encoder.reset_parameters()
#        self.feat_encoder.reset_parameters()
#
#    def forward(self, edge_feats, edge_ts):
#        edge_time_feats = self.time_encoder(edge_ts)
#        x = torch.cat([edge_feats, edge_time_feats], dim=1)
#        return self.feat_encoder(x)
#
#
# class MLPMixer2(nn.Module):
#    def __init__(
#        self,
#        per_graph_size,
#        time_channels,
#        input_channels,
#        hidden_channels,
#        out_channels,
#        num_layers=2,
#        dropout=0.5,
#        token_expansion_factor=0.5,
#        channel_expansion_factor=4,
#    ):
#        super().__init__()
#        self.per_graph_size = per_graph_size
#        self.num_layers = num_layers
#
#        # input & output classifer
#        self.feat_encoder = FeatEncode(time_channels, input_channels, hidden_channels)
#        self.layernorm = nn.LayerNorm(hidden_channels)
#        self.mlp_head = nn.Linear(hidden_channels, out_channels)
#
#        # inner layers
#        self.mixer_blocks = torch.nn.ModuleList()
#        for _ in range(num_layers):
#            self.mixer_blocks.append(
#                MLPMixer(
#                    per_graph_size,
#                    hidden_channels,
#                    token_expansion_factor,
#                    channel_expansion_factor,
#                    dropout,
#                )
#            )
#
#    def forward(self, edge_feats, edge_ts, batch_size, inds):
#        # x :     [ batch_size, graph_size, edge_dims+time_dims]
#        edge_time_feats = self.feat_encoder(edge_feats, edge_ts)
#        x = torch.zeros((batch_size * self.per_graph_size, edge_time_feats.size(1))).to(
#            edge_feats.device
#        )
#        x[inds] = x[inds] + edge_time_feats
#        x = torch.split(x, self.per_graph_size)
#        x = torch.stack(x)
#
#        # apply to original feats
#        for i in range(self.num_layers):
#            # apply to channel + feat dim
#            x = self.mixer_blocks[i](x)
#        x = self.layernorm(x)
#        x = torch.mean(x, dim=1)
#        x = self.mlp_head(x)
#        return x
#
#
# class EdgePredictor_per_node(torch.nn.Module):
#    def __init__(self, dim_in_time, dim_in_node):
#        super().__init__()
#
#        self.dim_in_time = dim_in_time
#        self.dim_in_node = dim_in_node
#
#        self.src_fc = torch.nn.Linear(dim_in_time + dim_in_node, 100)
#        self.dst_fc = torch.nn.Linear(dim_in_time + dim_in_node, 100)
#        self.out_fc = torch.nn.Linear(100, 1)
#
#    def forward(self, h, neg_samples=1):
#        num_edge = h.shape[0] // (neg_samples + 2)
#        h_src = self.src_fc(h[:num_edge])
#        h_pos_dst = self.dst_fc(h[num_edge : 2 * num_edge])
#        h_neg_dst = self.dst_fc(h[2 * num_edge :])
#        h_pos_edge = torch.nn.functional.relu(h_src + h_pos_dst)
#        h_neg_edge = torch.nn.functional.relu(h_src.tile(neg_samples, 1) + h_neg_dst)
#        return self.out_fc(h_pos_edge), self.out_fc(h_neg_edge)
#
#
# class Mixer_per_node(nn.Module):
#    def __init__(self, mlp_mixer_configs, edge_predictor_configs):
#        super().__init__()
#
#        self.node_feats_dim = edge_predictor_configs['dim_in_node']
#        self.base_model = MLPMixer2(**mlp_mixer_configs)
#        self.edge_predictor = EdgePredictor_per_node(**edge_predictor_configs)
#
#    def forward(self, model_inputs, has_temporal_neighbors, neg_samples, node_feats):
#        x = self.base_model(*model_inputs)
#        if self.node_feats_dim > 0:
#            x = torch.cat([x, node_feats], dim=1)
#
#        pred_pos, pred_neg = self.edge_predictor(x, neg_samples=neg_samples)
#        pos_mask, neg_mask = self.pos_neg_mask(has_temporal_neighbors, neg_samples)
#        loss = F.binary_cross_entropy_with_logits(pred_pos, torch.ones_like(pred_pos))[
#            pos_mask
#        ].mean()
#        loss += F.binary_cross_entropy_with_logits(
#            pred_neg, torch.zeros_like(pred_neg)
#        )[neg_mask].mean()
#        return loss
#
#    def pos_neg_mask(self, mask, neg_samples):
#        num_edge = len(mask) // (neg_samples + 2)
#        src_mask = mask[:num_edge]
#        pos_dst_mask = mask[num_edge : 2 * num_edge]
#        neg_dst_mask = mask[2 * num_edge :]
#
#        pos_mask = [(i and j) for i, j in zip(src_mask, pos_dst_mask)]
#        neg_mask = [(i and j) for i, j in zip(src_mask * neg_samples, neg_dst_mask)]
#        return pos_mask, neg_mask
