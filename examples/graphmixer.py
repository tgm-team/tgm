import argparse
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from tqdm import tqdm

from opendg.graph import DGBatch, DGraph
from opendg.hooks import NegativeEdgeSamplerHook
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
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--dropout', type=str, default=0.1, help='dropout rate')
parser.add_argument('--n-layers', type=int, default=2, help='number of MLP layers')
parser.add_argument('--embed-dim', type=int, default=128, help='embedding dimension')
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
parser.add_argument(
    '--time-gran',
    type=str,
    default='h',
    help='time granularity to operate on for snapshots',
)


class GraphMixer(nn.Module):
    def __init__(
        self,
        time_dim: int,
        num_tokens: int,
        num_layers: int = 2,
        token_dim_expansion: float = 0.5,
        channel_dim_expansion: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        node_feat_dim = self.node_raw_features.shape[1]
        edge_feat_dim = self.edge_raw_features.shape[1]

        # in GraphMixer, the time encoding function is not trainable
        self.time_encoder = Time2Vec(time_dim=time_dim, parameter_requires_grad=False)
        self.projection_layer = nn.Linear(edge_feat_dim + time_dim, edge_feat_dim)
        self.mlp_mixers = nn.ModuleList(
            [
                MLPMixer(
                    num_tokens=num_tokens,
                    num_channels=edge_feat_dim,
                    token_dim_expansion=token_dim_expansion,
                    channel_dim_expansion=channel_dim_expansion,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_layer = nn.Linear(
            in_features=edge_feat_dim + node_feat_dim, out_features=node_feat_dim
        )

    def forward(self, batch: DGBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        def _compute_node_temporal_embeddings(
            node_ids: torch.Tensor,
            node_interact_times: torch.Tensor,
            num_neighbors: int = 20,
            time_gap: int = 2000,
        ) -> torch.Tensor:
            nbr_nodes, nbr_edges, nbr_times = (
                self.neighbor_sampler.get_historical_neighbors(
                    node_ids=node_ids,
                    node_interact_times=node_interact_times,
                    num_neighbors=num_neighbors,
                )
            )
            # (B, num_nbrs, edge_dim)
            edge_feats = self.edge_raw_features[torch.from_numpy(nbr_edges)]
            nbr_time_feats = self.time_encoder(
                timestamps=torch.from_numpy(
                    node_interact_times[:, np.newaxis] - nbr_times
                )
                .float()
                .to(self.device)
            )  # (B, num_nbrs, time_dim)

            # set the time features to all zeros for the padded timestamp
            nbr_time_feats[torch.from_numpy(nbr_nodes == 0)] = 0.0

            # (B, num_nbrs, edge_dim + time_dim)
            feats = torch.cat([edge_feats, nbr_time_feats], dim=-1)
            feats = self.projection_layer(feats)  # (B, num_nbrs, num_channels)
            for mlp_mixer in self.mlp_mixers:
                feats = mlp_mixer(x=feats)  # (B, num_nbrs, num_channels)
            feats = torch.mean(feats, dim=1)  # (B, num_channels)

            # get temporal neighbors of nodes, including neighbor ids
            time_gap_nbr_node_ids, _, _ = (
                self.neighbor_sampler.get_historical_neighbors(
                    node_ids=node_ids,
                    node_interact_times=node_interact_times,
                    num_neighbors=time_gap,
                )
            )  # (B, time_gap)
            time_gap_nbr_node_feats = self.node_raw_features[
                torch.from_numpy(time_gap_nbr_node_ids)
            ]  # (B, time_gap, node_feat_dim)
            valid_time_gap_nbr_mask = torch.from_numpy(
                (time_gap_nbr_node_ids > 0).astype(np.float32)
            )  # (B, time_gap)
            valid_time_gap_nbr_mask[valid_time_gap_nbr_mask == 0] = -1e10
            scores = torch.softmax(valid_time_gap_nbr_mask, dim=1).to(self.device)
            time_gap_nbr_node_agg_feats = torch.mean(
                time_gap_nbr_node_feats * scores.unsqueeze(dim=-1), dim=1
            )  # (B, node_feat_dim), average over the time_gap nbrs
            output_node_feats = (
                time_gap_nbr_node_agg_feats
                + self.node_raw_features[torch.from_numpy(node_ids)]
            )  # (B, node_feat_dim), add features of nodes in node_ids

            # (B, node_feat_dim)
            z = self.output_layer(torch.cat([feats, output_node_feats], dim=1))
            return z

        z_src = _compute_node_temporal_embeddings(
            node_ids=batch.src,
            node_interact_times=batch.time,
        )
        z_dst = _compute_node_temporal_embeddings(
            node_ids=batch.dst,
            node_interact_times=batch.time,
        )
        return z_src, z_dst


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
        self.token_feedforward = FeedForwardNet(
            input_dim=num_tokens,
            dim_expansion_factor=token_dim_expansion,
            dropout=dropout,
        )
        self.channel_norm = nn.LayerNorm(num_channels)
        self.channel_feedforward = FeedForwardNet(
            input_dim=num_channels,
            dim_expansion_factor=channel_dim_expansion,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mix tokens
        z = self.token_norm(x.permute(0, 2, 1))  # (B, num_channels, num_tokens)
        z = self.token_feedforward(z).permute(0, 2, 1)  # (B, num_tokens, num_channels)
        out = z + x

        # mix channels
        z = self.channel_norm(out)  # (B, num_tokens, num_channels)
        z = self.channel_feedforward(z)  # (B, num_tokens, num_channels)
        out = z + out
        return out


def train(
    loader: DGDataLoader,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    node_feat: torch.Tensor,
) -> float:
    model.train()
    total_loss = 0
    for batch in tqdm(loader):
        # TODO: Consider skipping empty batches natively, when iterating by time (instead of events)
        if not len(batch.src):
            continue

        opt.zero_grad()
        pos_out, neg_out = model(batch, node_feat)
        loss = F.mse_loss(pos_out, torch.ones_like(pos_out))
        loss += F.mse_loss(neg_out, torch.zeros_like(neg_out))
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
        # TODO: Consider skipping empty batches natively, when iterating by time (instead of events)
        if not len(batch.src):
            continue

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

train_dg = DGraph(args.dataset, time_delta=TimeDeltaDG('s'), split='train')
val_dg = DGraph(args.dataset, time_delta=TimeDeltaDG('s'), split='valid')
test_dg = DGraph(args.dataset, time_delta=TimeDeltaDG('s'), split='test')

train_loader = DGDataLoader(
    train_dg,
    hook=NegativeEdgeSamplerHook(low=0, high=train_dg.num_nodes),
    batch_unit=args.time_gran,
)
val_loader = DGDataLoader(
    val_dg,
    hook=NegativeEdgeSamplerHook(low=0, high=val_dg.num_nodes),
    batch_unit=args.time_gran,
)
test_loader = DGDataLoader(
    test_dg,
    hook=NegativeEdgeSamplerHook(low=0, high=test_dg.num_nodes),
    batch_unit=args.time_gran,
)

if train_dg.node_feats_dim is not None:
    raise ValueError(
        'node features are not supported yet, make sure to incorporate them in the model'
    )

# TODO: add static node features to DGraph
args.node_dim = args.embed_dim
static_node_feats = torch.randn((test_dg.num_nodes, args.node_dim), device=args.device)

model = GraphMixer(
    time_dim=args.time_dim,
    num_tokens=args.num_tokens,
    num_layers=args.num_layers,
    token_dim_expansion=float(args.token_dim_expansion),
    channel_dim_expansion=float(args.channel_dim_expansion),
    dropout=float(args.dropout),
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
