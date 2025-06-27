import argparse
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from tqdm import tqdm

from tgm.graph import DGBatch, DGraph
from tgm.hooks import NegativeEdgeSamplerHook
from tgm.loader import DGDataLoader
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='GCN Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--dropout', type=str, default=0.1, help='dropout rate')
parser.add_argument('--n-layers', type=int, default=2, help='number of GCN layers')
parser.add_argument('--embed-dim', type=int, default=128, help='embedding dimension')
parser.add_argument(
    '--time-gran',
    type=str,
    default='s',
    help='raw time granularity for dataset',
)
parser.add_argument(
    '--batch-time-gran',
    type=str,
    default='h',
    help='time granularity to operate on for snapshots',
)


class GCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.encoder = GCNEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            out_channels=embed_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder = LinkPredictor(dim=embed_dim)

    def forward(
        self, batch: DGBatch, node_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_index = torch.stack([batch.src, batch.dst], dim=0)
        z = self.encoder(node_feat, edge_index)
        z_src, z_dst, z_neg = z[batch.src_idx], z[batch.dst_idx], z[batch.neg_idx]  # type: ignore
        pos_out = self.decoder(z_src, z_dst)
        neg_out = self.decoder(z_src, z_neg)
        return pos_out, neg_out


class GCNEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(GCNConv(in_channels, embed_dim, cached=True))
        self.bns.append(torch.nn.BatchNorm1d(embed_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(embed_dim, embed_dim, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(embed_dim))
        self.convs.append(GCNConv(embed_dim, out_channels, cached=True))

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


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

train_dg = DGraph(
    args.dataset,
    time_delta=args.time_gran,
    split='train',
    device=args.device,
)
val_dg = DGraph(
    args.dataset,
    time_delta=args.time_gran,
    split='val',
    device=args.device,
)
test_dg = DGraph(
    args.dataset,
    time_delta=args.time_gran,
    split='test',
    device=args.device,
)

train_loader = DGDataLoader(
    train_dg,
    hook=NegativeEdgeSamplerHook(low=0, high=train_dg.num_nodes),
    batch_unit=args.batch_time_gran,
)
val_loader = DGDataLoader(
    val_dg,
    hook=NegativeEdgeSamplerHook(low=0, high=val_dg.num_nodes),
    batch_unit=args.batch_time_gran,
)
test_loader = DGDataLoader(
    test_dg,
    hook=NegativeEdgeSamplerHook(low=0, high=test_dg.num_nodes),
    batch_unit=args.batch_time_gran,
)

if train_dg.dynamic_node_feats_dim is not None:
    raise ValueError(
        'node features are not supported yet, make sure to incorporate them in the model'
    )

# TODO: add static node features to DGraph
args.node_dim = args.embed_dim
static_node_feats = torch.randn((test_dg.num_nodes, args.node_dim), device=args.device)

model = GCN(
    in_channels=args.embed_dim,
    embed_dim=args.embed_dim,
    num_layers=args.n_layers,
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
