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

from tgm import DGBatch, DGData, DGraph
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
    '--node-dim', type=int, default=100, help='node feat dimension if not provided'
)
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
        z_src = z[batch.global_to_local(batch.src)]
        z_dst = z[batch.global_to_local(batch.dst)]
        z_neg = z[batch.global_to_local(batch.neg)]
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
        loss = F.mse_loss(pos_out, torch.ones_like(pos_out))
        loss += F.mse_loss(neg_out, torch.zeros_like(neg_out))
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

train_data = train_data.discretize(args.time_gran)
val_data = val_data.discretize(args.time_gran)
test_data = test_data.discretize(args.time_gran)

train_dg = DGraph(train_data, device=args.device)
val_dg = DGraph(val_data, device=args.device)
test_dg = DGraph(test_data, device=args.device)

train_loader = DGDataLoader(
    train_dg,
    hook=NegativeEdgeSamplerHook(
        low=int(train_dg.edges[1].min()), high=int(train_dg.edges[1].max())
    ),
    batch_unit=args.batch_time_gran,
)
val_loader = DGDataLoader(
    val_dg,
    hook=NegativeEdgeSamplerHook(
        low=int(val_dg.edges[1].min()), high=int(val_dg.edges[1].max())
    ),
    batch_unit=args.batch_time_gran,
)
test_loader = DGDataLoader(
    test_dg,
    hook=NegativeEdgeSamplerHook(
        low=int(test_dg.edges[1].min()), high=int(test_dg.edges[1].max())
    ),
    batch_unit=args.batch_time_gran,
)

if train_dg.static_node_feats is not None:
    static_node_feats = train_dg.static_node_feats
else:
    static_node_feats = torch.randn(
        (test_dg.num_nodes, args.node_dim), device=args.device
    )


model = GCN(
    in_channels=static_node_feats.shape[1],
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
    loss = train(train_loader, static_node_feats, model, opt)
    end_time = time.perf_counter()
    latency = end_time - start_time

    val_results = eval(val_loader, static_node_feats, model, val_metrics)
    val_metrics.reset()
    print(
        f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f} '
        + ' '.join(f'{k}={v.item():.4f}' for k, v in val_results.items())
    )

test_results = eval(test_loader, static_node_feats, model, test_metrics)
print(' '.join(f'{k}={v.item():.4f}' for k, v in test_results.items()))
