import argparse
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from tqdm import tqdm

from tgm import DGBatch, DGData, DGraph
from tgm.hooks import NegativeEdgeSamplerHook
from tgm.loader import DGDataLoader
from tgm.nn.recurrent import GCLSTM
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='GCLSTM Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
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


class GCLSTM_Model(nn.Module):
    def __init__(self, node_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.encoder = RecurrentGCN(node_dim=node_dim, embed_dim=embed_dim, K=1)
        self.decoder = LinkPredictor(embed_dim)

    def forward(
        self,
        batch: DGBatch,
        node_feat: torch.Tensor,
        h_0: torch.Tensor | None = None,
        c_0: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, ...]:
        edge_index = torch.stack([batch.src, batch.dst], dim=0)
        edge_weight = batch.edge_weight if hasattr(batch, 'edge_weight') else None  # type: ignore
        z, h_0, c_0 = self.encoder(node_feat, edge_index, edge_weight, h_0, c_0)
        z_src = z[batch.global_to_local(batch.src)]
        z_dst = z[batch.global_to_local(batch.dst)]
        z_neg = z[batch.global_to_local(batch.neg)]
        pos_out = self.decoder(z_src, z_dst)
        neg_out = self.decoder(z_src, z_neg)
        return pos_out, neg_out, h_0, c_0


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_dim: int, embed_dim: int, K=1) -> None:
        super().__init__()
        self.recurrent = GCLSTM(in_channels=node_dim, out_channels=embed_dim, K=K)
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        h: torch.Tensor | None,
        c: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, ...]:
        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
        h = F.relu(h_0)
        h = self.linear(h)
        return h, h_0, c_0


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
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    model.train()
    total_loss = 0
    h_0, c_0 = None, None
    for batch in tqdm(loader):
        # TODO: Consider skipping empty batches natively, when iterating by time (instead of events)
        if not len(batch.src):
            continue

        opt.zero_grad()
        pos_out, neg_out, h_0, c_0 = model(batch, node_feat, h_0, c_0)
        loss = F.mse_loss(pos_out, torch.ones_like(pos_out))
        loss += F.mse_loss(neg_out, torch.zeros_like(neg_out))
        loss.backward()
        opt.step()
        total_loss += float(loss)
        h_0, c_0 = h_0.detach(), c_0.detach()
    return total_loss, h_0, c_0


@torch.no_grad()
def eval(
    loader: DGDataLoader,
    model: nn.Module,
    metrics: Metric,
    node_feat: torch.Tensor,
    h_0: torch.Tensor | None = None,
    c_0: torch.Tensor | None = None,
) -> Tuple[dict, torch.Tensor, torch.Tensor]:
    model.eval()
    for batch in tqdm(loader):
        # TODO: Consider skipping empty batches natively, when iterating by time (instead of events)
        if not len(batch.src):
            continue

        pos_out, neg_out, h_0, c_0 = model(batch, node_feat, h_0, c_0)
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
    return metrics.compute(), h_0, c_0


args = parser.parse_args()
seed_everything(args.seed)

train_data = DGData.from_tgb(args.dataset, split='train')
train_dg = DGraph(train_data, train_data.time_delta, device=args.device).discretize(
    args.time_gran
)

val_data = DGData.from_tgb(args.dataset, split='val')
val_dg = DGraph(val_data, val_data.time_delta, device=args.device).discretize(
    args.time_gran
)

test_data = DGData.from_tgb(args.dataset, split='test')
test_dg = DGraph(test_data, test_data.time_delta, device=args.device).discretize(
    args.time_gran
)

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

if train_dg.dynamic_node_feats is not None:
    raise ValueError(
        'node features are not supported yet, make sure to incorporate them in the model'
    )

# TODO: add static node features to DGraph
args.node_dim = args.embed_dim
static_node_feats = torch.randn((test_dg.num_nodes, args.node_dim), device=args.device)

model = GCLSTM_Model(node_dim=args.node_dim, embed_dim=args.embed_dim).to(args.device)

opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
metrics = [BinaryAveragePrecision(), BinaryAUROC()]
val_metrics = MetricCollection(metrics, prefix='Validation')
test_metrics = MetricCollection(metrics, prefix='Test')

for epoch in range(1, args.epochs + 1):
    start_time = time.perf_counter()
    loss, h_0, c_0 = train(train_loader, model, opt, static_node_feats)
    end_time = time.perf_counter()
    latency = end_time - start_time

    val_results, h_0, c_0 = eval(
        val_loader, model, val_metrics, static_node_feats, h_0, c_0
    )
    val_metrics.reset()

    print(
        f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f} '
        + ' '.join(f'{k}={v.item():.4f}' for k, v in val_results.items())
    )

test_results, h_0, c_0 = eval(test_loader, model, test_metrics, static_node_feats)
print(' '.join(f'{k}={v.item():.4f}' for k, v in test_results.items()))
