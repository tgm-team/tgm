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
from tgm.hooks import HookManager, NegativeEdgeSamplerHook
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
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    model.train()
    total_loss = 0
    h_0, c_0 = None, None
    for batch in tqdm(loader):
        opt.zero_grad()
        pos_out, neg_out, h_0, c_0 = model(batch, static_node_feats, h_0, c_0)
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
    static_node_feats: torch.Tensor,
    h_0: torch.Tensor,
    c_0: torch.Tensor,
    model: nn.Module,
    metrics: Metric,
) -> Tuple[dict, torch.Tensor, torch.Tensor]:
    model.eval()
    for batch in tqdm(loader):
        pos_out, neg_out, h_0, c_0 = model(batch, static_node_feats, h_0, c_0)
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


train_data, val_data, test_data = DGData.from_tgb(args.dataset).split()

train_data = train_data.discretize(args.time_gran)
val_data = val_data.discretize(args.time_gran)
test_data = test_data.discretize(args.time_gran)

train_dg = DGraph(train_data, device=args.device)
val_dg = DGraph(val_data, device=args.device)
test_dg = DGraph(test_data, device=args.device)

train_neg_hook = NegativeEdgeSamplerHook(
    low=int(train_dg.edges[1].min()), high=int(train_dg.edges[1].max())
)
val_neg_hook = NegativeEdgeSamplerHook(
    low=int(val_dg.edges[1].min()), high=int(val_dg.edges[1].max())
)
test_neg_hook = NegativeEdgeSamplerHook(
    low=int(test_dg.edges[1].min()), high=int(test_dg.edges[1].max())
)

hm = HookManager(keys=['train', 'val', 'test'])
hm.register('train', train_neg_hook)
hm.register('val', val_neg_hook)
hm.register('test', test_neg_hook)

train_loader = DGDataLoader(train_dg, batch_unit=args.batch_time_gran, hook_manager=hm)
val_loader = DGDataLoader(val_dg, batch_unit=args.batch_time_gran, hook_manager=hm)
test_loader = DGDataLoader(test_dg, batch_unit=args.batch_time_gran, hook_manager=hm)

if train_dg.static_node_feats is not None:
    static_node_feats = train_dg.static_node_feats
else:
    static_node_feats = torch.randn(
        (test_dg.num_nodes, args.node_dim), device=args.device
    )

model = GCLSTM_Model(node_dim=static_node_feats.shape[1], embed_dim=args.embed_dim).to(
    args.device
)

opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
metrics = [BinaryAveragePrecision(), BinaryAUROC()]
val_metrics = MetricCollection(metrics, prefix='Validation')
test_metrics = MetricCollection(metrics, prefix='Test')

for epoch in range(1, args.epochs + 1):
    with hm.activate('train'):
        start_time = time.perf_counter()
        loss, h_0, c_0 = train(train_loader, static_node_feats, model, opt)
        end_time = time.perf_counter()
        latency = end_time - start_time

    with hm.activate('val'):
        val_results, h_0, c_0 = eval(
            val_loader, static_node_feats, h_0, c_0, model, val_metrics
        )
        val_metrics.reset()

    print(
        f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f} '
        + ' '.join(f'{k}={v.item():.4f}' for k, v in val_results.items())
    )


with hm.activate('test'):
    test_results, h_0, c_0 = eval(
        test_loader, static_node_feats, h_0, c_0, model, test_metrics
    )
    print(' '.join(f'{k}={v.item():.4f}' for k, v in test_results.items()))
