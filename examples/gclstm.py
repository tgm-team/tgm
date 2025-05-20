"""Adapted from: https://github.com/shenyangHuang/UTG/blob/main/dtdg_gclstm.py."""

import argparse
from pprint import pprint
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from tqdm import tqdm

from opendg.graph import DGBatch, DGraph
from opendg.hooks import NegativeEdgeSamplerHook
from opendg.loader import DGDataLoader
from opendg.nn.recurrent import GCLSTM
from opendg.util.perf import Usage
from opendg.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='GCLSTM Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--embed-dim', type=int, default=128, help='embedding dimension')
parser.add_argument('--num_layers', type=int, default=2, help='number of GCN layers')
parser.add_argument('--device', type=str, default='cpu', help='training device')
parser.add_argument('--device_id', type=str, default='0', help='device id for gpu')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument(
    '--time_gran',
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
        edge_index = torch.stack([batch.src, batch.dst], dim=0).to(node_feat.device)
        if hasattr(batch, 'edge_weight'):
            edge_weight = batch.edge_weight
        else:
            edge_weight = torch.ones(edge_index.size(1), dtype=torch.float).to(
                node_feat.device
            )
        z, h_0, c_0 = self.encoder(node_feat, edge_index, edge_weight, h_0, c_0)
        z_src, z_dst, z_neg = z[batch.src], z[batch.dst], z[batch.neg]  # type: ignore
        pos_out = self.decoder(z_src, z_dst)
        neg_out = self.decoder(z_src, z_neg)
        return pos_out, neg_out, h_0, c_0


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_dim: int, embed_dim: int, K=1) -> None:
        super().__init__()
        self.recurrent = GCLSTM(in_channels=node_dim, out_channels=embed_dim, K=K)
        self.linear = torch.nn.Linear(embed_dim, embed_dim)

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
) -> Tuple[float, DGBatch]:
    model.train()
    total_loss = 0
    input_batch, h_0, c_0 = None, None, None
    for batch in tqdm(loader):
        opt.zero_grad()
        if input_batch is None:
            input_batch = batch
        pos_out, neg_out, h_0, c_0 = model(input_batch, node_feat, h_0, c_0)
        loss = F.mse_loss(pos_out, torch.ones_like(pos_out))
        loss += F.mse_loss(neg_out, torch.zeros_like(neg_out))
        loss.backward()
        opt.step()
        total_loss += float(loss)
        input_batch = batch
        h_0, c_0 = h_0.detach(), c_0.detach()
    return total_loss, input_batch, h_0, c_0


@torch.no_grad()
def eval(
    loader: DGDataLoader,
    model: nn.Module,
    metrics: Metric,
    input_batch: DGBatch,
    node_feat: torch.Tensor,
    h_0: torch.Tensor | None = None,
    c_0: torch.Tensor | None = None,
) -> Tuple[float, DGBatch]:
    model.eval()
    for batch in tqdm(loader):
        pos_out, neg_out, h_0, c_0 = model(batch, node_feat, h_0, c_0)
        y_pred = torch.cat([pos_out, neg_out], dim=0).float()
        y_true = torch.cat(
            [torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0
        ).long()
        y_true = y_true.to(y_pred.device)
        indexes = torch.zeros(y_pred.size(0), dtype=torch.long, device=y_pred.device)
        metrics(y_pred, y_true, indexes=indexes)
        input_batch = batch
    pprint(metrics.compute())
    return input_batch, h_0, c_0


args = parser.parse_args()
seed_everything(args.seed)

if int(args.device_id) >= 0 and torch.cuda.is_available():
    args.device = torch.device('cuda'.format(args.device_id))
    print('INFO: using gpu:{} to train the model'.format(args.device_id))
else:
    args.device = torch.device('cpu')
    print('INFO: using cpu to train the model')

train_dg = DGraph(args.dataset, split='train')
val_dg = DGraph(args.dataset, split='valid')
test_dg = DGraph(args.dataset, split='test')

train_loader = DGDataLoader(
    train_dg,
    hook=NegativeEdgeSamplerHook(low=0, high=train_dg.num_nodes),
    batch_size=1,
    batch_unit=args.time_gran,
)
val_loader = DGDataLoader(
    val_dg,
    hook=NegativeEdgeSamplerHook(low=0, high=val_dg.num_nodes),
    batch_size=1,
    batch_unit=args.time_gran,
)
test_loader = DGDataLoader(
    test_dg,
    hook=NegativeEdgeSamplerHook(low=0, high=test_dg.num_nodes),
    batch_size=1,
    batch_unit=args.time_gran,
)

if train_dg.node_feats_dim is not None:
    raise ValueError(
        'node features are not supported yet, make sure to incorporate them in the model'
    )

# TODO: add static node features to DGraph
args.node_dim = args.embed_dim
static_node_feats = torch.randn((test_dg.num_nodes, args.node_dim), device=args.device)

model = GCLSTM_Model(node_dim=args.node_dim, embed_dim=args.embed_dim).to(args.device)

opt = torch.optim.Adam(model.parameters(), lr=args.lr)
metrics = [BinaryAveragePrecision(), BinaryAUROC()]
val_metrics = MetricCollection(metrics, prefix='Validation')
test_metrics = MetricCollection(metrics, prefix='Test')

with Usage(prefix='GCLSTM Training'):
    for epoch in range(1, args.epochs + 1):
        loss, input_batch, h_0, c_0 = train(train_loader, model, opt, static_node_feats)
        pprint(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
        input_batch, h_0, c_0 = eval(
            val_loader, model, val_metrics, input_batch, static_node_feats, h_0, c_0
        )
        input_batch, h_0, c_0 = eval(
            test_loader, model, test_metrics, input_batch, static_node_feats, h_0, c_0
        )
        val_metrics.reset()
        test_metrics.reset()
