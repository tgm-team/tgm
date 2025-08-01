import argparse
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from tqdm import tqdm

from tgm import DGBatch, DGraph
from tgm.hooks import (
    DGHook,
    NegativeEdgeSamplerHook,
    NeighborSamplerHook,
    RecencyNeighborHook,
)
from tgm.loader import DGDataLoader
from tgm.nn import TemporalAttention, Time2Vec
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='TGAT Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', type=str, default=0.0001, help='learning rate')
parser.add_argument('--dropout', type=str, default=0.1, help='dropout rate')
parser.add_argument('--n-heads', type=int, default=2, help='number of attention heads')
parser.add_argument(
    '--n-nbrs',
    type=int,
    nargs='+',
    default=[20, 20],
    help='num sampled nbrs at each hop',
)
parser.add_argument('--time-dim', type=int, default=100, help='time encoding dimension')
parser.add_argument('--embed-dim', type=int, default=172, help='attention dimension')
parser.add_argument(
    '--sampling',
    type=str,
    default='recency',
    choices=['uniform', 'recency'],
    help='sampling strategy',
)


class TGAT(nn.Module):
    def __init__(
        self,
        edge_dim: int,
        time_dim: int,
        embed_dim: int,
        num_layers: int,
        n_heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.link_predictor = LinkPredictor(dim=embed_dim)
        self.time_encoder = Time2Vec(time_dim=time_dim)
        self.attn = nn.ModuleList(
            [
                TemporalAttention(
                    n_heads=n_heads,
                    node_dim=embed_dim,
                    edge_dim=edge_dim,
                    time_dim=time_dim,
                    out_dim=embed_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, batch: DGBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        device = batch.src.device
        z = torch.zeros(len(batch.unique_nids), self.embed_dim, device=device)

        for hop in reversed(range(self.num_layers)):
            seed_nodes = batch.nids[hop]
            nbrs = batch.nbr_nids[hop]
            nbr_mask = batch.nbr_mask[hop]
            if seed_nodes.numel() == 0:
                continue

            # TODO: Check and read static node features
            node_feat = STATIC_NODE_FEAT[seed_nodes]
            node_time_feat = self.time_encoder(torch.zeros_like(seed_nodes))

            # If next next hops embeddings exist, use them instead of raw features
            nbr_feat = STATIC_NODE_FEAT[nbrs]
            if hop < self.num_layers - 1:
                valid_nbrs = nbrs[nbr_mask.bool()]
                nbr_feat[nbr_mask.bool()] = z[batch.global_to_local(valid_nbrs)]

            delta_time = batch.times[hop][:, None] - batch.nbr_times[hop]
            delta_time = delta_time.masked_fill(~nbr_mask, 0)
            nbr_time_feat = self.time_encoder(delta_time)

            out = self.attn[hop](
                node_feat=node_feat,
                time_feat=node_time_feat,
                edge_feat=batch.nbr_feats[hop],
                nbr_node_feat=nbr_feat,
                nbr_time_feat=nbr_time_feat,
                nbr_mask=nbr_mask,
            )
            z[batch.global_to_local(seed_nodes)] = out

        z_src = z[batch.global_to_local(batch.src)]
        z_dst = z[batch.global_to_local(batch.dst)]
        z_neg = z[batch.global_to_local(batch.neg)]

        pos_out = self.link_predictor(z_src, z_dst)
        neg_out = self.link_predictor(z_src, z_neg)
        return pos_out, neg_out


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


def train(loader: DGDataLoader, model: nn.Module, opt: torch.optim.Optimizer) -> float:
    model.train()
    total_loss = 0
    for batch in tqdm(loader):
        opt.zero_grad()
        pos_out, neg_out = model(batch)
        loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        loss += F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
        loss.backward()
        opt.step()
        total_loss += float(loss)
    return total_loss


@torch.no_grad()
def eval(loader: DGDataLoader, model: nn.Module, metrics: Metric) -> dict:
    model.eval()
    for batch in tqdm(loader):
        pos_out, neg_out = model(batch)
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

train_dg = DGraph(args.dataset, split='train', device=args.device)
val_dg = DGraph(args.dataset, split='val', device=args.device)
test_dg = DGraph(args.dataset, split='test', device=args.device)

# TODO: Read from graph
NUM_NODES, NODE_FEAT_DIM = test_dg.num_nodes, args.embed_dim
STATIC_NODE_FEAT = torch.randn((NUM_NODES, NODE_FEAT_DIM), device=args.device)


def _init_hooks(dg: DGraph, sampling_type: str) -> List[DGHook]:
    if sampling_type == 'uniform':
        nbr_hook = NeighborSamplerHook(num_nbrs=args.n_nbrs)
    elif sampling_type == 'recency':
        nbr_hook = RecencyNeighborHook(
            num_nbrs=args.n_nbrs,
            num_nodes=dg.num_nodes,
            edge_feats_dim=dg.edge_feats_dim,
        )
    else:
        raise ValueError(f'Unknown sampling type: {args.sampling}')

    # Always produce negative edge prior to neighbor sampling for link prediction
    neg_hook = NegativeEdgeSamplerHook(low=0, high=dg.num_nodes)
    return [neg_hook, nbr_hook]


test_loader = DGDataLoader(
    test_dg,
    hook=_init_hooks(test_dg, args.sampling),
    batch_size=args.bsize,
)

model = TGAT(
    edge_dim=train_dg.edge_feats_dim or args.embed_dim,
    time_dim=args.time_dim,
    embed_dim=train_dg.static_node_feats_dim or args.embed_dim,
    num_layers=len(args.n_nbrs),
    n_heads=args.n_heads,
    dropout=float(args.dropout),
).to(args.device)
opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

metrics = [BinaryAveragePrecision(), BinaryAUROC()]
val_metrics = MetricCollection(metrics, prefix='Validation')
test_metrics = MetricCollection(metrics, prefix='Test')

for epoch in range(1, args.epochs + 1):
    # TODO: Need a clean way to clear nbr state across epochs
    train_loader = DGDataLoader(
        train_dg,
        hook=_init_hooks(test_dg, args.sampling),
        batch_size=args.bsize,
    )
    val_loader = DGDataLoader(
        val_dg,
        hook=_init_hooks(test_dg, args.sampling),
        batch_size=args.bsize,
    )
    start_time = time.perf_counter()
    loss = train(train_loader, model, opt)
    end_time = time.perf_counter()
    latency = end_time - start_time

    val_results = eval(val_loader, model, val_metrics)
    val_metrics.reset()

    print(
        f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f} '
        + ' '.join(f'{k}={v.item():.4f}' for k, v in val_results.items())
    )

test_results = eval(test_loader, model, test_metrics)
print(' '.join(f'{k}={v.item():.4f}' for k, v in test_results.items()))
