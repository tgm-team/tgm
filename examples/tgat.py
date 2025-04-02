import argparse
from pprint import pprint
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torchmetrics.retrieval import RetrievalHitRate, RetrievalMRR
from tqdm import tqdm

from opendg.graph import DGBatch, DGraph
from opendg.hooks import NeighborSamplerHook
from opendg.loader import DGDataLoader
from opendg.nn import TemporalAttention, Time2Vec
from opendg.util.perf import Usage
from opendg.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='TGAT Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--gpu', type=int, default=-1, help='gpu to use (or -1 for cpu)')
parser.add_argument('--lr', type=str, default=0.0001, help='learning rate')
parser.add_argument('--dropout', type=str, default=0.1, help='dropout rate')
parser.add_argument('--n-heads', type=int, default=2, help='number of attention heads')
parser.add_argument('--n-nbrs', type=int, default=[20], help='num sampled nbrs')
parser.add_argument('--time-dim', type=int, default=100, help='time encoding dimension')
parser.add_argument('--embed-dim', type=int, default=100, help='attention dimension')
parser.add_argument(
    '--sampling',
    type=str,
    default='uniform',
    choices=['uniform'],
    help='sampling strategy',
)


class TGAT(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        time_dim: int,
        embed_dim: int,
        num_layers: int,
        n_heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.link_predictor = LinkPredictor(dim=embed_dim)
        self.time_encoder = Time2Vec(time_dim=time_dim)
        self.attn = nn.ModuleList(
            [
                TemporalAttention(
                    n_heads=n_heads,
                    node_dim=node_dim if i == 0 else embed_dim,
                    edge_dim=edge_dim,
                    time_dim=time_dim,
                    out_dim=embed_dim,
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )

        # Temporary
        self.embed_dim = embed_dim

    def forward(self, batch: DGBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(batch.src)

        # TODO: Get these from the batch
        num_nbrs = 20
        nbr_nodes = batch.src.unsqueeze(dim=1).repeat(1, num_nbrs)
        nbr_times = batch.time.unsqueeze(dim=1).repeat(1, num_nbrs).float()
        nbr_edge_feats = batch.edge_feats.unsqueeze(dim=1).repeat(1, num_nbrs, 1)

        def _recursive_forward(
            src: torch.Tensor, time: torch.Tensor, layer: int
        ) -> torch.Tensor:
            if layer == 0:
                return torch.zeros((*src.shape, self.embed_dim))
            return self.attn[layer - 1](
                node_feat=_recursive_forward(src, time, layer - 1),
                nbr_node_feat=_recursive_forward(nbr_nodes, nbr_times, layer - 1),
                time_feat=self.time_encoder(torch.zeros(batch_size)),
                nbr_time_feat=self.time_encoder(nbr_times - time[:, None]),
                edge_feat=nbr_edge_feats,
                nbr_mask=nbr_nodes >= 0,
            )

        # TODO: Make this a single forward pass (add src/dst/neg mask as in TGN)
        z_src = _recursive_forward(batch.src, batch.time, layer=1)
        z_dst = _recursive_forward(batch.dst, batch.time, layer=1)
        z_neg = _recursive_forward(batch.src, batch.time, layer=1)  # TODO: batch.neg

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
def eval(loader: DGDataLoader, model: nn.Module, metrics: Metric) -> None:
    model.eval()
    for batch in tqdm(loader):
        pos_out, neg_out = model(batch)
        y_pred = torch.cat([pos_out, neg_out], dim=0).float()
        y_true = torch.cat(
            [torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0
        ).long()
        indexes = torch.zeros(y_pred.size(0), dtype=torch.long)
        metrics(y_pred, y_true, indexes=indexes)
    pprint(metrics.compute())


args = parser.parse_args()
seed_everything(args.seed)

train_dg = DGraph(args.dataset, split='train')
val_dg = DGraph(args.dataset, split='valid')
test_dg = DGraph(args.dataset, split='test')

train_loader = DGDataLoader(
    train_dg,
    hook=NeighborSamplerHook(num_nbrs=args.n_nbrs),
    batch_size=args.bsize,
)
val_loader = DGDataLoader(
    val_dg,
    hook=NeighborSamplerHook(num_nbrs=args.n_nbrs),
    batch_size=args.bsize,
)
test_loader = DGDataLoader(
    test_dg,
    hook=NeighborSamplerHook(num_nbrs=args.n_nbrs),
    batch_size=args.bsize,
)

device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu')
model = TGAT(
    node_dim=train_dg.node_feats_dim or args.embed_dim,  # TODO: verify
    edge_dim=train_dg.edge_feats_dim or args.embed_dim,
    time_dim=args.time_dim,
    embed_dim=args.embed_dim,
    num_layers=len(args.n_nbrs),
    n_heads=args.n_heads,
    dropout=float(args.dropout),
).to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)

metrics = [BinaryAveragePrecision(), BinaryAUROC(), RetrievalHitRate(), RetrievalMRR()]
val_metrics = MetricCollection(metrics, prefix='Validation')
test_metrics = MetricCollection(metrics, prefix='Test')

with Usage(prefix='TGAT Training'):
    for epoch in range(1, args.epochs + 1):
        loss = train(train_loader, model, opt)
        pprint(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
        eval(val_loader, model, val_metrics)
        eval(test_loader, model, test_metrics)
        val_metrics.reset()
        test_metrics.reset()
