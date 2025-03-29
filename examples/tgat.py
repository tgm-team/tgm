import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from opendg.graph import DGBatch, DGraph
from opendg.loader import DGNeighborLoader
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
        # TODO: TGAT Multi-hop forward pass
        # src, dst, time = batch.src, batch.dst, batch.time
        # node_feats, edge_feats = batch.node_feats, batch.edge_feats
        # out = self.attn(node_feats, time_feat, edge_feat, nbr_node_feat, nbr_time_feat, nbr_mask)
        # pos_prob = self.link_predictor(self.src_embed, self.dst_embed)
        # neg_prob = self.link_predictor(self.src_embed, self.neg_embed)

        z_src = torch.rand(len(batch.src), self.embed_dim)
        z_dst = torch.rand(len(batch.src), self.embed_dim)
        z_neg = torch.rand(len(batch.src), self.embed_dim)

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
        return self.lin_out(h)


def train(
    loader: DGNeighborLoader, model: nn.Module, opt: torch.optim.Optimizer
) -> float:
    model.train()
    total_loss = 0
    for batch in loader:
        opt.zero_grad()
        pos_out, neg_out = model(batch)
        loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        loss += F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
        loss.backward()
        opt.step()
        total_loss += float(loss)
    return total_loss


@torch.no_grad()
def eval(loader: DGNeighborLoader, model: nn.Module) -> float:
    model.eval()
    mrrs = []
    for batch in loader:
        pos_out, neg_out = model(batch)
        mrrs.append(0)  # TODO: MRR eval
    return sum(mrrs) / len(mrrs)


args = parser.parse_args()
seed_everything(args.seed)

train_dg = DGraph(args.dataset, split='train')
val_dg = DGraph(args.dataset, split='valid')
test_dg = DGraph(args.dataset, split='test')

train_loader = DGNeighborLoader(train_dg, num_nbrs=args.n_nbrs, batch_size=args.bsize)
val_loader = DGNeighborLoader(val_dg, num_nbrs=args.n_nbrs, batch_size=args.bsize)
test_loader = DGNeighborLoader(test_dg, num_nbrs=args.n_nbrs, batch_size=args.bsize)

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

with Usage(prefix='TGAT Training'):
    for epoch in range(1, args.epochs + 1):
        loss = train(train_loader, model, opt)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
        val_mrr = eval(val_loader, model)
        test_mrr = eval(test_loader, model)
        print(f'Val MRR: {val_mrr:.4f}')
        print(f'Test MRR: {test_mrr:.4f}')
