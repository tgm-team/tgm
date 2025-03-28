import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from opendg.graph import DGBatch, DGraph
from opendg.loader import DGNeighborLoader
from opendg.nn import TemporalAttention
from opendg.util.perf import Profiling
from opendg.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='TGAT Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
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
        self.edge_predictor = EdgePredictor(dim=embed_dim)
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
        # TODO: Temporary
        self.src_embed = torch.rand(embed_dim)
        self.dst_embed = torch.rand(embed_dim)
        self.neg_embed = torch.rand(embed_dim)

    def forward(self, batch: DGBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst, time = batch.src, batch.dst, batch.time
        node_feats, edge_feats = batch.node_feats, batch.edge_feats

        # TODO: Get the negative edges

        # TODO: TGAT Multi-hop forward pass
        # out = self.attn(node_feats, time_feat, edge_feat, nbr_node_feat, nbr_time_feat, nbr_mask)
        pos_prob = self.edge_predictor(self.src_embed, self.dst_embed)
        neg_prob = self.edge_predictor(self.src_embed, self.neg_embed)
        return pos_prob, neg_prob


class EdgePredictor(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.src_fc = nn.Linear(dim, dim)
        self.dst_fc = nn.Linear(dim, dim)
        self.out_fc = nn.Linear(dim, 1)

    def forward(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        h_src = self.src_fc(src)
        h_dst = self.dst_fc(dst)
        h_out = F.relu(h_src + h_dst)
        return self.out_fc(h_out)


def train(
    train_dg: DGraph,
    val_dg: DGraph,
    model: nn.Module,
    epochs: int,
    n_nbrs: List[int],
    bsize: int,
    lr: float,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_mrr = 0
    for e in range(epochs):
        print(f'epoch {e}:')
        model.train()
        e_loss = 0.0
        for batch in DGNeighborLoader(train_dg, n_nbrs, bsize):
            opt.zero_grad()
            pred_pos, pred_neg = model(batch)
            loss = criterion(pred_pos, torch.ones_like(pred_pos)) + criterion(
                pred_neg, torch.zeros_like(pred_neg)
            )
            loss.backward()
            opt.step()
            e_loss += float(loss)

        if (mrr := eval(val_dg, model, n_nbrs, bsize)) < best_mrr:
            best_mrr = mrr
        print(f'  loss:{e_loss:.4f} MRR:{mrr:.4f}')
    print(f'Best MRR: {best_mrr:.4f}')


@torch.no_grad()
def eval(dg: DGraph, model: nn.Module, n_nbrs: List[int], bsize: int) -> float:
    model.eval()
    perf_list = []
    for batch in DGNeighborLoader(dg, n_nbrs, bsize):
        prob_pos, prob_neg = model(batch)
        perf_list.append(0)  # TODO: MRR eval
    return sum(perf_list) / len(perf_list)


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    train_dg = DGraph(args.dataset, split='train')
    val_dg = DGraph(args.dataset, split='valid')

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

    with Profiling(filename='stat'):
        train(
            train_dg,
            val_dg,
            model,
            args.epochs,
            args.n_nbrs,
            args.bsize,
            float(args.lr),
        )


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
