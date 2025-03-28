import argparse
from typing import List, Tuple

import torch
import torch.nn as nn

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
        self.edge_predictor = LinkPredictor(dim=embed_dim)
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
        # Helper vector to map global node indices to local ones.
        # self.assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

    def forward(self, batch: DGBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        # batch.neg = torch.randint(0, 1337, (batch.dst.size(0),), dtype=torch.long)
        # batch.n_id = torch.cat([batch.src, batch.dst, batch.neg]).unique()

        # nbrs = self.nbrs[batch.n_id]
        # nodes = batch.n_id.view(-1, 1).repeat(1, self.size)
        # e_id = self.e_id[batch.n_id]

        ## Relabel node indices.
        # n_id = torch.cat([batch.n_id, nbrs]).unique()
        # self._assoc[n_id] = torch.arange(n_id.size(0))
        # nbrs, nodes = self._assoc[nbrs], self._assoc[nodes]
        # edge_idx = torch.stack([nbrs, nodes])

        # batch.assoc[n_id] = torch.arange(n_id.size(0))

        # batch.src_mask = batch.assoc[batch.src]
        # batch.dst_mask = batch.assoc[batch.dst]
        # batch.neg_mask = batch.assoc[batch.neg]

        # z = gnn(edge_index, data.time[e_id], data.edge_feats[e_id])
        # z = self.attn(
        #    node_feats, time_feats, edge_feats, nbr_node_feat, nbr_time_feat, nbr_mask
        # )
        pos_out = self.edge_predictor(z[batch.src_mask], z[batch.dst_mask])
        neg_out = self.edge_predictor(z[batch.src_mask], z[batch.neg_mask])
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
            loss = criterion(pred_pos, torch.ones_like(pred_pos))
            loss += criterion(pred_neg, torch.ones_like(pred_neg))
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
        pos_out, neg_out = model(batch)
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

    with Usage(prefix='TGAT Training'):
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
