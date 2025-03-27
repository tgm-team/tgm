import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from opendg.graph import DGBatch, DGraph
from opendg.loader import DGBaseLoader, DGNeighborLoader
from opendg.nn import TemporalAttention
from opendg.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='TGAT Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--gpu', type=int, default=-1, help='gpu to use (or -1 for cpu)')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument('--lr', type=str, default=0.0001, help='learning rate')
parser.add_argument('--dropout', type=str, default=0.1, help='dropout rate')
parser.add_argument('--n-heads', type=int, default=2, help='number of attention heads')
parser.add_argument(
    '--n-nbrs', type=int, default=[20], help='num sampled nbrs'
)  # TODO: multi-hop
parser.add_argument('--time-dim', type=int, default=100, help='time encoding dimension')
parser.add_argument('--embed-dim', type=int, default=100, help='attention dimension')
parser.add_argument(
    '--dataset', type=str, required=True, default='tgbl-wiki', help='Dataset name'
)
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

    def forward(self, batch: DGBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        src, pos_dst, neg_dst, time, features = batch
        # head = batch.block(self.ctx)
        # for i in range(self.num_layers):
        #    tail = head if i == 0 else tail.next_block(include_dst=True)
        #    tail = tg.op.dedup(tail) if self.dedup else tail
        #    tail = tg.op.cache(self.ctx, tail.layer, tail)
        #    tail = self.sampler.sample(tail)
        # tg.op.preload(head, use_pin=True)
        # if tail.num_dst() > 0:
        #    tail.dstdata['h'] = tail.dstfeat()
        #    tail.srcdata['h'] = tail.srcfeat()
        # embeds = tg.op.aggregate(head, list(reversed(self.attn)), key='h')
        # src, dst, neg = batch.split_data(embeds)

        pos_prob = self.edge_predictor(src, pos_dst)
        neg_prob = self.edge_predictor(src, neg_dst)
        return pos_prob, neg_prob


class EdgePredictor(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.src_fc = nn.Linear(dim, dim)
        self.dst_fc = nn.Linear(dim, dim)
        self.out_fc = nn.Linear(dim, 1)
        self.act = nn.ReLU()

    def forward(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        h_src = self.src_fc(src)
        h_dst = self.dst_fc(dst)
        h_out = self.act(h_src + h_dst)
        return self.out_fc(h_out)


def train(
    train_loader: DGBaseLoader,
    val_loader: DGBaseLoader,
    model: nn.Module,
    criterion: nn.Module,
    opt: torch.optim.Optimizer,
    epochs: int,
) -> None:
    best_mrr = 0
    for e in range(epochs):
        print(f'epoch {e}:')
        model.train()
        e_loss = 0.0
        for batch in train_loader:
            opt.zero_grad()
            pred_pos, pred_neg = model(batch)
            loss = criterion(pred_pos, torch.ones_like(pred_pos)) + criterion(
                pred_neg, torch.zeros_like(pred_neg)
            )
            loss.backward()
            opt.step()
            e_loss += float(loss)

        if mrr := eval(val_loader, model) > best_mrr:
            best_mrr = mrr
        print(f'  loss:{e_loss:.4f} MRR:{mrr:.4f}')
    print(f'Best MRR: {best_mrr:.4f}')


@torch.no_grad()
def eval(loader: DGBaseLoader, model: nn.Module) -> float:
    model.eval()
    perf_list = []
    for batch in loader:
        prob_pos, prob_neg = model(batch)
        perf_list.append(prob_pos - prob_neg)  # TODO: MRR eval
    return np.mean(perf_list)


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    train_dg = DGraph(args.dataset, split='train')
    val_dg = DGraph(args.dataset, split='valid')

    # TODO: Would be convenient to have a dispatcher based on sampling_type
    nbr_loader_args = {'num_nbrs': args.n_nbrs, 'batch_size': args.bsize}
    train_loader = DGNeighborLoader(train_dg, **nbr_loader_args)
    val_loader = DGNeighborLoader(val_dg, **nbr_loader_args)

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
    criterion = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    train(train_loader, val_loader, model, criterion, opt, args.epochs)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
