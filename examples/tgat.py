import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from opendg.graph import DGBatch, DGraph
from opendg.loader.neighbor_loader import DGBaseLoader, DGNeighborLoader
from opendg.nn import TemporalAttention
from opendg.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='TGAT Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument(
    '-d', '--dataset', type=str, required=True, default='tgbl-wiki', help='Dataset name'
)
parser.add_argument(
    '--gpu', type=int, default=-1, help='gpu device to use (or -1 for cpu)'
)
parser.add_argument(
    '--epochs', type=int, default=100, help='number of epochs (default: 100)'
)
parser.add_argument('--bsize', type=int, default=200, help='batch size (default: 200)')
parser.add_argument(
    '--lr', type=str, default=0.0001, help='learning rate (default: 1e-4)'
)
parser.add_argument(
    '--dropout', type=str, default=0.1, help='dropout rate (default: 0.1)'
)
parser.add_argument(
    '--n-layers', type=int, default=2, help='number of layers (default: 2)'
)
parser.add_argument(
    '--n-heads', type=int, default=2, help='number of attention heads (default: 2)'
)
parser.add_argument(
    '--n-nbrs', type=int, default=20, help='number of neighbors to sample (default: 20)'
)
parser.add_argument(
    '--time-dim',
    type=int,
    default=100,
    help='dimension of time features (default: 100)',
)
parser.add_argument(
    '--embed-dim', type=int, default=100, help='dimension of embeddings (default: 100)'
)
parser.add_argument(
    '--sampling',
    type=str,
    default='recent',
    choices=['recent', 'uniform'],
    help='sampling strategy (default: recent)',
)


class TGAT(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        time_dim: int,
        embed_dim: int,
        sampler: tg.TSampler,
        num_layers=2,
        n_heads=2,
        dropout=0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.sampler = sampler
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
        #head = batch.block(self.ctx)
        #for i in range(self.num_layers):
        #    tail = head if i == 0 else tail.next_block(include_dst=True)
        #    tail = tg.op.dedup(tail) if self.dedup else tail
        #    tail = tg.op.cache(self.ctx, tail.layer, tail)
        #    tail = self.sampler.sample(tail)

        #tg.op.preload(head, use_pin=True)
        #if tail.num_dst() > 0:
        #    tail.dstdata['h'] = tail.dstfeat()
        #    tail.srcdata['h'] = tail.srcfeat()
        #embeds = tg.op.aggregate(head, list(reversed(self.attn)), key='h')
        #src, dst, neg = batch.split_data(embeds)

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


class LinkPredTrainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        model_path: str,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.model_path = model_path

    def train(self, train_loader: DGBaseLoader, val_loader: DGBaseLoader) -> None:
        best_mrr = 0
        for e in range(self.epochs):
            print(f'epoch {e}:')
            self.model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                self.optimizer.zero_grad()
                pred_pos, pred_neg = self.model(batch)
                loss = self.criterion(pred_pos, torch.ones_like(pred_pos)) +
                loss += self.criterion(pred_neg, torch.zeros_like(pred_neg))
                loss.backward()
                self.optimizer.step()
                epoch_loss += float(loss)

            if mrr := self.eval(val_loader) > best_mrr:
                best_mrr = mrr
                torch.save(self.model.state_dict(), self.model_path)
            print(f'  loss:{epoch_loss:.4f} MRR:{mrr:.4f}')

    @torch.no_grad()
    def eval(self, loader: DGBaseLoader) -> float:
        self.model.eval()
        perf_list = []
        for batch in loader:
            prob_pos, prob_neg = self.model(batch)
            perf_list.append(prob_pos - prob_neg)  # TODO: MRR eval
        return np.mean(perf_list)

    def test(self, loader: DGBaseLoader) -> None:
        print('Testing...')
        self.model.load_state_dict(torch.load(self.model_path))
        mrr = self.eval(loader)
        print(f' MRR: {mrr:.4f}')


def run_tgat(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    train_dg = DGraph(args.dataset, split='train')
    val_dg = DGraph(args.dataset, split='valid')
    test_dg = DGraph(args.dataset, split='test')

    edge_dim, node_dim = 2, 3  # TODO: Get these properties from DGraph

    if args.sampling != 'uniform':
        raise NotImplementedError(f'Unsupported sampling: {args.sampling}')
    nbr_loader_args = {'num_nbrs': [args.n_nbrs], 'batch_size': args.batch_size}
    train_loader = DGNeighborLoader(train_dg, **nbr_loader_args)
    val_loader = DGNeighborLoader(val_dg, **nbr_loader_args)
    test_loader = DGNeighborLoader(test_dg, **nbr_loader_args)

    model = TGAT(
        node_dim=node_dim,
        edge_dim=edge_dim,
        time_dim=args.time_dim,
        embed_dim=args.embed_dim,
        sampler=sampler,
        num_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=float(args.dropout),
    )

    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu')
    model = model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    trainer = LinkPredTrainer(
        model,
        criterion,
        optimizer,
        epochs=args.epochs,
        model_path='',
    )
    trainer.train(train_loader, val_loader)
    trainer.test(test_loader)


if __name__ == '__main__':
    args = parser.parse_args()
    run_tgat(args)
