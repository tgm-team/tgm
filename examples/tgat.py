import argparse
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor, nn

from opendg.graph import DGraph
from opendg.nn import TemporalAttention
from opendg.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='TGAT Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
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
    '--dim-time',
    type=int,
    default=100,
    help='dimension of time features (default: 100)',
)
parser.add_argument(
    '--dim-embed', type=int, default=100, help='dimension of embeddings (default: 100)'
)
parser.add_argument('--seed', type=int, default=-1, help='random seed to use')
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
        dim_embed: int,
        sampler: tg.TSampler,
        num_layers=2,
        num_heads=2,
        dropout=0.1,
        dedup: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.attn = nn.ModuleList(
            [
                TemporalAttention(
                    num_heads=num_heads,
                    node_dim=node_dim if i == 0 else dim_embed,
                    edge_dim=edge_dim,
                    time_dim=time_dim,
                    dim_out=dim_embed,
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )
        self.sampler = sampler
        self.edge_predictor = support.EdgePredictor(dim=dim_embed)
        self.dedup = dedup

    def forward(self, batch: tg.TBatch) -> Tensor:
        head = batch.block(self.ctx)
        for i in range(self.num_layers):
            tail = head if i == 0 else tail.next_block(include_dst=True)
            tail = tg.op.dedup(tail) if self.dedup else tail
            tail = tg.op.cache(self.ctx, tail.layer, tail)
            tail = self.sampler.sample(tail)

        tg.op.preload(head, use_pin=True)
        if tail.num_dst() > 0:
            tail.dstdata['h'] = tail.dstfeat()
            tail.srcdata['h'] = tail.srcfeat()
        embeds = tg.op.aggregate(head, list(reversed(self.attn)), key='h')
        del head
        del tail

        src, dst, neg = batch.split_data(embeds)
        scores = self.edge_predictor(src, dst)
        if batch.neg_nodes is not None:
            scores = (scores, self.edge_predictor(src, neg))
        return scores


device = support.make_device(args.gpu)
seed_everything(args.seed)

DATA: str = args.data
EPOCHS: int = args.epochs
BATCH_SIZE: int = args.bsize
LEARN_RATE: float = float(args.lr)
DROPOUT: float = float(args.dropout)
N_LAYERS: int = args.n_layers
N_HEADS: int = args.n_heads
N_NBRS: int = args.n_nbrs
DIM_TIME: int = args.time_dim
DIM_EMBED: int = args.dim_embed
SAMPLING: str = args.sampling


def make_device(gpu: int) -> torch.device:
    return torch.device(f'cuda:{gpu}' if gpu >= 0 else 'cpu')


def make_model_mem_path(model: str, prefix: str, data: str) -> str:
    Path(f'models/{model}').mkdir(parents=True, exist_ok=True)
    if prefix:
        return f'models/{model}/{prefix}-{data}-mem.pt'
    else:
        return f'models/{model}/{data}-mem-{time.time()}.pt'


class EdgePredictor(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.src_fc = nn.Linear(dim, dim)
        self.dst_fc = nn.Linear(dim, dim)
        self.out_fc = nn.Linear(dim, 1)
        self.act = nn.ReLU()

    def forward(self, src: Tensor, dst: Tensor) -> Tensor:
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
        neg_sampler: Callable,
        epochs: int,
        bsize: int,
        train_end: int,
        val_end: int,
        model_path: str,
        model_mem_path: Optional[str],
    ):
        self.g = ctx.graph
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.neg_sampler = neg_sampler
        self.epochs = epochs
        self.bsize = bsize
        self.train_end = train_end
        self.val_end = val_end
        self.model_path = model_path
        self.model_mem_path = model_mem_path

    def train(self):
        tt.csv_open('out-stats.csv')
        tt.csv_write_header()
        best_epoch = 0
        best_ap = 0
        for e in range(self.epochs):
            print(f'epoch {e}:')
            torch.cuda.synchronize()
            t_epoch = tt.start()

            self.ctx.train()
            self.model.train()
            if self.g.mem is not None:
                self.g.mem.reset()
            if self.g.mailbox is not None:
                self.g.mailbox.reset()

            epoch_loss = 0.0
            t_loop = tt.start()
            for batch in tg.iter_edges(self.g, size=self.bsize, end=self.train_end):
                t_start = tt.start()
                batch.neg_nodes = self.neg_sampler(len(batch))
                tt.t_prep_batch += tt.elapsed(t_start)

                t_start = tt.start()
                self.optimizer.zero_grad()
                pred_pos, pred_neg = self.model(batch)
                tt.t_forward += tt.elapsed(t_start)

                t_start = tt.start()
                loss = self.criterion(pred_pos, torch.ones_like(pred_pos))
                loss += self.criterion(pred_neg, torch.zeros_like(pred_neg))
                epoch_loss += float(loss)
                loss.backward()
                self.optimizer.step()
                tt.t_backward += tt.elapsed(t_start)
            tt.t_loop = tt.elapsed(t_loop)

            t_eval = tt.start()
            ap, auc = self.eval(start_idx=self.train_end, end_idx=self.val_end)
            tt.t_eval = tt.elapsed(t_eval)

            torch.cuda.synchronize()
            tt.t_epoch = tt.elapsed(t_epoch)
            if e == 0 or ap > best_ap:
                best_epoch = e
                best_ap = ap
                torch.save(self.model.state_dict(), self.model_path)
                if self.g.mem is not None:
                    torch.save(self.g.mem.backup(), self.model_mem_path)
            print(
                '  loss:{:.4f} val ap:{:.4f} val auc:{:.4f}'.format(epoch_loss, ap, auc)
            )
            tt.csv_write_line(epoch=e)
            tt.print_epoch()
            tt.reset_epoch()
        tt.csv_close()
        print('best model at epoch {}'.format(best_epoch))

    @torch.no_grad()
    def eval(self, start_idx: int, end_idx: int = None):
        self.ctx.eval()
        self.model.eval()
        val_aps = []
        val_auc = []
        for batch in tg.iter_edges(
            self.g, size=self.bsize, start=start_idx, end=end_idx
        ):
            size = len(batch)
            batch.neg_nodes = self.neg_sampler(size)
            prob_pos, prob_neg = self.model(batch)
            prob_pos = prob_pos.cpu()
            prob_neg = prob_neg.cpu()
            pred_score = torch.cat([prob_pos, prob_neg], dim=0).sigmoid()
            true_label = torch.cat([torch.ones(size), torch.zeros(size)])
            val_aps.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))
        return np.mean(val_aps), np.mean(val_auc)

    def test(self):
        print('loading saved checkpoint and testing model...')
        self.model.load_state_dict(torch.load(self.model_path))
        if self.g.mem is not None:
            self.g.mem.restore(torch.load(self.model_mem_path))
        t_test = tt.start()
        ap, auc = self.eval(start_idx=self.val_end)
        t_test = tt.elapsed(t_test)
        print('  test time:{:.2f}s AP:{:.4f} AUC:{:.4f}'.format(t_test, ap, auc))


def run_tgat(dataset: str) -> None:
    dg = DGraph(dataset)
    print(dg)

    dim_efeat = 0 if g.efeat is None else g.efeat.shape[1]
    dim_nfeat = g.nfeat.shape[1]
    g.set_compute(device)


    ### model
    sampler = tg.TSampler(N_NBRS, strategy=SAMPLING)
    model = TGAT(
        node_dim=dim_nfeat,
        edge_dim=dim_efeat,
        time_dim=DIM_TIME,
        dim_embed=DIM_EMBED,
        sampler=sampler,
        num_layers=N_LAYERS,
        num_heads=N_HEADS,
        dropout=DROPOUT,
    )
    model = model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)


    ### training
    train_end, val_end = support.data_split(g.num_edges(), 0.7, 0.15)
    neg_sampler = lambda size: np.random.randint(0, g.num_nodes(), size)

    trainer = support.LinkPredTrainer(
        model,
        criterion,
        optimizer,
        neg_sampler,
        args.epochs,
        args.batch_size,
        train_end,
        val_end,
        model_path='',
        None,
    )
    trainer.train()
    trainer.test()


def main() -> None:
    args = parser.parse_args()
    print(args)
    run_tgat(args.dataset)


if __name__ == '__main__':
    main()
