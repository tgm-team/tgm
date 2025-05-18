import argparse
from pprint import pprint
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator
from tqdm import tqdm

from opendg.graph import DGBatch, DGraph
from opendg.hooks import (
    HookManager,
    NegativeEdgeSamplerHook,
    RecencyNeighborHook,
    TGBNeighborSamplerHook,
)
from opendg.loader import DGDataLoader
from opendg.nn import TemporalAttention, Time2Vec
from opendg.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='TGAT Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--lr', type=str, default=0.0001, help='learning rate')
parser.add_argument('--dropout', type=str, default=0.1, help='dropout rate')
parser.add_argument('--n-heads', type=int, default=2, help='number of attention heads')
parser.add_argument('--n-nbrs', type=int, default=[20], help='num sampled nbrs')
parser.add_argument('--time-dim', type=int, default=100, help='time encoding dimension')
parser.add_argument('--embed-dim', type=int, default=100, help='attention dimension')
parser.add_argument(
    '--sampling',
    type=str,
    default='recency',
    choices=['recency'],
    help='sampling strategy, currently only recency is supported',
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
        # TODO: Go back to recursive embedding for multi-hop
        hop = 0
        node_feat = torch.zeros((*batch.nids[hop].shape, self.embed_dim))
        nbr_node_feat = torch.zeros((*batch.nbr_nids[hop].shape, self.embed_dim))
        time_feat = self.time_encoder(torch.zeros(len(batch.nids[hop])))
        nbr_time_feat = self.time_encoder(
            batch.nbr_times[hop] - batch.time.unsqueeze(dim=1).repeat(3, 1)
        )
        z = self.attn[hop](
            node_feat=node_feat,
            nbr_node_feat=nbr_node_feat,
            time_feat=time_feat,
            nbr_time_feat=nbr_time_feat,
            edge_feat=batch.nbr_feats[hop],
            nbr_mask=batch.nbr_mask[hop],
        )
        return z


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
    decoder: nn.Module,
    opt: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0
    for batch in tqdm(loader):
        opt.zero_grad()
        z = model(batch)
        z_src, z_dst, z_neg = z.chunk(3, dim=0)
        pos_out = decoder(z_src, z_dst)
        neg_out = decoder(z_src, z_neg)
        loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        loss += F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
        loss.backward()
        opt.step()
        total_loss += float(loss)
    return total_loss


@torch.no_grad()
def eval(
    loader: DGDataLoader,
    model: nn.Module,
    metric: str,
    evaluator: object,
) -> None:
    r"""Debug TGB MRR data loading."""
    model.eval()
    perf_list = []
    for batch in tqdm(loader):
        z = model(batch)
        neg_batch_list = batch.neg_batch_list
        for idx, neg_batch in enumerate(neg_batch_list):
            pos_src = batch.src[idx]
            pos_dst = batch.dst[idx]
            src = torch.full((1 + len(neg_batch),), pos_src, device=batch.src.device)
            dst = torch.tensor(
                np.concatenate(
                    ([np.array([pos_dst.cpu().numpy()[idx]]), np.array(neg_batch)]),
                    axis=0,
                ),
                device=batch.dst.device,
                dtype=torch.long,
            )
            y_pred = decoder(z[src], z[dst])
            # compute MRR
            input_dict = {
                'y_pred_pos': np.array([y_pred[0, :].squeeze(dim=-1).cpu()]),
                'y_pred_neg': np.array(y_pred[1:, :].squeeze(dim=-1).cpu()),
                'eval_metric': [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])
    perf_metrics = float(torch.tensor(perf_list).mean())
    pprint(f'MRR Performance: {perf_metrics:.4f}')


args = parser.parse_args()
seed_everything(args.seed)

# * setting up tgb neg sampler
dataset = PyGLinkPropPredDataset(name=args.dataset, root='datasets')
metric = dataset.eval_metric
evaluator = Evaluator(name=args.dataset)
neg_sampler = dataset.negative_sampler
dataset.load_val_ns()
dataset.load_test_ns()

train_dg = DGraph(args.dataset, split='train')
val_dg = DGraph(args.dataset, split='valid')
test_dg = DGraph(args.dataset, split='test')

# * define hook manager
if args.sampling == 'recency':
    #! train hook works, actually using hook manager with random for val and test works.
    train_hook = HookManager(
        [
            NegativeEdgeSamplerHook(
                low=0, high=train_dg.num_nodes, neg_sampling_ratio=1.0
            ),
            RecencyNeighborHook(num_nbrs=args.n_nbrs, num_nodes=train_dg.num_nodes),
        ]
    )

    #! some bug with node id, num_nodes is not accurate
    val_hook = HookManager(
        [
            TGBNeighborSamplerHook(neg_sampler=neg_sampler, split_mode='val'),
            RecencyNeighborHook(num_nbrs=args.n_nbrs, num_nodes=test_dg.num_nodes),
        ]
    )
    test_hook = HookManager(
        [
            TGBNeighborSamplerHook(neg_sampler=neg_sampler, split_mode='test'),
            RecencyNeighborHook(num_nbrs=args.n_nbrs, num_nodes=test_dg.num_nodes),
        ]
    )
elif args.sampling == 'uniform':
    raise ValueError('Uniform sampling is not supported yet with HookManager')
else:
    raise ValueError(f'Unknown sampling type: {args.sampling}')

train_loader = DGDataLoader(train_dg, hook=train_hook, batch_size=args.bsize)
val_loader = DGDataLoader(val_dg, hook=val_hook, batch_size=args.bsize)
test_loader = DGDataLoader(test_dg, hook=test_hook, batch_size=args.bsize)

model = TGAT(
    node_dim=train_dg.node_feats_dim or args.embed_dim,  # TODO: verify
    edge_dim=train_dg.edge_feats_dim or args.embed_dim,
    time_dim=args.time_dim,
    embed_dim=args.embed_dim,
    num_layers=len(args.n_nbrs),
    n_heads=args.n_heads,
    dropout=float(args.dropout),
)
decoder = LinkPredictor(dim=args.embed_dim)
opt = torch.optim.Adam(set(model.parameters()) | set(decoder.parameters()), lr=args.lr)

# with Usage(prefix='TGAT Training'):
for epoch in range(1, args.epochs + 1):
    loss = train(train_loader, model, decoder, opt)
    pprint(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
    eval(val_loader, model, metric, evaluator)
    eval(test_loader, model, metric, evaluator)
