import argparse
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator
from tqdm import tqdm

from tgm import DGBatch, DGData, DGraph
from tgm.hooks import (
    DGHook,
    NegativeEdgeSamplerHook,
    NeighborSamplerHook,
    RecencyNeighborHook,
    TGBNegativeEdgeSamplerHook,
)
from tgm.loader import DGDataLoader
from tgm.nn import TemporalAttention, Time2Vec
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='TGAT TGB Example',
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


class MergeLayer(nn.Module):
    def __init__(self, in_dim1: int, in_dim2: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim1 + in_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        h = self.fc1(torch.cat([x1, x2], dim=1))
        h = h.relu()
        return self.fc2(h)


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
        """In this implementation, the node embedding dimension must be the same as hidden embedding dimension."""
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.time_encoder = Time2Vec(time_dim=time_dim)
        self.attn, self.merge_layers = nn.ModuleList(), nn.ModuleList()
        for i in range(num_layers):
            self.attn.append(
                TemporalAttention(
                    n_heads=n_heads,
                    node_dim=node_dim if i == 0 else embed_dim,
                    edge_dim=edge_dim,
                    time_dim=time_dim,
                    dropout=dropout,
                )
            )
            self.merge_layers.append(
                MergeLayer(
                    in_dim1=self.attn[-1].out_dim,
                    in_dim2=node_dim,
                    hidden_dim=embed_dim,
                    output_dim=embed_dim,
                )
            )

    def forward(
        self, batch: DGBatch, static_node_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = batch.src.device
        z = {j: {} for j in range(self.num_layers + 1)}  # z[j][i] = z of nbr^i at hop j

        # Layer 0 (leaf nodes): z[0][i] = static_node_feat
        z[0][0] = static_node_feat[batch.nids[0]]
        for i in range(1, self.num_layers + 1):
            z[0][i] = static_node_feat[batch.nbr_nids[i - 1].flatten()]

        # Layers 1..H: aggregate z[j][i] = agg(z[j - 1][i], z[j - 1][i + 1])
        for j in range(1, self.num_layers + 1):
            for i in range(self.num_layers - j + 1):
                num_nodes = z[j - 1][i].size(0)
                num_nbr = batch.nbr_nids[j - 1].shape[-1]
                out = self.attn[j - 1](
                    node_feat=z[j - 1][i],
                    time_feat=self.time_encoder(torch.zeros(num_nodes, device=device)),
                    nbr_node_feat=z[j - 1][i + 1].reshape(num_nodes, num_nbr, -1),
                    edge_feat=batch.nbr_feats[i],
                    nbr_mask=batch.nbr_nids[i],
                    nbr_time_feat=self.time_encoder(
                        batch.times[i][:, None] - batch.nbr_times[i]
                    ),
                )
                z[j][i] = self.merge_layers[j - 1](out, z[0][i])

        return z[self.num_layers][0]


class LinkPredictor(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2 * dim, dim)
        self.fc2 = nn.Linear(dim, 1)

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        h = self.fc1(torch.cat([z_src, z_dst], dim=1))
        h = h.relu()
        return self.fc2(h).sigmoid().view(-1)


def train(
    loader: DGDataLoader,
    static_node_feats: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    opt: torch.optim.Optimizer,
) -> float:
    encoder.train()
    decoder.train()
    total_loss = 0
    idx, rocs, aps = 0, [], []
    for batch in tqdm(loader):
        opt.zero_grad()
        z = encoder(batch, static_node_feats)
        S, D = batch.src.numel(), batch.dst.numel()
        z_src, z_dst, z_neg = z[:D], z[S : S + D], z[S + D :]

        pos_out = decoder(z_src, z_dst)
        neg_out = decoder(z_src, z_neg)

        loss_func = nn.BCELoss()
        predicts = torch.cat([pos_out, neg_out], dim=0)
        labels = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)], dim=0)
        loss = loss_func(input=predicts, target=labels)
        loss.backward()
        opt.step()
        total_loss += float(loss)
        labels, predicts = labels.cpu().numpy(), predicts.cpu().detach().numpy()
        aps.append(average_precision_score(y_true=labels, y_score=predicts))
        rocs.append(roc_auc_score(y_true=labels, y_score=predicts))
        if idx > 5:
            break
        idx += 1
    print(f'Epoch: {epoch + 1}, train loss: {(total_loss / (idx + 1)):.4f}')
    print(f'ap, {np.mean(aps):.4f}\nroc, {np.mean([rocs]):.4f}')
    return total_loss


@torch.no_grad()
def eval(
    loader: DGDataLoader,
    static_node_feats: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    eval_metric: str,
    evaluator: Evaluator,
) -> dict:
    encoder.eval()
    decoder.eval()
    perf_list = []
    batch_id = 0
    for batch in tqdm(loader):
        z = encoder(batch, static_node_feats)
        id_map = {nid.item(): i for i, nid in enumerate(batch.nids[0])}
        for idx, neg_batch in enumerate(batch.neg_batch_list):
            dst_ids = torch.cat([batch.dst[idx].unsqueeze(0), neg_batch])
            src_ids = batch.src[idx].repeat(len(dst_ids))

            src_idx = torch.tensor([id_map[n.item()] for n in src_ids], device=z.device)
            dst_idx = torch.tensor([id_map[n.item()] for n in dst_ids], device=z.device)
            z_src = z[src_idx]
            z_dst = z[dst_idx]
            y_pred = decoder(z_src, z_dst)

            input_dict = {
                'y_pred_pos': y_pred[0].detach().cpu().numpy(),
                'y_pred_neg': y_pred[1:].detach().cpu().numpy(),
                'eval_metric': [eval_metric],
            }
            perf_list.append(evaluator.eval(input_dict)[eval_metric])
            print(f'batch ID: {batch_id}, MRR, {perf_list[-1]}')
        if batch_id > 20:
            break
        batch_id += 1
    metric_dict = {}
    metric_dict[eval_metric] = float(np.mean(perf_list))
    return metric_dict


args = parser.parse_args()
seed_everything(args.seed)

dataset = PyGLinkPropPredDataset(name=args.dataset, root='datasets')
eval_metric = dataset.eval_metric
neg_sampler = dataset.negative_sampler
evaluator = Evaluator(name=args.dataset)
dataset.load_val_ns()
dataset.load_test_ns()

train_data, val_data, test_data = DGData.from_tgb(args.dataset).split()

train_dg = DGraph(train_data, device=args.device)
val_dg = DGraph(val_data, device=args.device)
test_dg = DGraph(test_data, device=args.device)

if train_dg.static_node_feats is not None:
    static_node_feats = train_dg.static_node_feats
else:
    static_node_feats = torch.zeros((test_dg.num_nodes, 1), device=args.device)


def _init_hooks(
    dg: DGraph, sampling_type: str, neg_sampler: object, split_mode: str
) -> List[DGHook]:
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
    if split_mode in ['val', 'test']:
        neg_hook = TGBNegativeEdgeSamplerHook(neg_sampler, split_mode=split_mode)
    else:
        _, dst, _ = dg.edges
        min_dst, max_dst = int(dst.min()), int(dst.max())
        neg_hook = NegativeEdgeSamplerHook(low=min_dst, high=max_dst)
    return [neg_hook, nbr_hook]


test_loader = DGDataLoader(
    test_dg,
    hook=_init_hooks(test_dg, args.sampling, neg_sampler, 'test'),
    batch_size=args.bsize,
)


encoder = TGAT(
    node_dim=1,
    edge_dim=train_dg.edge_feats_dim,
    time_dim=args.time_dim,
    embed_dim=args.embed_dim,
    num_layers=len(args.n_nbrs),
    n_heads=args.n_heads,
    dropout=float(args.dropout),
).to(args.device)
decoder = LinkPredictor(dim=args.embed_dim).to(args.device)
opt = torch.optim.Adam(
    set(encoder.parameters()) | set(decoder.parameters()), lr=float(args.lr)
)

for epoch in range(1, args.epochs + 1):
    # TODO: Need a clean way to clear nbr state across epochs
    train_loader = DGDataLoader(
        train_dg,
        hook=_init_hooks(test_dg, args.sampling, neg_sampler, 'train'),
        batch_size=args.bsize,
    )
    shared_nbr = RecencyNeighborHook(
        test_dg.num_nodes, args.n_nbrs, test_dg.edge_feats_dim
    )
    _, dst, _ = test_dg.edges
    neg_hook = NegativeEdgeSamplerHook(low=int(dst.min()), high=int(dst.max()))
    foo_loader = DGDataLoader(train_dg, hook=[neg_hook, shared_nbr], batch_size=2000)
    for _ in tqdm(foo_loader):
        pass
    neg_hook = TGBNegativeEdgeSamplerHook(neg_sampler, split_mode='val')
    val_loader = DGDataLoader(val_dg, hook=[neg_hook, shared_nbr], batch_size=1)
    start_time = time.perf_counter()
    loss = train(train_loader, static_node_feats, encoder, decoder, opt)
    end_time = time.perf_counter()
    latency = end_time - start_time

    val_results = eval(
        val_loader, static_node_feats, encoder, decoder, eval_metric, evaluator
    )
    exit()

    print(
        f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f} '
        + ' '.join(f'{k}={v:.4f}' for k, v in val_results.items())
    )

test_results = eval(
    test_loader, static_node_feats, encoder, decoder, eval_metric, evaluator
)
print(' '.join(f'{k}={v:.4f}' for k, v in test_results.items()))
