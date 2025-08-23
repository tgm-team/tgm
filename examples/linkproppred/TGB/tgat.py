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
        num_nbrs=None,
    ) -> None:
        """In this implementation, the node embedding dimension must be the same as hidden embedding dimension."""
        super().__init__()
        self.num_layers = num_layers
        self.num_nbrs = num_nbrs
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

        def get_embeddings(
            node_ids: torch.Tensor, node_times: torch.Tensor, hop: int, is_src: bool
        ):
            if hop == 0:
                return static_node_feat[node_ids]

            node_feat = get_embeddings(node_ids, node_times, hop - 1, is_src)
            node_time_feat = self.time_encoder(torch.zeros_like(node_ids))

            num_nbrs = self.num_nbrs[-hop]  # recursing from hop = self.num_layers
            layer_0_nums = [batch.src.numel(), batch.neg.numel()]
            layer_1_nums = [x * num_nbrs for x in layer_0_nums]
            if len(node_ids) in layer_0_nums:
                i = 0
            elif len(node_ids) in layer_1_nums:
                i = 1
            else:
                assert False
            nbr_nids = batch.nbr_nids[i].flatten()
            nbr_times = batch.nbr_times[i].flatten()
            nbr_feat = batch.nbr_feats[i].reshape(-1, batch.nbr_feats[i].size(-1))

            if batch.dst.numel() != batch.neg.numel() and batch.is_negative:
                bsize = num_nbrs if i == 0 else num_nbrs**2
            else:
                bsize = len(node_ids) * num_nbrs

            def _split(x):
                return x[0:bsize], x[bsize : 2 * bsize], x[2 * bsize :]

            idx = 0 if is_src else 2 if batch.is_negative else 1

            nbr_node_ids = _split(nbr_nids)[idx].reshape(node_ids.shape[0], -1)
            nbr_time = _split(nbr_times)[idx].reshape(node_ids.shape[0], -1)
            nbr_edge_feat = _split(nbr_feat)[idx].reshape(
                node_ids.shape[0], -1, nbr_feat.shape[-1]
            )

            # (B, num_nbrs, output_dim or node_feat_dim)
            nbr_feat = get_embeddings(
                nbr_node_ids.flatten(), nbr_time.flatten(), hop - 1, is_src
            ).reshape(node_ids.shape[0], num_nbrs, -1)

            delta_time = node_times[:, None] - nbr_time
            nbr_time_feat = self.time_encoder(delta_time)

            out = self.attn[hop - 1](
                node_feat=node_feat,
                time_feat=node_time_feat,
                nbr_node_feat=nbr_feat,
                nbr_time_feat=nbr_time_feat,
                edge_feat=nbr_edge_feat,
                nbr_mask=nbr_node_ids,
            )
            return self.merge_layers[hop - 1](out, static_node_feat[node_ids])

        z_src = get_embeddings(batch.src_ids, batch.time, self.num_layers, is_src=True)
        z_dst = get_embeddings(batch.dst_ids, batch.time, self.num_layers, is_src=False)
        return z_src, z_dst


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
    idx, losses, rocs, aps = 0, [], [], []
    for batch in tqdm(loader):
        opt.zero_grad()
        batch.src_ids, batch.dst_ids, batch.is_negative = batch.src, batch.dst, False
        z_src, z_dst = encoder(batch, static_node_feats)

        batch.dst_ids, batch.is_negative = batch.neg, True
        _, z_neg_dst = encoder(batch, static_node_feats)

        pos_prob = decoder(z_src, z_dst)
        neg_prob = decoder(z_src, z_neg_dst)

        loss_func = nn.BCELoss()
        predicts = torch.cat([pos_prob, neg_prob], dim=0)
        labels = torch.cat(
            [torch.ones_like(pos_prob), torch.zeros_like(neg_prob)], dim=0
        )
        loss = loss_func(input=predicts, target=labels)
        loss.backward()
        opt.step()
        total_loss += float(loss)
        losses.append(loss.item())
        labels, predicts = labels.cpu().numpy(), predicts.cpu().detach().numpy()
        aps.append(average_precision_score(y_true=labels, y_score=predicts))
        rocs.append(roc_auc_score(y_true=labels, y_score=predicts))
        if idx > 5:
            break
        idx += 1
    print(f'Epoch: {epoch + 1}, train loss: {np.mean(losses):.4f}')
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
        for idx, neg_batch in enumerate(batch.neg_batch_list):
            batch.src_ids = torch.tensor([batch.src[idx]])
            batch.dst_ids = torch.tensor([batch.dst[idx]])
            batch.time = torch.tensor([batch.time[idx]]).repeat(len(batch.dst_ids))
            batch.is_negative = False
            z_src, z_dst = encoder(batch, static_node_feats)

            batch.dst_ids = torch.tensor(neg_batch)
            batch.is_negative = True
            _, z_neg_dst = encoder(batch, static_node_feats)
            z_neg_src = z_src.repeat(z_neg_dst.shape[0], 1)

            pos_prob = decoder(z_src, z_dst)
            neg_prob = decoder(z_neg_src, z_neg_dst)

            input_dict = {
                'y_pred_pos': pos_prob[0].detach().cpu().numpy(),
                'y_pred_neg': neg_prob.detach().cpu().numpy(),
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
    num_nbrs=args.n_nbrs,
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
