import argparse
from os import stat
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
        self.act = nn.ReLU()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x = torch.cat([x1, x2], dim=1)
        h = self.fc2(self.act(self.fc1(x)))
        return h


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
        super().__init__()
        self.num_layers = num_layers
        self.num_nbrs = num_nbrs
        self.time_encoder = Time2Vec(time_dim=time_dim)

        self.attn, self.merge_layers = nn.ModuleList(), nn.ModuleList()
        for i in range(num_layers):
            in_dim = node_dim if i == 0 else embed_dim
            self.attn.append(
                TemporalAttention(
                    n_heads=n_heads,
                    node_dim=in_dim,
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
        self,
        src_ids: torch.Tensor,
        dst_ids: torch.Tensor,
        interact_times: torch.Tensor,
        batch: DGBatch,
        static_node_feats: torch.Tensor,
        is_negative=False,
        inference=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = batch.src.device

        def compute_embeddings(
            node_ids: torch.Tensor, node_times: torch.Tensor, hop: int, is_src=False
        ):
            num_nbrs = self.num_nbrs[-hop]  # recursing from hop = self.num_layers
            node_time_feat = self.time_encoder(torch.zeros_like(node_ids))
            node_raw_features = static_node_feats[node_ids]

            if hop == 0:
                return node_raw_features
            else:
                node_feat = compute_embeddings(node_ids, node_times, hop - 1, is_src)

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

                if inference:
                    if is_negative:
                        bsize = num_nbrs if node_ids.shape[0] == 999 else num_nbrs**2
                    else:
                        bsize = node_ids.shape[0] * num_nbrs
                else:
                    bsize = nbr_nids.shape[0] // 3

                def _split(x):
                    return x[0:bsize], x[bsize : 2 * bsize], x[2 * bsize :]

                idx = 0 if is_src else 2 if is_negative else 1
                nbr_node_ids = _split(nbr_nids)[idx]
                nbr_time = _split(nbr_times)[idx]
                nbr_edge_feat = _split(nbr_feat)[idx]

                nbr_node_ids = nbr_node_ids.reshape(node_ids.shape[0], -1)
                nbr_time = nbr_time.reshape(node_ids.shape[0], -1)

                # (B * num_nbrs, output_dim or node_feat_dim)
                nbr_feat = compute_embeddings(
                    nbr_node_ids.flatten(), nbr_time.flatten(), hop - 1, is_src
                )
                # (B, num_nbrs, output_dim or node_feat_dim)
                nbr_feat = nbr_feat.reshape(node_ids.shape[0], num_nbrs, -1)
                # (B, num_nbrs)
                delta_time = node_times[:, None] - nbr_time
                nbr_time_feat = self.time_encoder(delta_time)
                nbr_edge_feat = nbr_edge_feat.reshape(
                    node_ids.shape[0], -1, nbr_edge_feat.shape[-1]
                )
                out = self.attn[hop - 1](
                    node_feat=node_feat,
                    time_feat=node_time_feat,
                    nbr_node_feat=nbr_feat,
                    nbr_time_feat=nbr_time_feat,
                    edge_feat=nbr_edge_feat,
                    nbr_mask=nbr_node_ids,
                )
                return self.merge_layers[hop - 1](out, node_raw_features)

        if inference and is_negative:
            z_src = None
        else:
            z_src = compute_embeddings(
                node_ids=src_ids,
                node_times=interact_times,
                hop=self.num_layers,
                is_src=True,
            )
        z_dst = compute_embeddings(
            node_ids=dst_ids,
            node_times=interact_times,
            hop=self.num_layers,
            is_src=False,
        )
        return z_src, z_dst


class LinkPredictor(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2 * dim, dim)
        self.fc2 = nn.Linear(dim, 1)
        self.act = nn.ReLU()

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        h = self.fc1(torch.cat([z_src, z_dst], dim=1))
        h = h.relu()
        return self.fc2(h)


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
    idx, losses, metrics = 0, [], []
    for batch in tqdm(loader):
        opt.zero_grad()

        z_src, z_dst = encoder(
            src_ids=batch.src,
            dst_ids=batch.dst,
            interact_times=batch.time,
            batch=batch,
            static_node_feats=static_node_feats,
            is_negative=False,
        )
        _, z_neg_dst = encoder(
            src_ids=batch.src,
            dst_ids=batch.neg,
            interact_times=batch.time,
            static_node_feats=static_node_feats,
            batch=batch,
            is_negative=True,
        )

        pos_prob = decoder(z_src, z_dst).squeeze(dim=-1).sigmoid()
        neg_prob = decoder(z_src, z_neg_dst).squeeze(dim=-1).sigmoid()

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
        labels = labels.cpu().numpy()
        predicts = predicts.cpu().detach().numpy()
        d = {}
        d['ap'] = average_precision_score(y_true=labels, y_score=predicts)
        d['roc_auc'] = roc_auc_score(y_true=labels, y_score=predicts)
        metrics.append(d)
        if idx > 5:
            break
        idx += 1
    print(f'Epoch: {epoch + 1}, train loss: {np.mean(losses):.4f}')
    print(f'ap, {np.mean([x["ap"] for x in metrics]):.4f}')
    print(f'roc, {np.mean([x["roc_auc"] for x in metrics]):.4f}')
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
            batch_src_node_ids = np.asarray([batch.src[idx]]).reshape(-1)
            batch_dst_node_ids = np.asarray([batch.dst[idx]]).reshape(-1)
            batch_neg_dst_node_ids = np.asarray(neg_batch)
            batch_neg_src_node_ids = (
                batch.src[idx].repeat(len(batch_neg_dst_node_ids)).cpu().numpy()
            )
            batch_node_interact_times = torch.tensor([batch.time[idx]]).repeat(
                batch_dst_node_ids.shape[0]
            )
            neg_batch_node_interact_times = torch.tensor([batch.time[idx]]).repeat(
                batch_neg_dst_node_ids.shape[0]
            )
            z_src, z_dst = encoder(
                src_ids=torch.from_numpy(batch_src_node_ids),
                dst_ids=torch.from_numpy(batch_dst_node_ids),
                interact_times=batch_node_interact_times,
                batch=batch,
                static_node_feats=static_node_feats,
                is_negative=False,
                inference=True,
            )
            _, z_neg_dst = encoder(
                src_ids=torch.from_numpy(batch_neg_src_node_ids),
                dst_ids=torch.from_numpy(batch_neg_dst_node_ids),
                interact_times=neg_batch_node_interact_times,
                batch=batch,
                static_node_feats=static_node_feats,
                is_negative=True,
                inference=True,
            )
            z_neg_src = z_src.repeat(z_neg_dst.shape[0], 1)

            pos_prob = decoder(z_src, z_dst).squeeze(dim=-1).sigmoid()
            neg_prob = decoder(z_neg_src, z_neg_dst).squeeze(dim=-1).sigmoid()

            input_dict = {
                'y_pred_pos': pos_prob[0].detach().cpu().numpy(),
                'y_pred_neg': neg_prob.detach().cpu().numpy(),
                'eval_metric': [eval_metric],
            }
            perf = evaluator.eval(input_dict)[eval_metric]
            perf_list.append(perf)
            print(f'batch ID: {batch_id}, MRR, {perf}')
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
    start_time = time.perf_counter()
    loss = train(train_loader, static_node_feats, encoder, decoder, opt)
    end_time = time.perf_counter()
    latency = end_time - start_time

    SHARED_NBR_HOOK = RecencyNeighborHook(
        num_nbrs=args.n_nbrs,
        num_nodes=test_dg.num_nodes,
        edge_feats_dim=test_dg.edge_feats_dim,
    )
    print('filling up neighbor hook in preperation for validation')
    _, dst, _ = test_dg.edges
    min_dst, max_dst = int(dst.min()), int(dst.max())
    neg_hook = NegativeEdgeSamplerHook(low=int(dst.min()), high=int(dst.max()))
    foo_loader = DGDataLoader(
        train_dg, hook=[neg_hook, SHARED_NBR_HOOK], batch_size=2000, drop_last=False
    )
    for _ in tqdm(foo_loader):
        pass
    neg_hook = TGBNegativeEdgeSamplerHook(neg_sampler, split_mode='val')
    val_loader = DGDataLoader(val_dg, hook=[neg_hook, SHARED_NBR_HOOK], batch_size=1)
    val_results = eval(
        val_loader, static_node_feats, encoder, decoder, eval_metric, evaluator
    )
    exit()

    print(
        f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f} '
        + ' '.join(f'{k}={v:.4f}' for k, v in val_results.items())
    )

test_results = eval(test_loader, encoder, decoder, eval_metric, evaluator)
print(' '.join(f'{k}={v:.4f}' for k, v in test_results.items()))
