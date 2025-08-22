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

from tgm.graph import DGBatch, DGraph
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
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        batch: DGBatch = None,
        is_negative=False,
        inference=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if inference and is_negative:
            z_src = None
        else:
            z_src = self.compute_embeddings(
                node_ids=src_node_ids,
                node_interact_times=node_interact_times,
                hop=self.num_layers,
                batch=batch,
                is_negative=is_negative,
                is_src=True,
                inference=inference,
            )
        z_dst = self.compute_embeddings(
            node_ids=dst_node_ids,
            node_interact_times=node_interact_times,
            hop=self.num_layers,
            batch=batch,
            is_negative=is_negative,
            is_src=False,
            inference=inference,
        )
        return z_src, z_dst

    def compute_embeddings(
        self,
        node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        hop: int,
        batch=None,
        is_negative=False,
        is_src=False,
        inference=False,
    ):
        device = STATIC_NODE_FEAT.device
        num_nbrs = self.num_nbrs[-hop]  # recursing from hop = self.num_layers
        node_time_features = self.time_encoder(
            torch.zeros(node_interact_times.shape).to(device)
        )
        node_raw_features = STATIC_NODE_FEAT[torch.from_numpy(node_ids)]

        if hop == 0:
            return node_raw_features
        else:
            node_conv_features = self.compute_embeddings(
                node_ids=node_ids,
                node_interact_times=node_interact_times,
                hop=hop - 1,
                batch=batch,
                is_negative=is_negative,
                is_src=is_src,
                inference=inference,
            )

            if len(node_ids) in {batch.src.numel(), batch.neg.numel()}:  # hop 0
                nbr_nids, nbr_feats = batch.nbr_nids[0].flatten(), batch.nbr_feats[0]
                nbr_times = batch.nbr_times[0].flatten()
            elif len(node_ids) in {  # hop 1
                batch.src.numel() * num_nbrs,
                batch.neg.numel() * num_nbrs,
            }:
                nbr_nids, nbr_feats = batch.nbr_nids[1].flatten(), batch.nbr_feats[1]
                nbr_times = batch.nbr_times[1].flatten()
            else:
                assert False

            if inference:
                if is_negative:
                    bsize = num_nbrs if node_ids.shape[0] == 999 else num_nbrs**2
                else:
                    bsize = node_ids.shape[0] * num_nbrs

                def _split(x):
                    return x[0:bsize], x[bsize : 2 * bsize], x[2 * bsize :]
            else:

                def _split(x):
                    return torch.chunk(x, 3)

            src_nbr_nids, dst_nbr_nids, neg_nbr_nids = _split(nbr_nids)
            src_nbr_times, dst_nbr_times, neg_nbr_times = _split(nbr_times)

            if inference:
                nbr_feats = nbr_feats.reshape(-1, nbr_feats.size(-1))
            src_nbr_feats, dst_nbr_feats, neg_nbr_feats = _split(nbr_feats)

            def _to_np(x):
                return x.cpu().numpy()

            if is_src:
                nbr_node_ids, nbr_time = _to_np(src_nbr_nids), _to_np(src_nbr_times)
                nbr_edge_feat = src_nbr_feats
            elif is_negative:
                nbr_node_ids, nbr_time = _to_np(neg_nbr_nids), _to_np(neg_nbr_times)
                nbr_edge_feat = neg_nbr_feats
            else:
                nbr_node_ids, nbr_time = _to_np(dst_nbr_nids), _to_np(dst_nbr_times)
                nbr_edge_feat = dst_nbr_feats

            nbr_node_ids = nbr_node_ids.reshape(node_ids.shape[0], -1)
            nbr_time = nbr_time.reshape(node_ids.shape[0], -1)

            # (batch_size * num_nbrs, output_dim or node_feat_dim)
            nbr_feat = self.compute_embeddings(
                node_ids=nbr_node_ids.flatten(),
                node_interact_times=nbr_time.flatten(),
                hop=hop - 1,
                batch=batch,
                is_negative=is_negative,
                is_src=is_src,
                inference=inference,
            )

            # (batch_size, num_nbrs, output_dim or node_feat_dim)
            nbr_feat = nbr_feat.reshape(node_ids.shape[0], num_nbrs, -1)

            # (batch_size, num_nbrs)
            delta_time = node_interact_times[:, np.newaxis] - nbr_time
            delta_time = torch.from_numpy(delta_time).float().to(device)
            nbr_time_feat = self.time_encoder(delta_time)
            if inference:
                nbr_edge_feat = nbr_edge_feat.reshape(
                    node_ids.shape[0], -1, nbr_edge_feat.shape[-1]
                )

            out = self.attn[hop - 1](
                node_feat=node_conv_features,
                time_feat=node_time_features,
                nbr_node_feat=nbr_feat,
                nbr_time_feat=nbr_time_feat,
                edge_feat=nbr_edge_feat,
                nbr_mask=torch.from_numpy(nbr_node_ids).to(device),
            )
            return self.merge_layers[hop - 1](out, node_raw_features)


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
            src_node_ids=batch.src.cpu().numpy(),
            dst_node_ids=batch.dst.cpu().numpy(),
            node_interact_times=batch.time.cpu().numpy(),
            batch=batch,
            is_negative=False,
        )
        _, z_neg_dst = encoder(
            src_node_ids=batch.src.cpu().numpy(),
            dst_node_ids=batch.neg.cpu().numpy(),
            node_interact_times=batch.time.cpu().numpy(),
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
                src_node_ids=batch_src_node_ids,
                dst_node_ids=batch_dst_node_ids,
                node_interact_times=batch_node_interact_times.cpu().numpy(),
                batch=batch,
                is_negative=False,
                inference=True,
            )
            _, z_neg_dst = encoder(
                src_node_ids=batch_neg_src_node_ids,
                dst_node_ids=batch_neg_dst_node_ids,
                node_interact_times=neg_batch_node_interact_times.cpu().numpy(),
                batch=batch,
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

train_dg = DGraph(args.dataset, split='train', device=args.device)
val_dg = DGraph(args.dataset, split='val', device=args.device)
test_dg = DGraph(args.dataset, split='test', device=args.device)

# TODO: Read from graph
NUM_NODES, NODE_FEAT_DIM = test_dg.num_nodes, 1
STATIC_NODE_FEAT = torch.zeros((NUM_NODES, NODE_FEAT_DIM), device=args.device)


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
    node_dim=NODE_FEAT_DIM,
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
    loss = train(train_loader, encoder, decoder, opt)
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
    val_results = eval(val_loader, encoder, decoder, eval_metric, evaluator)
    exit()

    print(
        f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f} '
        + ' '.join(f'{k}={v:.4f}' for k, v in val_results.items())
    )

test_results = eval(test_loader, encoder, decoder, eval_metric, evaluator)
print(' '.join(f'{k}={v:.4f}' for k, v in test_results.items()))
