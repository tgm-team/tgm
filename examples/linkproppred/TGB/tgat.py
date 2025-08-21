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
    def __init__(
        self, input_dim1: int, input_dim2: int, hidden_dim: int, output_dim: int
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim1 + input_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor):
        x = torch.cat([input_1, input_2], dim=1)
        h = self.fc2(self.act(self.fc1(x)))
        return h


class TGAT(nn.Module):
    def __init__(
        self,
        node_raw_features: np.ndarray,
        edge_dim: int,
        time_dim: int,
        embed_dim: int,
        num_layers: int,
        n_heads: int = 2,
        dropout: float = 0.1,
        device: str = 'cpu',
        num_nbrs=None,
    ) -> None:
        super().__init__()

        self.num_nbrs = num_nbrs
        self.node_raw_features = torch.from_numpy(
            node_raw_features.astype(np.float32)
        ).to(device)

        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = edge_dim
        self.num_layers = num_layers
        self.time_encoder = Time2Vec(time_dim=time_dim)
        self.attn = nn.ModuleList(
            [
                TemporalAttention(
                    node_dim=self.node_feat_dim,
                    edge_dim=self.edge_feat_dim,
                    time_dim=time_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                )
            ]
        )
        self.merge_layers = nn.ModuleList(
            [
                MergeLayer(
                    input_dim1=self.attn[-1].query_dim,
                    input_dim2=self.node_feat_dim,
                    hidden_dim=embed_dim,
                    output_dim=embed_dim,
                )
            ]
        )

        if num_layers > 1:
            for _ in range(num_layers - 1):
                self.attn.append(
                    TemporalAttention(
                        node_dim=embed_dim,
                        edge_dim=self.edge_feat_dim,
                        time_dim=time_dim,
                        n_heads=n_heads,
                        dropout=dropout,
                    )
                )
                self.merge_layers.append(
                    MergeLayer(
                        input_dim1=self.attn[-1].query_dim,
                        input_dim2=self.node_feat_dim,
                        hidden_dim=embed_dim,
                        output_dim=embed_dim,
                    )
                )

    def forward(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        num_neighbors: int = 20,
        batch=None,
        is_negative=False,
        idx=-1,
        inference=False,
    ):
        if inference and is_negative:
            src_node_embeddings = None
        else:
            src_node_embeddings = self.compute_node_temporal_embeddings(
                node_ids=src_node_ids,
                node_interact_times=node_interact_times,
                current_layer_num=self.num_layers,
                num_neighbors=num_neighbors,
                batch=batch,
                is_negative=is_negative,
                is_src=True,
                idx=idx,
                inference=inference,
            )
        dst_node_embeddings = self.compute_node_temporal_embeddings(
            node_ids=dst_node_ids,
            node_interact_times=node_interact_times,
            current_layer_num=self.num_layers,
            num_neighbors=num_neighbors,
            batch=batch,
            is_negative=is_negative,
            is_src=False,
            idx=idx,
            inference=inference,
        )
        return src_node_embeddings, dst_node_embeddings

    def compute_node_temporal_embeddings(
        self,
        node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        current_layer_num: int,
        num_neighbors: int = 20,
        batch=None,
        is_negative=False,
        is_src=False,
        idx=-1,
        inference=False,
    ):
        device = self.node_raw_features.device

        # query (source) node always has the start time with time interval == 0
        # Tensor, shape (batch_size, 1, time_feat_dim)
        node_time_features = self.time_encoder(
            torch.zeros(node_interact_times.shape).unsqueeze(dim=1).to(device)
        )
        # Tensor, shape (batch_size, node_feat_dim)
        node_raw_features = self.node_raw_features[torch.from_numpy(node_ids)]

        if current_layer_num == 0:
            return node_raw_features
        else:
            # get source node representations by aggregating embeddings from the previous (current_layer_num - 1)-th layer
            # Tensor, shape (batch_size, output_dim or node_feat_dim)
            node_conv_features = self.compute_node_temporal_embeddings(
                node_ids=node_ids,
                node_interact_times=node_interact_times,
                current_layer_num=current_layer_num - 1,
                num_neighbors=self.num_nbrs[-current_layer_num],
                batch=batch,
                is_negative=is_negative,
                is_src=is_src,
                idx=idx,
                inference=inference,
            )

            # get temporal neighbors, including neighbor ids, edge ids and time information
            if len(node_ids) == batch.src.numel():
                nbr_nids = batch.nids[1]
                if inference and is_negative:
                    nbr_times = batch.nbr_times[0].flatten()
                else:
                    nbr_times = batch.times[1]
                nbr_feats = batch.nbr_feats[0]
            elif len(node_ids) == batch.src.numel() * num_neighbors:
                nbr_nids = batch.nbr_nids[1].flatten()
                nbr_times = batch.nbr_times[1].flatten()
                nbr_feats = batch.nbr_feats[1]
            elif len(node_ids) == (batch.neg.numel()):
                nbr_nids = batch.nids[1]

                if inference and is_negative:
                    nbr_times = batch.nbr_times[0].flatten()
                else:
                    nbr_times = batch.times[1].flatten()

                nbr_feats = batch.nbr_feats[0]
            elif len(node_ids) == (batch.neg.numel()) * num_neighbors:
                nbr_nids = batch.nbr_nids[1].flatten()
                nbr_times = batch.nbr_times[1].flatten()
                nbr_feats = batch.nbr_feats[1]
            else:
                assert False

            if not inference:
                src_nbr_nids, dst_nbr_nids, neg_nbr_nids = torch.chunk(
                    nbr_nids, chunks=3, dim=0
                )
                src_nbr_times, dst_nbr_times, neg_nbr_times = torch.chunk(
                    nbr_times, chunks=3, dim=0
                )
                src_nbr_feats, dst_nbr_feats, neg_nbr_feats = torch.chunk(
                    nbr_feats, chunks=3, dim=0
                )
            else:
                if is_negative:
                    if node_ids.shape[0] == 999:
                        bsize = num_neighbors
                    else:
                        bsize = num_neighbors * num_neighbors
                else:
                    bsize = node_ids.shape[0] * num_neighbors

                src_nbr_nids, dst_nbr_nids, neg_nbr_nids = (
                    nbr_nids[0:bsize],
                    nbr_nids[bsize : 2 * bsize],
                    nbr_nids[2 * bsize :],
                )
                src_nbr_times, dst_nbr_times, neg_nbr_times = (
                    nbr_times[0:bsize],
                    nbr_times[bsize : 2 * bsize],
                    nbr_times[2 * bsize :],
                )

                nbr_feats = nbr_feats.reshape(-1, nbr_feats.size(-1))
                src_nbr_feats, dst_nbr_feats, neg_nbr_feats = (
                    nbr_feats[0:bsize],
                    nbr_feats[bsize : 2 * bsize],
                    nbr_feats[2 * bsize :],
                )

            if is_src:
                neighbor_node_ids = src_nbr_nids.cpu().numpy()
                neighbor_times = src_nbr_times.cpu().numpy()
                neighbor_edge_features = src_nbr_feats
            elif is_negative:
                neighbor_node_ids = neg_nbr_nids.cpu().numpy()
                neighbor_times = neg_nbr_times.cpu().numpy()
                neighbor_edge_features = neg_nbr_feats
            else:
                neighbor_node_ids = dst_nbr_nids.cpu().numpy()
                neighbor_times = dst_nbr_times.cpu().numpy()
                neighbor_edge_features = dst_nbr_feats

            neighbor_node_ids = neighbor_node_ids.reshape(node_ids.shape[0], -1)
            neighbor_times = neighbor_times.reshape(node_ids.shape[0], -1)

            if inference:
                edge_feat_dim = neighbor_edge_features.shape[-1]
                neighbor_edge_features = neighbor_edge_features.reshape(
                    node_ids.shape[0], -1, edge_feat_dim
                )

            # get neighbor features from previous layers
            # shape (batch_size * num_neighbors, output_dim or node_feat_dim)
            neighbor_node_conv_features = self.compute_node_temporal_embeddings(
                node_ids=neighbor_node_ids.flatten(),
                node_interact_times=neighbor_times.flatten(),
                current_layer_num=current_layer_num - 1,
                num_neighbors=num_neighbors,
                batch=batch,
                is_negative=is_negative,
                is_src=is_src,
                idx=idx,
                inference=inference,
            )

            # shape (batch_size, num_neighbors, output_dim or node_feat_dim)
            neighbor_node_conv_features = neighbor_node_conv_features.reshape(
                node_ids.shape[0], num_neighbors, -1
            )
            # adarray, shape (batch_size, num_neighbors)
            neighbor_delta_times = node_interact_times[:, np.newaxis] - neighbor_times

            # shape (batch_size, num_neighbors, time_feat_dim)
            neighbor_time_features = self.time_encoder(
                torch.from_numpy(neighbor_delta_times).float().to(device)
            )
            with open('tgm_out.txt', mode='a') as f:
                lll = node_ids.reshape(-1)
                print(
                    f'BATCH {idx} IS_SRC_{is_src}_IS_NEG_{is_negative}_NODE_IDS',
                    file=f,
                )
                print(' '.join(f'{x:.8f}' for x in lll), file=f)

                lll = node_interact_times.reshape(-1)
                print(
                    f'BATCH {idx} IS_SRC_{is_src}_IS_NEG_{is_negative}_NODE_TIMES',
                    file=f,
                )
                print(' '.join(f'{x:.8f}' for x in lll), file=f)

                lll = neighbor_delta_times.reshape(-1)
                print(
                    f'BATCH {idx} IS_SRC_{is_src}_IS_NEG_{is_negative}_NBR_DELTA_TIMES',
                    file=f,
                )
                print(' '.join(f'{x:.8f}' for x in lll), file=f)

                lll = neighbor_node_ids.reshape(-1)
                print(
                    f'BATCH {idx} IS_SRC_{is_src}_IS_NEG_{is_negative}_NBR_IDS',
                    file=f,
                )
                print(' '.join(f'{x:.8f}' for x in lll), file=f)

            # temporal graph convolution
            # Tensor, output shape (batch_size, query_dim)
            output = self.attn[current_layer_num - 1](
                node_feat=node_conv_features,
                time_feat=node_time_features,
                nbr_node_feat=neighbor_node_conv_features,
                nbr_time_feat=neighbor_time_features,
                nbr_edge_feat=neighbor_edge_features,
                nbr_mask=neighbor_node_ids,
            )

            # Tensor, output shape (batch_size, output_dim)
            # follow the TGAT paper, use merge layer to combine the attention results and node original feature
            output = self.merge_layers[current_layer_num - 1](
                input_1=output, input_2=node_raw_features
            )
            return output


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
    losses, metrics = [], []

    for idx, batch in enumerate(tqdm(loader)):
        opt.zero_grad()

        batch_src_node_ids = batch.src.cpu().numpy()
        batch_dst_node_ids = batch.dst.cpu().numpy()
        batch_node_interact_times = batch.time.cpu().numpy()

        batch_neg_dst_node_ids = batch.neg.cpu().numpy()
        batch_neg_src_node_ids = batch_src_node_ids

        z_src, z_dst = encoder(
            src_node_ids=batch_src_node_ids,
            dst_node_ids=batch_dst_node_ids,
            node_interact_times=batch_node_interact_times,
            num_neighbors=NBRS,
            batch=batch,
            is_negative=False,
            idx=idx,
        )
        _, z_neg_dst = encoder(
            src_node_ids=batch_neg_src_node_ids,
            dst_node_ids=batch_neg_dst_node_ids,
            node_interact_times=batch_node_interact_times,
            num_neighbors=NBRS,
            batch=batch,
            is_negative=True,
            idx=idx,
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
        metrics.append(
            {
                'average_precision': average_precision_score(
                    y_true=labels.cpu().numpy(),
                    y_score=predicts.cpu().detach().numpy(),
                ),
                'roc_auc': roc_auc_score(
                    y_true=labels.cpu().numpy(),
                    y_score=predicts.cpu().detach().numpy(),
                ),
            }
        )

        if idx > 5:
            break

    print(f'Epoch: {epoch + 1}, train loss: {np.mean(losses):.4f}')
    for metric_name in metrics[0].keys():
        print(
            f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in metrics]):.4f}'
        )
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

            batch_node_interact_times = (
                torch.tensor([batch.time[idx]])
                .repeat(batch_dst_node_ids.shape[0])
                .cpu()
                .numpy()
            )
            assert batch_node_interact_times.shape[0] == len(batch_src_node_ids)

            z_src, z_dst = encoder(
                src_node_ids=batch_src_node_ids,
                dst_node_ids=batch_dst_node_ids,
                node_interact_times=batch_node_interact_times,
                num_neighbors=NBRS,
                batch=batch,
                is_negative=False,
                idx=idx,
                inference=True,
            )

            neg_batch_node_interact_times = (
                torch.tensor([batch.time[idx]])
                .repeat(batch_neg_dst_node_ids.shape[0])
                .cpu()
                .numpy()
            )

            _, z_neg_dst = encoder(
                src_node_ids=batch_neg_src_node_ids,
                dst_node_ids=batch_neg_dst_node_ids,
                node_interact_times=neg_batch_node_interact_times,
                num_neighbors=NBRS,
                batch=batch,
                is_negative=True,
                idx=idx,
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

NBRS = args.n_nbrs[0]

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
    dg: DGraph, sampling_type: str, neg_sampler: object, split_mode: str, nbr_hook=None
) -> List[DGHook]:
    if sampling_type == 'uniform':
        nbr_hook = NeighborSamplerHook(num_nbrs=args.n_nbrs)
    elif sampling_type == 'recency':
        if nbr_hook is None:
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


node_raw_features = STATIC_NODE_FEAT.cpu().numpy()

from tgb.linkproppred.dataset import LinkPropPredDataset

data = LinkPropPredDataset(
    name=args.dataset, root='datasets', preprocess=True
).full_data
edge_raw_features = data['edge_feat'].astype(np.float64)


encoder = TGAT(
    node_raw_features=node_raw_features,
    edge_dim=edge_raw_features.shape[1],
    time_dim=args.time_dim,
    embed_dim=args.embed_dim,
    num_layers=len(args.n_nbrs),
    n_heads=args.n_heads,
    dropout=float(args.dropout),
    device=args.device,
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

    SHARED_NBR_HOOK = RecencyNeighborHook(
        num_nbrs=args.n_nbrs,
        num_nodes=test_dg.num_nodes,
        edge_feats_dim=test_dg.edge_feats_dim,
    )
    foo_train_loader = DGDataLoader(
        train_dg,
        hook=_init_hooks(
            test_dg, args.sampling, neg_sampler, 'train', nbr_hook=SHARED_NBR_HOOK
        ),
        batch_size=2000,
        drop_last=False,
    )
    print('filling up neighbor hook in preperation for validation')
    for batch in tqdm(foo_train_loader):
        continue

    val_loader = DGDataLoader(
        val_dg,
        hook=_init_hooks(
            test_dg, args.sampling, neg_sampler, 'val', nbr_hook=SHARED_NBR_HOOK
        ),
        batch_size=1,
    )
    start_time = time.perf_counter()
    loss = train(train_loader, encoder, decoder, opt)
    end_time = time.perf_counter()
    latency = end_time - start_time

    val_results = eval(val_loader, encoder, decoder, eval_metric, evaluator)
    exit()

    print(
        f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f} '
        + ' '.join(f'{k}={v:.4f}' for k, v in val_results.items())
    )

test_results = eval(test_loader, encoder, decoder, eval_metric, evaluator)
print(' '.join(f'{k}={v:.4f}' for k, v in test_results.items()))
