import argparse
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
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
parser.add_argument('--seed', type=int, default=0, help='random seed to use')
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


##########################################################
class TimeEncoder(nn.Module):
    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        super().__init__()

        self.time_dim = time_dim
        # trainable parameters for time encoding
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter(
            (
                torch.from_numpy(
                    1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32)
                )
            ).reshape(time_dim, -1)
        )
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor):
        # Tensor, shape (batch_size, seq_len, 1)
        timestamps = timestamps.unsqueeze(dim=2)

        # Tensor, shape (batch_size, seq_len, time_dim)
        output = torch.cos(self.w(timestamps))

        return output


class MergeLayer(nn.Module):
    def __init__(
        self, input_dim1: int, input_dim2: int, hidden_dim: int, output_dim: int
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim1 + input_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor):
        # Tensor, shape (*, input_dim1 + input_dim2)
        x = torch.cat([input_1, input_2], dim=1)
        # Tensor, shape (*, output_dim)
        h = self.fc2(self.act(self.fc1(x)))
        return h


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        time_feat_dim: int,
        num_heads: int = 2,
        dropout: float = 0.1,
    ):
        super(MultiHeadAttention, self).__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_heads = num_heads

        self.query_dim = node_feat_dim + time_feat_dim
        self.key_dim = node_feat_dim + edge_feat_dim + time_feat_dim

        if self.query_dim % num_heads != 0:
            print(
                'warning: the query_dim cannot be divided by num_heads, perform padding to support the computation'
            )
            self.pad_dim = num_heads - self.query_dim % num_heads
            self.query_dim += self.pad_dim
        else:
            self.pad_dim = 0

        assert self.query_dim % num_heads == 0, (
            'The sum of node_feat_dim and time_feat_dim should be divided by num_heads!'
        )

        self.head_dim = self.query_dim // num_heads

        self.query_projection = nn.Linear(
            self.query_dim, num_heads * self.head_dim, bias=False
        )
        self.key_projection = nn.Linear(
            self.key_dim, num_heads * self.head_dim, bias=False
        )
        self.value_projection = nn.Linear(
            self.key_dim, num_heads * self.head_dim, bias=False
        )

        self.scaling_factor = self.head_dim**-0.5

        self.layer_norm = nn.LayerNorm(self.query_dim)

        self.residual_fc = nn.Linear(num_heads * self.head_dim, self.query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        node_time_features: torch.Tensor,
        neighbor_node_features: torch.Tensor,
        neighbor_node_time_features: torch.Tensor,
        neighbor_node_edge_features: torch.Tensor,
        neighbor_masks: np.ndarray,
    ):
        # Tensor, shape (batch_size, 1, node_feat_dim)
        node_features = torch.unsqueeze(node_features, dim=1)

        # we need to pad for the inputs
        if self.pad_dim != 0:
            node_features = torch.cat(
                [
                    node_features,
                    torch.zeros(
                        node_features.shape[0], node_features.shape[1], self.pad_dim
                    ).to(node_features.device),
                ],
                dim=2,
            )

        # Tensor, shape (batch_size, 1, query_dim)
        query = residual = torch.cat([node_features, node_time_features], dim=2)
        # shape (batch_size, 1, num_heads, self.head_dim)
        query = self.query_projection(query).reshape(
            query.shape[0], query.shape[1], self.num_heads, self.head_dim
        )

        # if (len(neighbor_node_edge_features.shape) < 3):
        #     neighbor_node_edge_features = neighbor_node_edge_features.unsqueeze(dim=0)

        # Tensor, shape (batch_size, num_neighbors, node_feat_dim + edge_feat_dim + time_feat_dim)
        key = value = torch.cat(
            [
                neighbor_node_features,
                neighbor_node_edge_features,
                neighbor_node_time_features,
            ],
            dim=2,
        )
        # Tensor, shape (batch_size, num_neighbors, num_heads, self.head_dim)
        self.key_projection(key)
        key = self.key_projection(key).reshape(
            key.shape[0], key.shape[1], self.num_heads, self.head_dim
        )
        # Tensor, shape (batch_size, num_neighbors, num_heads, self.head_dim)
        value = self.value_projection(value).reshape(
            value.shape[0], value.shape[1], self.num_heads, self.head_dim
        )

        # Tensor, shape (batch_size, num_heads, 1, self.head_dim)
        query = query.permute(0, 2, 1, 3)
        # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
        key = key.permute(0, 2, 1, 3)
        # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
        value = value.permute(0, 2, 1, 3)

        # Tensor, shape (batch_size, num_heads, 1, num_neighbors)
        attention = torch.einsum('bhld,bhnd->bhln', query, key)
        attention = attention * self.scaling_factor

        # Tensor, shape (batch_size, 1, num_neighbors)
        attention_mask = (
            torch.from_numpy(neighbor_masks).to(node_features.device).unsqueeze(dim=1)
        )
        attention_mask = attention_mask == 0
        # Tensor, shape (batch_size, self.num_heads, 1, num_neighbors)
        attention_mask = torch.stack(
            [attention_mask for _ in range(self.num_heads)], dim=1
        )

        # Tensor, shape (batch_size, self.num_heads, 1, num_neighbors)
        # note that if a node has no valid neighbor (whose neighbor_masks are all zero), directly set the masks to -np.inf will make the
        # attention scores after softmax be nan. Therefore, we choose a very large negative number (-1e10 following TGAT) instead of -np.inf to tackle this case
        attention = attention.masked_fill(attention_mask, -1e10)

        # Tensor, shape (batch_size, num_heads, 1, num_neighbors)
        attention_scores = self.dropout(torch.softmax(attention, dim=-1))

        # Tensor, shape (batch_size, num_heads, 1, self.head_dim)
        attention_output = torch.einsum('bhln,bhnd->bhld', attention_scores, value)

        # Tensor, shape (batch_size, 1, num_heads * self.head_dim), where num_heads * self.head_dim is equal to query_dim
        attention_output = attention_output.permute(0, 2, 1, 3).flatten(start_dim=2)

        # Tensor, shape (batch_size, 1, query_dim)
        output = self.dropout(self.residual_fc(attention_output))

        # Tensor, shape (batch_size, 1, query_dim)
        output = self.layer_norm(output + residual)

        # Tensor, shape (batch_size, query_dim)
        output = output.squeeze(dim=1)
        # Tensor, shape (batch_size, num_heads, num_neighbors)
        attention_scores = attention_scores.squeeze(dim=2)

        return output, attention_scores


######################################################


class TGAT(nn.Module):
    def __init__(
        self,
        node_raw_features: np.ndarray,
        edge_raw_features: np.ndarray,
        neighbor_sampler,
        time_dim: int,
        output_dim: int = 172,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        device: str = 'cpu',
        num_nbrs=None,
    ):
        super().__init__()

        self.num_nbrs = num_nbrs

        self.node_raw_features = torch.from_numpy(
            node_raw_features.astype(np.float32)
        ).to(device)
        self.edge_raw_features = torch.from_numpy(
            edge_raw_features.astype(np.float32)
        ).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.num_layers = num_layers

        self.time_encoder = TimeEncoder(time_dim=time_dim)
        self.temporal_conv_layers = nn.ModuleList(
            [
                MultiHeadAttention(
                    node_feat_dim=self.node_feat_dim,
                    edge_feat_dim=self.edge_feat_dim,
                    time_feat_dim=time_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            ]
        )
        # follow the TGAT paper, use merge layer to combine the attention results and node original feature
        self.merge_layers = nn.ModuleList(
            [
                MergeLayer(
                    input_dim1=self.temporal_conv_layers[-1].query_dim,
                    input_dim2=self.node_feat_dim,
                    hidden_dim=output_dim,
                    output_dim=output_dim,
                )
            ]
        )

        if num_layers > 1:
            for _ in range(num_layers - 1):
                self.temporal_conv_layers.append(
                    MultiHeadAttention(
                        node_feat_dim=output_dim,
                        edge_feat_dim=self.edge_feat_dim,
                        time_feat_dim=time_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                    )
                )
                self.merge_layers.append(
                    MergeLayer(
                        input_dim1=self.temporal_conv_layers[-1].query_dim,
                        input_dim2=self.node_feat_dim,
                        hidden_dim=output_dim,
                        output_dim=output_dim,
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
        # Tensor, shape (batch_size, output_dim)
        if inference and is_negative:
            # TODO: Going to copy from z_src (positive)
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
        assert current_layer_num >= 0
        device = self.node_raw_features.device

        # query (source) node always has the start time with time interval == 0
        # Tensor, shape (batch_size, 1, time_feat_dim)
        node_time_features = self.time_encoder(
            timestamps=torch.zeros(node_interact_times.shape)
            .unsqueeze(dim=1)
            .to(device)
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
            # neighbor_node_ids, ndarray, shape (batch_size, num_neighbors)
            # neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors)
            # neighbor_times, ndarray, shape (batch_size, num_neighbors)

            if len(node_ids) == batch.src.numel():
                nbr_nids = batch.nids[1]
                if inference and is_negative:
                    print('GRABBING TIMES')
                    nbr_times = batch.nbr_times[0].flatten()
                    print(nbr_times)
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
                    print('GRABBING TIMES batch.neg')
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

            #! equal chunks during training
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
                # print(
                #    f'[{current_layer_num}] Trying to chunk with node ids: ',
                #    node_ids.shape,
                # )
                # print('nbr nids: ', nbr_nids.shape)
                # print('nbr feats: ', nbr_feats.shape)
                # input()
                if is_negative:
                    # TODO: Sketchy hardcode
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
            if inference and is_negative:
                print(
                    f'\n[{current_layer_num}]',
                    ' ids: ',
                    node_ids.shape,
                    f'  {node_ids[-10:]}  is_src: {is_src} ',
                    f'is_neg: {is_negative}',
                    f'nbrs: {neighbor_node_ids[-10:]}',
                    f'nbr times: {neighbor_times}',
                )
                print(batch.nids[0][-3:])
                print(batch.nids[1][-3 * 5 :])
                input()

            # TODO: Ensure this doesnt break train
            if inference:
                edge_feat_dim = neighbor_edge_features.shape[-1]
                neighbor_edge_features = neighbor_edge_features.reshape(
                    node_ids.shape[0], -1, edge_feat_dim
                )

            # print(f'[{current_layer_num}] Nbr_nids: ', neighbor_node_ids.shape)
            # print(f'[{current_layer_num}] Nbr_edge: ', neighbor_edge_features.shape)
            # input()

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

            # print(
            #    f'[{current_layer_num}] got recurse feats: ',
            #    neighbor_node_conv_features.shape,
            # )
            # input()
            # shape (batch_size, num_neighbors, output_dim or node_feat_dim)

            # print (neighbor_node_conv_features.shape)
            neighbor_node_conv_features = neighbor_node_conv_features.reshape(
                node_ids.shape[0], num_neighbors, -1
            )

            # compute time interval between current time and historical interaction time
            # adarray, shape (batch_size, num_neighbors)
            # print('node interact times:', node_interact_times.astype(int))
            # print('neighbor times: ', neighbor_times.astype(int))
            # print('neighbor node_ids: ', neighbor_node_ids.astype(int))
            neighbor_delta_times = node_interact_times[:, np.newaxis] - neighbor_times

            # shape (batch_size, num_neighbors, time_feat_dim)
            neighbor_time_features = self.time_encoder(
                timestamps=torch.from_numpy(neighbor_delta_times).float().to(device)
            )

            # get edge features, shape (batch_size, num_neighbors, edge_feat_dim)
            # neighbor_edge_features = self.edge_raw_features[
            #    torch.from_numpy(neighbor_edge_ids)
            # ]

            # temporal graph convolution
            # Tensor, output shape (batch_size, query_dim)

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

                # print(
                #    f'[{current_layer_num}] attention node_features ({node_conv_features.shape}), time features ({node_time_features.shape}), nbr node features ({neighbor_node_conv_features.shape}), nbr time features ({neighbor_time_features.shape}), nbr edge features ({neighbor_edge_features.shape}), nbr mask ({neighbor_node_ids.shape})'
                # )
                # input()

            output, _ = self.temporal_conv_layers[current_layer_num - 1](
                node_features=node_conv_features,
                node_time_features=node_time_features,
                neighbor_node_features=neighbor_node_conv_features,
                neighbor_node_time_features=neighbor_time_features,
                neighbor_node_edge_features=neighbor_edge_features,
                neighbor_masks=neighbor_node_ids,
            )

            # Tensor, output shape (batch_size, output_dim)
            # follow the TGAT paper, use merge layer to combine the attention results and node original feature
            output = self.merge_layers[current_layer_num - 1](
                input_1=output, input_2=node_raw_features
            )
            # print(f'[{current_layer_num}] final out shape: ', output.shape)
            return output


class LinkPredictor(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        input_dim1 = input_dim2 = hidden_dim = dim
        output_dim = 1

        self.fc1 = nn.Linear(input_dim1 + input_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor):
        # Tensor, shape (*, input_dim1 + input_dim2)
        x = torch.cat([input_1, input_2], dim=1)
        # Tensor, shape (*, output_dim)
        h = self.fc2(self.act(self.fc1(x)))
        return h


def train(
    loader: DGDataLoader,
    encoder: nn.Module,
    decoder: nn.Module,
    opt: torch.optim.Optimizer,
) -> float:
    encoder.train()
    decoder.train()
    total_loss = 0
    losses = []
    metrics = []

    tt = tqdm(loader, ncols=120)
    for idx, batch in enumerate(tt):
        opt.zero_grad()

        batch_src_node_ids = batch.src.cpu().numpy()
        batch_dst_node_ids = batch.dst.cpu().numpy()
        batch_node_interact_times = batch.time.cpu().numpy()

        batch_neg_dst_node_ids = batch.neg.cpu().numpy()
        batch_neg_src_node_ids = batch_src_node_ids

        # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
        # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
        # get temporal embedding of source and destination nodes
        # two Tensors, with shape (batch_size, output_dim)
        z_src, z_dst = encoder(
            src_node_ids=batch_src_node_ids,
            dst_node_ids=batch_dst_node_ids,
            node_interact_times=batch_node_interact_times,
            num_neighbors=NBRS,
            batch=batch,
            is_negative=False,
            idx=idx,
        )

        # get temporal embedding of negative source and negative destination nodes
        # two Tensors, with shape (batch_size, output_dim)
        z_neg_src, z_neg_dst = encoder(
            src_node_ids=batch_neg_src_node_ids,
            dst_node_ids=batch_neg_dst_node_ids,
            node_interact_times=batch_node_interact_times,
            num_neighbors=NBRS,
            batch=batch,
            is_negative=True,
            idx=idx,
        )

        pos_prob = decoder(z_src, z_dst).squeeze(dim=-1).sigmoid()
        neg_prob = decoder(z_neg_src, z_neg_dst).squeeze(dim=-1).sigmoid()

        loss_func = nn.BCELoss()
        predicts = torch.cat([pos_prob, neg_prob], dim=0)
        labels = torch.cat(
            [torch.ones_like(pos_prob), torch.zeros_like(neg_prob)], dim=0
        )

        loss = loss_func(input=predicts, target=labels)
        loss.backward()
        opt.step()
        total_loss += float(loss)

        tt.set_description(
            f'Epoch: {1}, train for the {idx + 1}-th batch, train loss: {loss.item()}'
        )
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

        if idx > 0:
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
        #! only evaluate for first edge in a batch for debugging purpose
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
            print(input_dict['y_pred_pos'], input_dict['y_pred_neg'][:3])
            print(f'---\nbatch ID: {batch_id}, MRR, {perf}---\n')
            input()

        batch_id += 1
        if batch_id > 5:
            break

    metric_dict = {}
    metric_dict[eval_metric] = float(np.mean(perf_list))
    print(f'validate {eval_metric}, {np.mean([metric_dict[eval_metric]])}')
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
NUM_NODES, NODE_FEAT_DIM = test_dg.num_nodes, 1  # CHANGED TO SINGLE FEATURE
STATIC_NODE_FEAT = torch.zeros((NUM_NODES, NODE_FEAT_DIM), device=args.device)


# TODO: trying to share this between hook managers
SHARED_NBR_HOOK = RecencyNeighborHook(
    num_nbrs=args.n_nbrs,
    num_nodes=test_dg.num_nodes,
    edge_feats_dim=test_dg.edge_feats_dim,
)


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
    edge_raw_features=edge_raw_features,
    neighbor_sampler=None,
    time_dim=args.time_dim,
    output_dim=args.embed_dim,
    num_layers=len(args.n_nbrs),
    num_heads=args.n_heads,
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

    # Reset the nbr hook, then loop through the entire training data to fill up it's state
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
        batch_size=200,
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
        # batch_size=args.bsize,
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
