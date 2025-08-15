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


class TGATTGM(nn.Module):
    def __init__(
        self,
        edge_dim: int,
        time_dim: int,
        embed_dim: int,
        num_layers: int,
        n_heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.time_encoder = Time2Vec(time_dim=time_dim)
        self.attn = nn.ModuleList(
            [
                TemporalAttention(
                    n_heads=n_heads,
                    node_dim=embed_dim,
                    edge_dim=edge_dim,
                    time_dim=time_dim,
                    out_dim=embed_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, batch: DGBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        device = batch.src.device
        z = torch.zeros(len(batch.unique_nids), self.embed_dim, device=device)

        for hop in reversed(range(self.num_layers)):
            seed_nodes = batch.nids[hop]
            nbrs = batch.nbr_nids[hop]
            nbr_mask = batch.nbr_mask[hop].bool()
            if seed_nodes.numel() == 0:
                continue

            # TODO: Check and read static node features
            node_feat = STATIC_NODE_FEAT[seed_nodes]
            node_time_feat = self.time_encoder(torch.zeros_like(seed_nodes))

            # If next next hops embeddings exist, use them instead of raw features
            nbr_feat = STATIC_NODE_FEAT[nbrs]
            if hop < self.num_layers - 1:
                valid_nbrs = nbrs[nbr_mask]
                nbr_feat[nbr_mask] = z[batch.global_to_local(valid_nbrs)]

            delta_time = batch.times[hop][:, None] - batch.nbr_times[hop]
            delta_time = delta_time.masked_fill(~nbr_mask, 0)

            nbr_time_feat = self.time_encoder(delta_time)

            out = self.attn[hop](
                node_feat=node_feat,
                time_feat=node_time_feat,
                edge_feat=batch.nbr_feats[hop],
                nbr_node_feat=nbr_feat,
                nbr_time_feat=nbr_time_feat,
                nbr_mask=nbr_mask,
            )
            z[batch.global_to_local(seed_nodes)] = out
        return z


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


class NeighborSampler:
    def __init__(
        self,
        adj_list: list,
        sample_neighbor_strategy: str = 'uniform',
        time_scaling_factor: float = 0.0,
        seed: int = None,
    ):
        self.sample_neighbor_strategy = sample_neighbor_strategy
        self.seed = seed

        # list of each node's neighbor ids, edge ids and interaction times, which are sorted by interaction times
        self.nodes_neighbor_ids = []
        self.nodes_edge_ids = []
        self.nodes_neighbor_times = []

        if self.sample_neighbor_strategy == 'time_interval_aware':
            self.nodes_neighbor_sampled_probabilities = []
            self.time_scaling_factor = time_scaling_factor

        # the list at the first position in adj_list is empty, hence, sorted() will return an empty list for the first position
        # its corresponding value in self.nodes_neighbor_ids, self.nodes_edge_ids, self.nodes_neighbor_times will also be empty with length 0
        for node_idx, per_node_neighbors in enumerate(adj_list):
            # per_node_neighbors is a list of tuples (neighbor_id, edge_id, timestamp)
            # sort the list based on timestamps, sorted() function is stable
            # Note that sort the list based on edge id is also correct, as the original data file ensures the interactions are chronological
            sorted_per_node_neighbors = sorted(per_node_neighbors, key=lambda x: x[2])
            self.nodes_neighbor_ids.append(
                np.array([x[0] for x in sorted_per_node_neighbors])
            )
            self.nodes_edge_ids.append(
                np.array([x[1] for x in sorted_per_node_neighbors])
            )
            self.nodes_neighbor_times.append(
                np.array([x[2] for x in sorted_per_node_neighbors])
            )

            # additional for time interval aware sampling strategy (proposed in CAWN paper)
            if self.sample_neighbor_strategy == 'time_interval_aware':
                self.nodes_neighbor_sampled_probabilities.append(
                    self.compute_sampled_probabilities(
                        np.array([x[2] for x in sorted_per_node_neighbors])
                    )
                )

        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def compute_sampled_probabilities(self, node_neighbor_times: np.ndarray):
        if len(node_neighbor_times) == 0:
            return np.array([])
        # compute the time delta with regard to the last time in node_neighbor_times
        node_neighbor_times = node_neighbor_times - np.max(node_neighbor_times)
        # compute the normalized sampled probabilities of historical neighbors
        exp_node_neighbor_times = np.exp(self.time_scaling_factor * node_neighbor_times)
        sampled_probabilities = exp_node_neighbor_times / np.cumsum(
            exp_node_neighbor_times
        )
        # note that the first few values in exp_node_neighbor_times may be all zero, which make the corresponding values in sampled_probabilities
        # become nan (divided by zero), so we replace the nan by a very large negative number -1e10 to denote the sampled probabilities
        sampled_probabilities[np.isnan(sampled_probabilities)] = -1e10
        return sampled_probabilities

    def find_neighbors_before(
        self,
        node_id: int,
        interact_time: float,
        return_sampled_probabilities: bool = False,
    ):
        # return index i, which satisfies list[i - 1] < v <= list[i]
        # return 0 for the first position in self.nodes_neighbor_times since the value at the first position is empty
        i = np.searchsorted(self.nodes_neighbor_times[node_id], interact_time)

        if return_sampled_probabilities:
            return (
                self.nodes_neighbor_ids[node_id][:i],
                self.nodes_edge_ids[node_id][:i],
                self.nodes_neighbor_times[node_id][:i],
                self.nodes_neighbor_sampled_probabilities[node_id][:i],
            )
        else:
            return (
                self.nodes_neighbor_ids[node_id][:i],
                self.nodes_edge_ids[node_id][:i],
                self.nodes_neighbor_times[node_id][:i],
                None,
            )

    def get_historical_neighbors(
        self,
        node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        num_neighbors: int = 20,
    ):
        assert num_neighbors > 0, (
            'Number of sampled neighbors for each node should be greater than 0!'
        )
        # All interactions described in the following three matrices are sorted in each row by time
        # each entry in position (i,j) represents the id of the j-th dst node of src node node_ids[i] with an interaction before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_ids = np.zeros((len(node_ids), num_neighbors)).astype(
            np.longlong
        )
        # each entry in position (i,j) represents the id of the edge with src node node_ids[i] and dst node nodes_neighbor_ids[i][j] with an interaction before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_edge_ids = np.zeros((len(node_ids), num_neighbors)).astype(np.longlong)
        # each entry in position (i,j) represents the interaction time between src node node_ids[i] and dst node nodes_neighbor_ids[i][j], before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_times = np.zeros((len(node_ids), num_neighbors)).astype(
            np.float32
        )

        # extracts all neighbors ids, edge ids and interaction times of nodes in node_ids, which happened before the corresponding time in node_interact_times
        for idx, (node_id, node_interact_time) in enumerate(
            zip(node_ids, node_interact_times)
        ):
            # find neighbors that interacted with node_id before time node_interact_time
            (
                node_neighbor_ids,
                node_edge_ids,
                node_neighbor_times,
                node_neighbor_sampled_probabilities,
            ) = self.find_neighbors_before(
                node_id=node_id,
                interact_time=node_interact_time,
                return_sampled_probabilities=self.sample_neighbor_strategy
                == 'time_interval_aware',
            )

            if len(node_neighbor_ids) > 0:
                if self.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
                    # when self.sample_neighbor_strategy == 'uniform', we shuffle the data before sampling with node_neighbor_sampled_probabilities as None
                    # when self.sample_neighbor_strategy == 'time_interval_aware', we sample neighbors based on node_neighbor_sampled_probabilities
                    # for time_interval_aware sampling strategy, we additionally use softmax to make the sum of sampled probabilities be 1
                    if node_neighbor_sampled_probabilities is not None:
                        # for extreme case that node_neighbor_sampled_probabilities only contains -1e10, which will make the denominator of softmax be zero,
                        # torch.softmax() function can tackle this case
                        node_neighbor_sampled_probabilities = torch.softmax(
                            torch.from_numpy(
                                node_neighbor_sampled_probabilities
                            ).float(),
                            dim=0,
                        ).numpy()
                    if self.seed is None:
                        sampled_indices = np.random.choice(
                            a=len(node_neighbor_ids),
                            size=num_neighbors,
                            p=node_neighbor_sampled_probabilities,
                        )
                    else:
                        sampled_indices = self.random_state.choice(
                            a=len(node_neighbor_ids),
                            size=num_neighbors,
                            p=node_neighbor_sampled_probabilities,
                        )

                    nodes_neighbor_ids[idx, :] = node_neighbor_ids[sampled_indices]
                    nodes_edge_ids[idx, :] = node_edge_ids[sampled_indices]
                    nodes_neighbor_times[idx, :] = node_neighbor_times[sampled_indices]

                    # resort based on timestamps, return the ids in sorted increasing order, note this maybe unstable when multiple edges happen at the same time
                    # (we still do this though this is unnecessary for TGAT or CAWN to guarantee the order of nodes,
                    # since TGAT computes in an order-agnostic manner with relative time encoding, and CAWN computes for each walk while the sampled nodes are in different walks)
                    sorted_position = nodes_neighbor_times[idx, :].argsort()
                    nodes_neighbor_ids[idx, :] = nodes_neighbor_ids[idx, :][
                        sorted_position
                    ]
                    nodes_edge_ids[idx, :] = nodes_edge_ids[idx, :][sorted_position]
                    nodes_neighbor_times[idx, :] = nodes_neighbor_times[idx, :][
                        sorted_position
                    ]
                elif self.sample_neighbor_strategy == 'recent':
                    # Take most recent interactions with number num_neighbors
                    node_neighbor_ids = node_neighbor_ids[-num_neighbors:]
                    node_edge_ids = node_edge_ids[-num_neighbors:]
                    node_neighbor_times = node_neighbor_times[-num_neighbors:]

                    # put the neighbors' information at the back positions
                    nodes_neighbor_ids[
                        idx, num_neighbors - len(node_neighbor_ids) :
                    ] = node_neighbor_ids
                    nodes_edge_ids[idx, num_neighbors - len(node_edge_ids) :] = (
                        node_edge_ids
                    )
                    nodes_neighbor_times[
                        idx, num_neighbors - len(node_neighbor_times) :
                    ] = node_neighbor_times
                else:
                    raise ValueError(
                        f'Not implemented error for sample_neighbor_strategy {self.sample_neighbor_strategy}!'
                    )

        # three ndarrays, with shape (batch_size, num_neighbors)
        return nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times

    def get_multi_hop_neighbors(
        self,
        num_hops: int,
        node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        num_neighbors: int = 20,
    ):
        assert num_hops > 0, 'Number of sampled hops should be greater than 0!'

        # get the temporal neighbors at the first hop
        # nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times -> ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = (
            self.get_historical_neighbors(
                node_ids=node_ids,
                node_interact_times=node_interact_times,
                num_neighbors=num_neighbors,
            )
        )
        # three lists to store the neighbor ids, edge ids and interaction timestamp information
        nodes_neighbor_ids_list = [nodes_neighbor_ids]
        nodes_edge_ids_list = [nodes_edge_ids]
        nodes_neighbor_times_list = [nodes_neighbor_times]
        for hop in range(1, num_hops):
            # get information of neighbors sampled at the current hop
            # three ndarrays, with shape (batch_size * num_neighbors ** hop, num_neighbors)
            nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = (
                self.get_historical_neighbors(
                    node_ids=nodes_neighbor_ids_list[-1].flatten(),
                    node_interact_times=nodes_neighbor_times_list[-1].flatten(),
                    num_neighbors=num_neighbors,
                )
            )
            # three ndarrays with shape (batch_size, num_neighbors ** (hop + 1))
            nodes_neighbor_ids = nodes_neighbor_ids.reshape(len(node_ids), -1)
            nodes_edge_ids = nodes_edge_ids.reshape(len(node_ids), -1)
            nodes_neighbor_times = nodes_neighbor_times.reshape(len(node_ids), -1)

            nodes_neighbor_ids_list.append(nodes_neighbor_ids)
            nodes_edge_ids_list.append(nodes_edge_ids)
            nodes_neighbor_times_list.append(nodes_neighbor_times)

        # tuple, each element in the tuple is a list of num_hops ndarrays, each with shape (batch_size, num_neighbors ** current_hop)
        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list

    def get_all_first_hop_neighbors(
        self, node_ids: np.ndarray, node_interact_times: np.ndarray
    ):
        # three lists to store the first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list = (
            [],
            [],
            [],
        )
        # get the temporal neighbors at the first hop
        for idx, (node_id, node_interact_time) in enumerate(
            zip(node_ids, node_interact_times)
        ):
            # find neighbors that interacted with node_id before time node_interact_time
            node_neighbor_ids, node_edge_ids, node_neighbor_times, _ = (
                self.find_neighbors_before(
                    node_id=node_id,
                    interact_time=node_interact_time,
                    return_sampled_probabilities=False,
                )
            )
            nodes_neighbor_ids_list.append(node_neighbor_ids)
            nodes_edge_ids_list.append(node_edge_ids)
            nodes_neighbor_times_list.append(node_neighbor_times)

        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)


def get_neighbor_sampler(
    data: 'Data',
    sample_neighbor_strategy: str = 'uniform',
    time_scaling_factor: float = 0.0,
    seed: int = None,
):
    max_node_id = max(data.src_node_ids.max(), data.dst_node_ids.max())
    # the adjacency vector stores edges for each node (source or destination), undirected
    # adj_list, list of list, where each element is a list of triple tuple (node_id, edge_id, timestamp)
    # the list at the first position in adj_list is empty
    adj_list = [[] for _ in range(max_node_id + 1)]
    for src_node_id, dst_node_id, edge_id, node_interact_time in zip(
        data.src_node_ids, data.dst_node_ids, data.edge_ids, data.node_interact_times
    ):
        adj_list[src_node_id].append((dst_node_id, edge_id, node_interact_time))
        adj_list[dst_node_id].append((src_node_id, edge_id, node_interact_time))

    return NeighborSampler(
        adj_list=adj_list,
        sample_neighbor_strategy=sample_neighbor_strategy,
        time_scaling_factor=time_scaling_factor,
        seed=seed,
    )


class TGAT(nn.Module):
    def __init__(
        self,
        node_raw_features: np.ndarray,
        edge_raw_features: np.ndarray,
        neighbor_sampler: NeighborSampler,
        time_dim: int,
        output_dim: int = 172,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        device: str = 'cpu',
    ):
        super().__init__()

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
        print(self.num_layers, ' number of layers')

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
                        num_heads=self.num_heads,
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
    ):
        # Tensor, shape (batch_size, output_dim)
        src_node_embeddings = self.compute_node_temporal_embeddings(
            node_ids=src_node_ids,
            node_interact_times=node_interact_times,
            current_layer_num=self.num_layers,
            num_neighbors=num_neighbors,
            batch=batch,
            is_negative=is_negative,
            is_src=True,
        )
        # Tensor, shape (batch_size, output_dim)
        dst_node_embeddings = self.compute_node_temporal_embeddings(
            node_ids=dst_node_ids,
            node_interact_times=node_interact_times,
            current_layer_num=self.num_layers,
            num_neighbors=num_neighbors,
            batch=batch,
            is_negative=is_negative,
            is_src=False,
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
            # print(node_ids.shape, node_interact_times.shape)
            # print('calling recursive forward for nodes')
            # input()
            node_conv_features = self.compute_node_temporal_embeddings(
                node_ids=node_ids,
                node_interact_times=node_interact_times,
                current_layer_num=current_layer_num - 1,
                num_neighbors=num_neighbors,
            )
            # print('node conv feats: ', node_conv_features.shape)

            # get temporal neighbors, including neighbor ids, edge ids and time information
            # neighbor_node_ids, ndarray, shape (batch_size, num_neighbors)
            # neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors)
            # neighbor_times, ndarray, shape (batch_size, num_neighbors)

            # TODO: Use batch
            # neighbor_node_ids, neighbor_edge_ids, neighbor_times = (
            #    self.neighbor_sampler.get_historical_neighbors(
            #        node_ids=node_ids,
            #        node_interact_times=node_interact_times,
            #        num_neighbors=num_neighbors,
            #    )
            # )
            nbr_nids = batch.nbr_nids[0]
            nbr_times = batch.nbr_times[0]
            nbr_feats = batch.nbr_feats[0]

            src_nbr_nids, dst_nbr_nids, neg_nbr_nids = torch.chunk(
                nbr_nids, chunks=3, dim=0
            )
            src_nbr_times, dst_nbr_times, neg_nbr_times = torch.chunk(
                nbr_times, chunks=3, dim=0
            )
            src_nbr_feats, dst_nbr_feats, neg_nbr_feats = torch.chunk(
                nbr_feats, chunks=3, dim=0
            )

            if is_src:
                neighbor_node_ids = src_nbr_nids.cpu().numpy()
                neighbor_times = src_nbr_times.cpu().numpy()
            elif is_negative:
                neighbor_node_ids = neg_nbr_nids.cpu().numpy()
                neighbor_times = neg_nbr_times.cpu().numpy()
            else:
                neighbor_node_ids = dst_nbr_nids.cpu().numpy()
                neighbor_times = dst_nbr_times.cpu().numpy()

            # print('Nbr_nids: ', neighbor_node_ids.shape)
            # print('Nbr_times: ', neighbor_times.shape)
            # input()

            neighbor_edge_ids = None

            # get neighbor features from previous layers
            # shape (batch_size * num_neighbors, output_dim or node_feat_dim)
            # print('calling recursive forward for nbrs')
            # print(neighbor_node_ids.shape)
            # input()
            neighbor_node_conv_features = self.compute_node_temporal_embeddings(
                node_ids=neighbor_node_ids.flatten(),
                node_interact_times=neighbor_times.flatten(),
                current_layer_num=current_layer_num - 1,
                num_neighbors=num_neighbors,
            )

            # print('nbr feats: ', neighbor_node_conv_features.shape)
            # input()
            # shape (batch_size, num_neighbors, output_dim or node_feat_dim)
            neighbor_node_conv_features = neighbor_node_conv_features.reshape(
                node_ids.shape[0], num_neighbors, neighbor_node_conv_features.shape[-1]
            )

            # print(neighbor_node_conv_features.shape)
            # input()

            # compute time interval between current time and historical interaction time
            # adarray, shape (batch_size, num_neighbors)
            neighbor_delta_times = node_interact_times[:, np.newaxis] - neighbor_times

            # shape (batch_size, num_neighbors, time_feat_dim)
            neighbor_time_features = self.time_encoder(
                timestamps=torch.from_numpy(neighbor_delta_times).float().to(device)
            )

            # get edge features, shape (batch_size, num_neighbors, edge_feat_dim)
            # neighbor_edge_features = self.edge_raw_features[
            #    torch.from_numpy(neighbor_edge_ids)
            # ]

            if is_src:
                neighbor_edge_features = src_nbr_feats
            elif is_negative:
                neighbor_edge_features = neg_nbr_feats
            else:
                neighbor_edge_features = dst_nbr_feats

            # print('neighb edge features: ', neighbor_edge_features.shape)

            # temporal graph convolution
            # Tensor, output shape (batch_size, query_dim)
            output, _ = self.temporal_conv_layers[current_layer_num - 1](
                node_features=node_conv_features,
                node_time_features=node_time_features,
                neighbor_node_features=neighbor_node_conv_features,
                neighbor_node_time_features=neighbor_time_features,
                neighbor_node_edge_features=neighbor_edge_features,
                neighbor_masks=neighbor_node_ids,
            )
            # print('attention out shape: ', output.shape)
            # input()

            # Tensor, output shape (batch_size, output_dim)
            # follow the TGAT paper, use merge layer to combine the attention results and node original feature
            output = self.merge_layers[current_layer_num - 1](
                input_1=output, input_2=node_raw_features
            )
            # print('final out shape: ', output.shape)
            # input()

            return output


class LinkPredictor(nn.Module):
    def __init__(self, dim: int) -> None:
        # E    super().__init__()
        # E    self.lin_src = nn.Linear(dim, dim)
        # E    self.lin_dst = nn.Linear(dim, dim)
        # E    self.lin_out = nn.Linear(dim, 1)

        # Edef forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        # E    h = self.lin_src(z_src) + self.lin_dst(z_dst)
        # E    h = h.relu()
        # E    return self.lin_out(h).sigmoid().view(-1)
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

        #        _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(
        #            size=len(batch_src_node_ids)
        #        )
        #        batch_neg_src_node_ids = batch_src_node_ids
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
            num_neighbors=20,
            batch=batch,
            is_negative=False,
        )

        # get temporal embedding of negative source and negative destination nodes
        # two Tensors, with shape (batch_size, output_dim)
        z_neg_src, z_neg_dst = encoder(
            src_node_ids=batch_neg_src_node_ids,
            dst_node_ids=batch_neg_dst_node_ids,
            node_interact_times=batch_node_interact_times,
            num_neighbors=20,
            batch=batch,
            is_negative=True,
        )
        # z = encoder(batch)

        # z_src = z[batch.global_to_local(batch.src)]
        # z_dst = z[batch.global_to_local(batch.dst)]
        # z_neg = z[batch.global_to_local(batch.neg)]

        pos_out = decoder(z_src, z_dst)
        neg_out = decoder(z_neg_src, z_neg_dst)

        pos_prob = pos_out.sigmoid()
        neg_prob = neg_out.sigmoid()

        loss_func = nn.BCELoss()
        predicts = torch.cat([pos_prob, neg_prob], dim=0)
        labels = torch.cat(
            [torch.ones_like(pos_prob), torch.zeros_like(neg_prob)], dim=0
        )

        # loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        # loss += F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))

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

    print(f'Epoch: {epoch + 1}, train loss: {np.mean(losses):.4f}')
    for metric_name in metrics[0].keys():
        print(
            f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in metrics]):.4f}'
        )
    exit()
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
    for batch in tqdm(loader):
        z = encoder(batch)

        for idx, neg_batch in enumerate(batch.neg_batch_list):
            dst_ids = torch.cat([batch.dst[idx].unsqueeze(0), neg_batch])
            src_ids = batch.src[idx].repeat(len(dst_ids))

            z_src = z[batch.global_to_local(src_ids)]
            z_dst = z[batch.global_to_local(dst_ids)]
            y_pred = decoder(z_src, z_dst)

            input_dict = {
                'y_pred_pos': y_pred[0].detach().cpu().numpy(),
                'y_pred_neg': y_pred[1:].detach().cpu().numpy(),
                'eval_metric': [eval_metric],
            }
            perf_list.append(evaluator.eval(input_dict)[eval_metric])

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
NUM_NODES, NODE_FEAT_DIM = test_dg.num_nodes, args.embed_dim
STATIC_NODE_FEAT = torch.randn((NUM_NODES, NODE_FEAT_DIM), device=args.device)


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


node_raw_features = STATIC_NODE_FEAT.cpu().numpy()

from tgb.linkproppred.dataset import LinkPropPredDataset

data = LinkPropPredDataset(
    name=args.dataset, root='datasets', preprocess=True
).full_data
edge_raw_features = data['edge_feat'].astype(np.float64)
print(node_raw_features.shape, edge_raw_features.shape)

# train_neighbor_sampler = get_neighbor_sampler(
#    data=train_data,
#    sample_neighbor_strategy=args.sample_neighbor_strategy,
#    time_scaling_factor=args.time_scaling_factor,
#    seed=0,
# )

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
).to(args.device)
decoder = LinkPredictor(dim=args.embed_dim).to(args.device)
opt = torch.optim.Adam(
    set(encoder.parameters()) | set(decoder.parameters()), lr=float(args.lr)
)

print('encoder: ', encoder)
for epoch in range(1, args.epochs + 1):
    # TODO: Need a clean way to clear nbr state across epochs
    train_loader = DGDataLoader(
        train_dg,
        hook=_init_hooks(test_dg, args.sampling, neg_sampler, 'train'),
        batch_size=args.bsize,
    )
    val_loader = DGDataLoader(
        val_dg,
        hook=_init_hooks(test_dg, args.sampling, neg_sampler, 'val'),
        batch_size=args.bsize,
    )
    start_time = time.perf_counter()
    loss = train(train_loader, encoder, decoder, opt)
    end_time = time.perf_counter()
    latency = end_time - start_time

    val_results = eval(val_loader, encoder, decoder, eval_metric, evaluator)

    print(
        f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f} '
        + ' '.join(f'{k}={v:.4f}' for k, v in val_results.items())
    )

test_results = eval(test_loader, encoder, decoder, eval_metric, evaluator)
print(' '.join(f'{k}={v:.4f}' for k, v in test_results.items()))
