from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from models.modules import MergeLayer, MultiHeadAttention, TimeEncoder


class TGAT(nn.Module):
    def __init__(
        self,
        node_raw_features: np.ndarray,
        edge_raw_features: np.ndarray,
        time_feat_dim: int,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        device: str = 'cpu',
    ) -> None:
        """Inductive respresentation learning on temporal graphs. Reference: https://arxiv.org/abs/2002.07962.

        Args:
            node_raw_features (ndarray): (num_nodes + 1, node_feat_dim)
            edge_raw_features (ndarray): (num_edges + 1, node_feat_dim)
            time_feat_dim(int): Dimension of time encoding features.
            num_layers (int): Number of temporal graph convolution layers.
            num_heads (int): Number of attention heads.
            dropout (float): dropout rate.
            device: (str, device): Device
        """
        super(TGAT, self).__init__()

        self.node_raw_features = torch.from_numpy(
            node_raw_features.astype(np.float32)
        ).to(device)
        self.edge_raw_features = torch.from_numpy(
            edge_raw_features.astype(np.float32)
        ).to(device)

        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)
        self.temporal_conv_layers = nn.ModuleList(
            [
                MultiHeadAttention(
                    node_feat_dim=self.node_feat_dim,
                    edge_feat_dim=self.edge_feat_dim,
                    time_feat_dim=self.time_feat_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
                for _ in range(num_layers)
            ]
        )
        # follow the TGAT paper, use merge layer to combine the attention results and node original feature
        self.merge_layers = nn.ModuleList(
            [
                MergeLayer(
                    input_dim1=self.node_feat_dim + self.time_feat_dim,
                    input_dim2=self.node_feat_dim,
                    hidden_dim=self.node_feat_dim,
                    output_dim=self.node_feat_dim,
                )
                for _ in range(num_layers)
            ]
        )

    def compute_src_dst_node_temporal_embeddings(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        num_neighbors: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute source and destination node temporal embeddings.

        Args:
             src_node_ids (ndarray):  (batch_size, )
             dst_node_ids (ndarray):  (batch_size, )
             node_interact_times (ndarray):  (batch_size, )
             num_neighbors (int): number of neighbor to sample for each node

        Returns:
            Source and destination node embeddings tensor.
        """
        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings = self.compute_node_temporal_embeddings(
            node_ids=src_node_ids,
            node_interact_times=node_interact_times,
            current_layer_num=self.num_layers,
            num_neighbors=num_neighbors,
        )
        # Tensor, shape (batch_size, node_feat_dim)
        dst_node_embeddings = self.compute_node_temporal_embeddings(
            node_ids=dst_node_ids,
            node_interact_times=node_interact_times,
            current_layer_num=self.num_layers,
            num_neighbors=num_neighbors,
        )
        return src_node_embeddings, dst_node_embeddings

    def compute_node_temporal_embeddings(
        self,
        node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        current_layer_num: int,
        num_neighbors: int = 20,
    ) -> torch.Tensor:
        """Given node ids node_ids, and the corresponding time node_interact_times.

        Args:
            node_ids (ndarray): (batch_size, ) or (*, ), node ids
            node_interact_times (ndarray): shape (batch_size, ) or (*, ), node interaction times
            current_layer_num (int): current layer number
            num_neighbors (int): number of neighbors to sample for each node

        Returns:
            Temporal embeddings after convolution at the current_layer_num
        """
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
            # Tensor, shape (batch_size, node_feat_dim)
            node_conv_features = self.compute_node_temporal_embeddings(
                node_ids=node_ids,
                node_interact_times=node_interact_times,
                current_layer_num=current_layer_num - 1,
                num_neighbors=num_neighbors,
            )

            # get temporal neighbors, including neighbor ids, edge ids and time information
            # neighbor_node_ids, ndarray, shape (batch_size, num_neighbors)
            # neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors)
            # neighbor_times, ndarray, shape (batch_size, num_neighbors)
            neighbor_node_ids, neighbor_edge_ids, neighbor_times = (
                self.neighbor_sampler.get_historical_neighbors(
                    node_ids=node_ids,
                    node_interact_times=node_interact_times,
                    num_neighbors=num_neighbors,
                )
            )

            # get neighbor features from previous layers
            # shape (batch_size * num_neighbors, node_feat_dim)
            neighbor_node_conv_features = self.compute_node_temporal_embeddings(
                node_ids=neighbor_node_ids.flatten(),
                node_interact_times=neighbor_times.flatten(),
                current_layer_num=current_layer_num - 1,
                num_neighbors=num_neighbors,
            )
            # shape (batch_size, num_neighbors, node_feat_dim)
            neighbor_node_conv_features = neighbor_node_conv_features.reshape(
                node_ids.shape[0], num_neighbors, self.node_feat_dim
            )

            # compute time interval between current time and historical interaction time
            # adarray, shape (batch_size, num_neighbors)
            neighbor_delta_times = node_interact_times[:, np.newaxis] - neighbor_times

            # shape (batch_size, num_neighbors, time_feat_dim)
            neighbor_time_features = self.time_encoder(
                timestamps=torch.from_numpy(neighbor_delta_times).float().to(device)
            )

            # get edge features, shape (batch_size, num_neighbors, edge_feat_dim)
            neighbor_edge_features = self.edge_raw_features[
                torch.from_numpy(neighbor_edge_ids)
            ]
            # temporal graph convolution
            # Tensor, output shape (batch_size, node_feat_dim + time_feat_dim)
            output, _ = self.temporal_conv_layers[current_layer_num - 1](
                node_features=node_conv_features,
                node_time_features=node_time_features,
                neighbor_node_features=neighbor_node_conv_features,
                neighbor_node_time_features=neighbor_time_features,
                neighbor_node_edge_features=neighbor_edge_features,
                neighbor_masks=neighbor_node_ids,
            )

            # Tensor, output shape (batch_size, node_feat_dim)
            # follow the TGAT paper, use merge layer to combine the attention results and node original feature
            output = self.merge_layers[current_layer_num - 1](
                input_1=output, input_2=node_raw_features
            )

            return output
