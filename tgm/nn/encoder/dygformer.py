"""Adapted from https://github.com/yule-BUAA/DyGLib_TGB."""

from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tgm.constants import PADDED_NODE_ID
from tgm.nn.modules import Time2Vec


class NeighborCooccurrenceEncoder(nn.Module):
    r"""An implementation of Neighbor Co-occurrence Encoding Scheme.

    Args:
        feat_dim (int): dimension of neighbor co-occurrence features (encodings).
        device (str): Device (cpu or gpu)

    Reference: https://arxiv.org/abs/2303.13047.
    """

    def __init__(self, feat_dim: int, device: str) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.device = device

        self.neighbor_co_occurrence_encoder = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.feat_dim, out_features=self.feat_dim),
        ).to(device)

    def _count_nodes_freq(
        self, src_nbrs: torch.Tensor, dst_nbrs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert src_nbrs.ndim == 2 and src_nbrs.shape == dst_nbrs.shape

        # cross occurrences count
        cross_mask = src_nbrs.unsqueeze(dim=1) == dst_nbrs.unsqueeze(dim=-1)

        # self occurrences count
        src_mask = src_nbrs.unsqueeze(dim=1) == src_nbrs.unsqueeze(dim=-1)
        dst_mask = dst_nbrs.unsqueeze(dim=1) == dst_nbrs.unsqueeze(dim=-1)

        src_freq = torch.stack([src_mask.sum(1), cross_mask.sum(1)], dim=2).float()
        dst_freq = torch.stack([dst_mask.sum(1), cross_mask.sum(2)], dim=2).float()

        # Mask out PADDED_NODE_ID
        src_freq[src_nbrs == PADDED_NODE_ID] = 0.0
        dst_freq[dst_nbrs == PADDED_NODE_ID] = 0.0
        return src_freq, dst_freq

    def forward(
        self,
        src_neighbour_nodes_ids: torch.Tensor,
        dst_neighbour_nodes_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass. Encode neighbor co-occurrence (Section 4.1).

        Args:
            src_neighbour_nodes_ids (Tensor): Padded list of source node's neighbour.
            dst_neighbour_nodes_ids (Tensor): Padded list of destination node's neighbour.

        Returns:
            X (PyTorch Float Tensor): Neighbor co-occurrence features (`X^{t}_{*,C}`).
        """
        source_freq, dst_freq = self._count_nodes_freq(
            src_neighbour_nodes_ids, dst_neighbour_nodes_ids
        )
        src_neighbors_co_occurrence_feat = self.neighbor_co_occurrence_encoder(
            source_freq.unsqueeze(dim=-1)
        ).sum(dim=2)
        dst_neighbors_co_occurrence_feat = self.neighbor_co_occurrence_encoder(
            dst_freq.unsqueeze(dim=-1)
        ).sum(dim=2)
        return src_neighbors_co_occurrence_feat, dst_neighbors_co_occurrence_feat


class TransformerEncoder(nn.Module):
    r"""An implementation of Transformer Encoder.

    Args:
        attention_dim (int): dimension of the attention vector.
        num_heads (int): number of attention heads.
        dropout (float): dropout rate.

    Reference: https://arxiv.org/abs/2303.13047.
    """

    def __init__(
        self, attention_dim: int, num_heads: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout

        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=attention_dim, num_heads=num_heads, dropout=dropout
        )

        self.dropout = nn.Dropout(self.dropout_rate)

        self.linear_layers = nn.ModuleList(
            [
                nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
                nn.Linear(in_features=4 * attention_dim, out_features=attention_dim),
            ]
        )
        self.norm_layers = nn.ModuleList(
            [nn.LayerNorm(attention_dim), nn.LayerNorm(attention_dim)]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward pass. Encode the inputs by Transformer encoder (Section 4.1).

        Args:
            inputs (PyTorch Float Tensor): `Z^{t} = [Z^{t}_u, Z^{t}_v]`.

        Returns:
            H (PyTorch Float Tensor): Representations of all nodes.
        """
        transposed_inputs = inputs.transpose(0, 1)
        transposed_inputs = self.norm_layers[0](transposed_inputs)

        # E.q 5 - Section 4.1
        hidden_states = self.multi_head_attention(
            query=transposed_inputs, key=transposed_inputs, value=transposed_inputs
        )[0].transpose(0, 1)

        # E.q 6 - Section 4.1
        outputs = inputs + self.dropout(hidden_states)

        # E.q 7 - Section 4.1
        hidden_states = self.linear_layers[1](
            self.dropout(F.gelu(self.linear_layers[0](self.norm_layers[1](outputs))))
        )

        # E.q 7 - Section 4.1
        outputs = outputs + self.dropout(hidden_states)

        return outputs


class DyGFormer(nn.Module):
    r"""An implementation of DyGFormer.

    Args:
        node_feat_dim (int): Dimension of static/dynamic node features (`d_N`).
        edge_feat_dim (int): Dimension of edge features (`d_E`).
        time_feat_dim (int): Dimension of time encodings (`d_T`).
        channel_embedding_dim (int): Dimension of each channel embedding.
        output_dim (int): Dimension of output embedding.
        patch_size (int): Path size (`\mathbf{P}`).
        num_layers (int): Number of transformer layers.
        num_heads (int): Number of attention heads.
        dropout (float): Drop out rate.
        max_input_sequence_length (int): maximal length of the input sequence for each node.
        time_encoder (PyTorch Module) : Time encoder module.
        device (str) : cpu or cuda

    Reference: https://arxiv.org/abs/2303.13047.
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        time_feat_dim: int,
        channel_embedding_dim: int,
        output_dim: int = 172,
        patch_size: int = 1,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        max_input_sequence_length: int = 512,
        num_channels: int = 4,
        time_encoder: Callable[..., nn.Module] = Time2Vec,
        device: str = 'cpu',
    ) -> None:
        super().__init__()
        if max_input_sequence_length % patch_size != 0:
            raise ValueError('Max sequence length must be a multiple of path size')

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.patch_size = patch_size
        self.max_input_sequence_length = max_input_sequence_length
        self.neighbor_co_occurrence_feat_dim = self.channel_embedding_dim
        self.device = device
        self.num_channels = num_channels
        self.num_patches = max_input_sequence_length // patch_size

        self.time_encoder = time_encoder(time_feat_dim)
        self.co_occurrence_encoder = NeighborCooccurrenceEncoder(
            feat_dim=self.neighbor_co_occurrence_feat_dim,
            device=self.device,
        )
        self.projection_layer = nn.ModuleDict(
            {
                'node': nn.Linear(
                    in_features=self.patch_size * self.node_feat_dim,
                    out_features=self.channel_embedding_dim,
                    bias=True,
                ),
                'edge': nn.Linear(
                    in_features=self.patch_size * self.edge_feat_dim,
                    out_features=self.channel_embedding_dim,
                    bias=True,
                ),
                'time': nn.Linear(
                    in_features=self.patch_size * self.time_feat_dim,
                    out_features=self.channel_embedding_dim,
                    bias=True,
                ),
                'neighbor_co_occurrence': nn.Linear(
                    in_features=self.patch_size * self.neighbor_co_occurrence_feat_dim,
                    out_features=self.channel_embedding_dim,
                    bias=True,
                ),
            }
        ).to(device)
        self.transformers = nn.ModuleList(
            [
                TransformerEncoder(
                    attention_dim=self.num_channels * self.channel_embedding_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        ).to(device)

        self.output_layer = nn.Linear(
            in_features=self.num_channels * self.channel_embedding_dim,
            out_features=output_dim,
            bias=True,
        ).to(device)

    def forward(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_time: torch.Tensor,
        neighbours: torch.Tensor,
        neighbours_time: torch.Tensor,
        neighbours_edge_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        f"""Forward pass.

        Args:
            X (PyTorch Float Tensor): Node features.
            edge_index (PyTorch Tensor): Graph edge indices.
            edge_feat (PyTorch Tensor): Edge feature vector.
            neighbours (PyTorch Tensor): Neighbours of src and dst nodes from edge_index
            neighbours_time (PyTorch Tensor): Interaction time of src/dst nodes and their neighbours
            neighbours_edge_feat (PyTorch Tensor): Features of edge between src/dst nodes and their neighbours

        Returns:
            H_source,H_dest (PyTorch Float Tensor): Time-aware representations of src and dst nodes.

        *Note: Information of about neighbours of both src and dst nodes are concated together. Neighbour information of all src nodes comes first, then that of all dst nodes*
        """
        src, dst = edge_index[0], edge_index[1]
        num_edge = src.shape[0]
        src_neighbours = neighbours[:num_edge]
        dst_neighbours = neighbours[num_edge : num_edge * 2]
        src_neighbours_time = neighbours_time[:num_edge]
        dst_neighbours_time = neighbours_time[num_edge : num_edge * 2]
        src_neighbours_edge_feat = neighbours_edge_feat[:num_edge]
        dst_neighbours_edge_feat = neighbours_edge_feat[num_edge : num_edge * 2]

        # include seed nodes are neighbors themselves
        src_neighbours = torch.cat([src[:, None], src_neighbours], dim=1)
        dst_neighbours = torch.cat([dst[:, None], dst_neighbours], dim=1)

        src_neighbours_time = torch.cat(
            [edge_time[:, None], src_neighbours_time], dim=1
        )
        dst_neighbours_time = torch.cat(
            [edge_time[:, None], dst_neighbours_time], dim=1
        )

        padding = torch.zeros(
            src_neighbours_edge_feat.shape[0],
            1,
            src_neighbours_edge_feat.shape[2],
            device=self.device,
            dtype=src_neighbours_edge_feat.dtype,
        )
        src_neighbours_edge_feat = torch.cat([padding, src_neighbours_edge_feat], dim=1)
        dst_neighbours_edge_feat = torch.cat([padding, dst_neighbours_edge_feat], dim=1)

        # Get node feat and time feat using Time Encoder
        src_neighbours_node_feats = X[src_neighbours, :]
        dst_neighbours_node_feats = X[dst_neighbours, :]
        src_neighbours_node_feats[src_neighbours == PADDED_NODE_ID] = 0
        dst_neighbours_node_feats[dst_neighbours == PADDED_NODE_ID] = 0

        src_neighbours_time_feats = self.time_encoder(
            edge_time.unsqueeze(1) - src_neighbours_time
        )
        dst_neighbours_time_feats = self.time_encoder(
            edge_time.unsqueeze(1) - dst_neighbours_time
        )

        src_neighbours_time_feats[(src_neighbours == PADDED_NODE_ID)] = 0

        dst_neighbours_time_feats[(dst_neighbours == PADDED_NODE_ID)] = 0

        src_co_occurrence_feats, dst_co_occurrence_feats = self.co_occurrence_encoder(
            src_neighbours, dst_neighbours
        )

        # Get patches for each features of src and dst
        neighbours_node_feats = self._get_patches(
            torch.cat([src_neighbours_node_feats, dst_neighbours_node_feats], dim=0)
        )
        neighbours_edge_feats = self._get_patches(
            torch.cat([src_neighbours_edge_feat, dst_neighbours_edge_feat], dim=0)
        )
        neighbours_time_feats = self._get_patches(
            torch.cat([src_neighbours_time_feats, dst_neighbours_time_feats], dim=0)
        )
        co_occurrence_feats = self._get_patches(
            torch.cat([src_co_occurrence_feats, dst_co_occurrence_feats], dim=0)
        )

        src_neighbours_node_features_patches = neighbours_node_feats[:num_edge]
        src_neighbours_edge_features_patches = neighbours_edge_feats[:num_edge]
        src_neighbours_time_features_patches = neighbours_time_feats[:num_edge]
        src_co_occurence_features_patches = co_occurrence_feats[:num_edge]

        dst_neighbours_node_features_patches = neighbours_node_feats[num_edge:]
        dst_neighbours_edge_features_patches = neighbours_edge_feats[num_edge:]
        dst_neighbours_time_features_patches = neighbours_time_feats[num_edge:]
        dst_co_occurence_features_patches = co_occurrence_feats[num_edge:]

        # Use projection to align the patch encoding dimension for both dst and src
        src_neighbours_node_features_patches = self.projection_layer['node'](
            src_neighbours_node_features_patches
        )
        src_neighbours_edge_features_patches = self.projection_layer['edge'](
            src_neighbours_edge_features_patches
        )
        src_neighbours_time_features_patches = self.projection_layer['time'](
            src_neighbours_time_features_patches
        )
        src_co_occurence_features_patches = self.projection_layer[
            'neighbor_co_occurrence'
        ](src_co_occurence_features_patches)

        # Tensor, shape (batch_size, dst_num_patches, channel_embedding_dim)
        dst_neighbours_node_features_patches = self.projection_layer['node'](
            dst_neighbours_node_features_patches
        )
        dst_neighbours_edge_features_patches = self.projection_layer['edge'](
            dst_neighbours_edge_features_patches
        )
        dst_neighbours_time_features_patches = self.projection_layer['time'](
            dst_neighbours_time_features_patches
        )
        dst_co_occurence_features_patches = self.projection_layer[
            'neighbor_co_occurrence'
        ](dst_co_occurence_features_patches)

        # Perform transformer operation
        batch_size = len(src_neighbours_node_features_patches)
        src_num_patches = src_neighbours_node_features_patches.shape[1]
        dst_num_patches = dst_neighbours_node_features_patches.shape[1]

        patches_nodes_neighbor_node_raw_features = torch.cat(
            [
                src_neighbours_node_features_patches,
                dst_neighbours_node_features_patches,
            ],
            dim=1,
        )
        patches_nodes_edge_raw_features = torch.cat(
            [
                src_neighbours_edge_features_patches,
                dst_neighbours_edge_features_patches,
            ],
            dim=1,
        )
        patches_nodes_neighbor_time_features = torch.cat(
            [
                src_neighbours_time_features_patches,
                dst_neighbours_time_features_patches,
            ],
            dim=1,
        )
        patches_nodes_neighbor_co_occurrence_features = torch.cat(
            [
                src_co_occurence_features_patches,
                dst_co_occurence_features_patches,
            ],
            dim=1,
        )

        patches_data = torch.stack(
            [
                patches_nodes_neighbor_node_raw_features,
                patches_nodes_edge_raw_features,
                patches_nodes_neighbor_time_features,
                patches_nodes_neighbor_co_occurrence_features,
            ],
            dim=2,
        )
        patches_data = patches_data.reshape(
            batch_size,
            src_num_patches + dst_num_patches,
            self.num_channels * self.channel_embedding_dim,
        )

        for transformer in self.transformers:
            patches_data = transformer(patches_data)

        src_patches_data = patches_data[:, :src_num_patches, :]
        dst_patches_data = patches_data[
            :, src_num_patches : src_num_patches + dst_num_patches, :
        ]
        src_patches_data = torch.mean(src_patches_data, dim=1)
        dst_patches_data = torch.mean(dst_patches_data, dim=1)

        src_node_embeddings = self.output_layer(src_patches_data)
        dst_node_embeddings = self.output_layer(dst_patches_data)
        return src_node_embeddings, dst_node_embeddings

    def _get_patches(self, feat: torch.Tensor) -> torch.Tensor:
        list_patches = []
        for patch_id in range(self.num_patches):
            start_idx = patch_id * self.patch_size
            end_idx = patch_id * self.patch_size + self.patch_size
            list_patches.append(feat[:, start_idx:end_idx, :])

        patches_feats = torch.stack(list_patches, dim=1).reshape(
            feat.shape[0], self.num_patches, self.patch_size * feat.shape[2]
        )

        return patches_feats
