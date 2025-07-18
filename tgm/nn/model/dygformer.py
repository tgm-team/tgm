"""Adapted from https://github.com/yule-BUAA/DyGLib_TGB."""

from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..time_encoding import Time2Vec


class NeighborCooccurrenceEncoder(nn.Module):
    r"""An implementation of Neighbor Co-occurrence Encoding Scheme.

    Args:
        feat_dim (int): dimension of neighbor co-occurrence features (encodings).
        device (str): Device (cpu or gpu)

    Reference: https://arxiv.org/abs/2303.13047.
    """

    def __init__(self, feat_dim: int, device: str) -> None:
        super(NeighborCooccurrenceEncoder, self).__init__()
        self.feat_dim = feat_dim
        self.device = device

        self.neighbor_co_occurrence_encoder = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.feat_dim, out_features=self.feat_dim),
        )

    def _count_nodes_freq(
        self, all_sources_neighbors: np.ndarray, all_dsts_neighbors: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(all_sources_neighbors.shape) == 2
        assert (
            all_sources_neighbors.shape[0] == all_dsts_neighbors.shape[0]
            and all_sources_neighbors.shape[1] == all_dsts_neighbors.shape[1]
        )

        source_freq, dst_freq = [], []
        for src_neighbors, dst_neighbors in zip(
            all_sources_neighbors, all_dsts_neighbors
        ):
            src_unique_keys, src_inverse_indices, src_counts = np.unique(
                src_neighbors, return_inverse=True, return_counts=True
            )
            # Fequency of each source's neighbor within source's neighbors
            src_neighbors_freq_src_neighbors = (
                torch.from_numpy(src_counts[src_inverse_indices])
                .float()
                .to(self.device)
            )
            src_mapping_dict = dict(zip(src_unique_keys, src_counts))

            dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(
                dst_neighbors, return_inverse=True, return_counts=True
            )
            # Fequency of each destination's neighbor within destination's neighbors
            dst_neighbors_freq_dst_neighbors = (
                torch.from_numpy(dst_counts[dst_inverse_indices])
                .float()
                .to(self.device)
            )
            dst_mapping_dict = dict(zip(dst_unique_keys, dst_counts))

            # Fequency of each source's neighbor within destination's neighbors
            src_neighbors_freq_dst_neighbors = (
                torch.from_numpy(src_neighbors.copy())
                .apply_(lambda neighbor_id: dst_mapping_dict.get(neighbor_id, 0.0))
                .float()
                .to(self.device)
            )
            # Fequency of each source's neighbor within destination's neighbors
            dst_neighbors_freq_src_neighbors = (
                torch.from_numpy(dst_neighbors.copy())
                .apply_(lambda neighbor_id: src_mapping_dict.get(neighbor_id, 0.0))
                .float()
                .to(self.device)
            )

            source_freq.append(
                torch.stack(
                    [
                        src_neighbors_freq_src_neighbors,
                        src_neighbors_freq_dst_neighbors,
                    ],
                    dim=1,
                )
            )
            dst_freq.append(
                torch.stack(
                    [
                        dst_neighbors_freq_dst_neighbors,
                        dst_neighbors_freq_src_neighbors,
                    ],
                    dim=1,
                )
            )

        source_freq_tensor = torch.stack(source_freq, dim=0)
        dst_freq_tensor = torch.stack(dst_freq, dim=0)

        # set the frequencies of the padded nodes (with zero index) to zeros
        source_freq_tensor[torch.from_numpy(all_sources_neighbors == 0)] = 0.0
        dst_freq_tensor[torch.from_numpy(all_dsts_neighbors == 0)] = 0.0
        return source_freq_tensor, dst_freq_tensor

    def forward(
        self, src_neighbour_nodes_ids: np.ndarray, dst_neighbour_nodes_ids: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass. Encode neighbor co-occurrence (Section 4.1).

        Args:
            src_neighbour_nodes_ids (Numpy array): Padded list of source node's neighbour.
            dst_neighbour_nodes_ids (Numpy array): Padded list of destination node's neighbour.

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
        super(TransformerEncoder, self).__init__()
        # @TODO
        raise Exception('Not implemented yet')

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward pass. Encode the inputs by Transformer encoder (Section 4.1).

        Args:
            inputs (PyTorch Float Tensor): `Z^{t} = [Z^{t}_u, Z^{t}_v]`.

        Returns:
            H (PyTorch Float Tensor): Representations of all nodes.
        """
        # @TODO
        raise Exception('Not implemented yet')


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
        time_encoder: Callable[..., nn.Module] = Time2Vec,
    ) -> None:
        super(DyGFormer, self).__init__()
        # @TODO
        raise Exception('Not implemented yet')

    def forward(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
    ) -> torch.Tensor:
        f"""Forward pass.

        Args:
            X (PyTorch Float Tensor): Node features.
            edge_index (PyTorch Long Tensor): Graph edge indices.
            edge_feat (PyTorch Long Tensor): Edge feature vector.

        Returns:
            H (PyTorch Float Tensor): Time-aware representations of all nodes.
        """
        # @TODO
        raise Exception('Not implemented yet')
