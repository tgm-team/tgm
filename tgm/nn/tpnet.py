from typing import Callable, Tuple

import torch
import torch.nn as nn

from .time_encoding import Time2Vec


class RandomProjectionModule(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_layer: float,
        time_decay_weight: float,
        beginning_time: torch.FloatType,
        use_matrix: bool = True,
        scale_random_projection: bool = True,  # @XXX: Use this variable with attenttion since it is opposite original implementation
        enforce_dim: int | None = None,
        num_edges: int | None = None,
        dim_factor: int | None = None,
        device: str = 'cpu',
    ) -> None:
        super(RandomProjectionModule, self).__init__()
        assert (not num_edges and not dim_factor) or not enforce_dim
        # @TODO
        raise Exception('Not yet implement')

    def forward(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        f"""Forward pass.
        Get pairwise feature for given source nodes and destination nodes.

        Args:
            X (PyTorch Float Tensor): Node features.
            src (PyTorch Tensor): Source node IDs
            dst (PyTorch Tensor): Destination node IDs.

        Returns:
            H_pairwise : Pairwise feature
        """
        # @TODO
        raise Exception('Not yet implement')


class FeedForwardNet(nn.Module):
    def __init__(
        self, input_dim: int, dim_expansion_factor: float, dropout: float = 0.0
    ) -> None:
        super(FeedForwardNet, self).__init__()
        # @TODO
        raise Exception('Not yet implement')

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        f"""Forward pass.

        Args:
            X (PyTorch Float Tensor): Input tensor

        Returns:
            H : Output tensor
        """
        # @TODO
        raise Exception('Not yet implement')


class MLPMixer(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        num_channels: int,
        token_dim_expansion_factor: float = 0.5,
        channel_dim_expansion_factor: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super(MLPMixer, self).__init__()
        # @TODO
        raise Exception('Not yet implement')

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        f"""Forward pass.

        Args:
            X (PyTorch Float Tensor): Input tensor

        Returns:
            H : Output tensor
        """
        # @TODO
        raise Exception('Not yet implement')


class TPNet(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        time_feat_dim: int,
        output_dim: int,
        num_nodes: int,
        dropout: float,
        num_layers: int,
        not_embedding: bool = False,
        device: str = 'cpu',
        time_encoder: Callable[..., nn.Module] = Time2Vec,
    ) -> None:
        super(TPNet, self).__init__()
        # @TODO
        raise Exception('Not yet implement')

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
        # @TODO
        raise Exception('Not yet implement')
