from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseMerge(ABC, nn.Module):
    r"""Base class for merge operation of source and destination nodes embeddings.

    Given 2 embeddings, this operation merges provided embeddings and return a single embeddings with the dimension that can be retrieved from `out_channels`
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def out_channels(self) -> int:
        r"""Output feature dimension after merging."""
        ...

    @abstractmethod
    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        r"""Merge source and destination node embeddings into a single representation.

        Args:
            z_src (torch.Tensor): Source node embeddings of shape (N, D), where N is
                the number of edges and D is the node embedding dimension.
            z_dst (torch.Tensor): Destination node embeddings of shape (N, D), where N is
                the number of edges and D is the node embedding dimension.

        Returns:
            torch.Tensor: Merged embedding of shape (N, out_channels).
        """
        ...


class ConcatMerge(BaseMerge):
    r"""Concat the provided embeddings into a single embedding and return it."""

    def __init__(self, node_dim: int):
        super().__init__()
        self.node_dim = node_dim

    @property
    def out_channels(self) -> int:
        r"""Dimension of the returned embedding after merging."""
        return self.node_dim * 2

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        r"""Merge source and destination node embeddings into a single representation.

        Args:
            z_src (torch.Tensor): Source node embeddings of shape (N, D), where N is
                the number of edges and D is the node embedding dimension.
            z_dst (torch.Tensor): Destination node embeddings of shape (N, D), where N is
                the number of edges and D is the node embedding dimension.

        Returns:
            torch.Tensor: Merged embedding of shape (N, out_channels).
        """
        return torch.cat([z_src, z_dst], dim=1)


class LearnableSumMerge(BaseMerge):
    r"""Sum node-level embeddings after a linear projection."""

    def __init__(self, node_dim: int) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.lin_src = nn.Linear(node_dim, node_dim)
        self.lin_dst = nn.Linear(node_dim, node_dim)

    @property
    def out_channels(self) -> int:
        r"""Dimension of the returned embedding after merging."""
        return self.node_dim

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        r"""Merge source and destination node embeddings into a single representation.

        Args:
            z_src (torch.Tensor): Source node embeddings of shape (N, D), where N is
                the number of edges and D is the node embedding dimension.
            z_dst (torch.Tensor): Destination node embeddings of shape (N, D), where N is
                the number of edges and D is the node embedding dimension.

        Returns:
            torch.Tensor: Merged embedding of shape (N, out_channels).
        """
        return self.lin_src(z_src) + self.lin_dst(z_dst)
