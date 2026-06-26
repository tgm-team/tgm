from typing import Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class Aggregator(Protocol):
    r"""The expected behaviors of aggregator."""

    @property
    def out_channels(self) -> int:
        r"""Output feature dimension produced by this transformation."""
        ...

    def __call__(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        r"""Apply the transformation to one or more node embedding tensors.

        Args:
            *args (torch.Tensor): One or more node embedding matrices.

            **kwargs (torch.Tensor): One or more node embedding matrices.

        Returns:
            torch.Tensor: Transformed embeddings whose last dimension
            equals :attr:`out_channels`.
        """
        ...


class ConcatMerge:
    r"""Concat the provided embeddings into a single embedding and return it."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    @property
    def out_channels(self) -> int:
        r"""Dimension of the returned embedding after merging."""
        return self.dim * 2

    def __call__(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
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


class LearnableSumMerge(nn.Module):
    r"""Sum node-level embeddings after a linear projection."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.lin_src = nn.Linear(dim, dim)
        self.lin_dst = nn.Linear(dim, dim)

    @property
    def out_channels(self) -> int:
        r"""Dimension of the returned embedding after merging."""
        return self.dim

    def __call__(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
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


class MeanEmbdPooling:
    r"""Pools node embeddings by taking their element-wise mean."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    @property
    def out_channels(self) -> int:
        r"""Dimension of the returned embedding after merging."""
        return self.dim

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        r"""Aggregate node embeddings into a single representation.

        Args:
            z (torch.Tensor): Node embeddings of shape (N, D), where N is the
                number of nodes and D is the embedding dimension.

        Returns:
            torch.Tensor: Mean-pooled embedding of shape (D,).
        """
        return torch.mean(z, dim=0).squeeze()


class SumEmbdPooling:
    r"""Pools node embeddings by taking their element-wise sum."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    @property
    def out_channels(self) -> int:
        r"""Dimension of the returned embedding after merging."""
        return self.dim

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        r"""Aggregate node embeddings into a single representation.

        Args:
            z (torch.Tensor): Node embeddings of shape (N, D), where N is the
                number of nodes and D is the embedding dimension.

        Returns:
            torch.Tensor: Sum-pooled embedding of shape (D,).
        """
        return torch.sum(z, dim=0).squeeze()
