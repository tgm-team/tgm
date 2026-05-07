from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseEmbdPooling(ABC, nn.Module):
    r"""Base class for pooling a set of node embeddings into a single embedding.

    Given a matrix of node embeddings (NxD), this operation aggregates them along
    the node dimension and returns a single embedding vector.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        r"""Aggregate node embeddings into a single representation.

        Args:
            z (torch.Tensor): Node embeddings of shape (N, D), where N is the
                number of nodes and D is the embedding dimension.

        Returns:
            torch.Tensor: Pooled embedding of shape (D,).
        """
        ...


class MeanEmbdPooling(BaseEmbdPooling):
    r"""Pools node embeddings by taking their element-wise mean."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        r"""Aggregate node embeddings into a single representation.

        Args:
            z (torch.Tensor): Node embeddings of shape (N, D), where N is the
                number of nodes and D is the embedding dimension.

        Returns:
            torch.Tensor: Mean-pooled embedding of shape (D,).
        """
        return torch.mean(z, dim=0).squeeze()


class SumEmbdPooling(BaseEmbdPooling):
    r"""Pools node embeddings by taking their element-wise sum."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        r"""Aggregate node embeddings into a single representation.

        Args:
            z (torch.Tensor): Node embeddings of shape (N, D), where N is the
                number of nodes and D is the embedding dimension.

        Returns:
            torch.Tensor: Sum-pooled embedding of shape (D,).
        """
        return torch.sum(z, dim=0).squeeze()
