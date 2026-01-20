from typing import Callable, Dict

import torch
import torch.nn as nn
from torch.nn import Sequential


def cat_merge(z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
    r"""Default merging operation: Concat."""
    # @TODO: we can define this in different module and have a base class for this
    return torch.cat([z_src, z_dst], dim=1)


class LearnableSumMerge(nn.Module):
    r"""Sum node-level embeddings after a linear projection."""

    def __init__(self, node_dim: int) -> None:
        super().__init__()
        self.lin_src = nn.Linear(node_dim, node_dim)
        self.lin_dst = nn.Linear(node_dim, node_dim)

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        return self.lin_src(z_src) + self.lin_dst(z_dst)


MERGE_OP: Dict[str, Callable] = {
    'concat': cat_merge,
    'sum': LearnableSumMerge,
}


class LinkPredictor(torch.nn.Module):
    r"""Compute edge embedding given src and dst node embeddings.

    Args:
        node_dim (int): Dimension of node embedding
        out_dim (int): Dimension of output
        nlayers (int): Number of layers
        hidden_dim (int): Size of each hidden embedding
        merge_op (str): Operation to merge 2 node embeddings (concat)
    """

    def __init__(
        self,
        node_dim: int,
        out_dim: int = 1,
        nlayers: int = 2,
        hidden_dim: int = 64,
        merge_op: str = 'concat',
    ) -> None:
        super().__init__()

        if merge_op not in MERGE_OP:
            raise ValueError(
                f'{merge_op} merge operations is not support. Please choose from {list(MERGE_OP.keys())}'
            )

        if merge_op == 'concat':
            self.merge = MERGE_OP[merge_op]
            in_dim = node_dim * 2
        else:
            self.merge = MERGE_OP[merge_op](node_dim)
            in_dim = node_dim

        self.model = Sequential()
        self.model.append(nn.Linear(in_dim, hidden_dim))
        self.model.append(nn.ReLU())

        for _ in range(1, nlayers - 1):
            self.model.append(nn.Linear(hidden_dim, hidden_dim))
            self.model.append(nn.ReLU())

        self.model.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        r"""Forward pass.

        Args:
            z_src (torch.Tensor): embedding of src node
            z_dst (torch.Tensor): embedding of dst node
        """
        h = self.merge(z_src, z_dst)
        return self.model(h).view(-1)
