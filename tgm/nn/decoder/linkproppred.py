import torch
import torch.nn as nn
from torch.nn import Sequential

from ..modules import BaseMerge, ConcatMerge


class LinkPredictor(torch.nn.Module):
    r"""Compute edge embedding given src and dst node embeddings.

    Args:
        node_dim (int): Dimension of node embedding
        out_dim (int): Dimension of output
        nlayers (int): Number of layers
        hidden_dim (int): Size of each hidden embedding
        merge_op (MergeBase): Operation to merge 2 node embeddings (ConcatMerge by default)

    Note:
        merge_op can be selected from [ConcatMerge, LearnableSumMerge] or any custom merging operation, provided it subclasses `BaseMerge`.
    """

    def __init__(
        self,
        node_dim: int,
        out_dim: int = 1,
        nlayers: int = 2,
        hidden_dim: int = 64,
        merge_op: BaseMerge | None = None,
    ) -> None:
        super().__init__()

        self.merge = (
            merge_op if merge_op is not None else ConcatMerge(node_dim=node_dim)
        )

        in_dim = self.merge.out_channels

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
