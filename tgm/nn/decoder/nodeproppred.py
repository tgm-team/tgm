import torch
import torch.nn as nn
from torch.nn import Sequential


class NodePredictor(torch.nn.Module):
    r"""Encoder for node property prediction.

    Args:
        in_dim (int): Dimension of input
        out_dim (int): Dimension of output
        hids_sizes (List[int]): Size of each hidden embeddings
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        nlayers: int = 2,
        hids_sizes: int = 32,
    ) -> None:
        super().__init__()

        self.model = Sequential()
        self.model.append(nn.Linear(in_dim, hids_sizes))
        self.model.append(nn.ReLU())

        for i in range(1, nlayers - 1):
            self.model.append(nn.Linear(hids_sizes, hids_sizes))
            self.model.append(nn.ReLU())

        self.model.append(nn.Linear(hids_sizes, out_dim))

    def forward(self, z_node: torch.Tensor) -> torch.Tensor:
        r"""Forward pass.

        Args:
            z_node (torch.Tensor): embedding of a node
        """
        return self.model(z_node)
