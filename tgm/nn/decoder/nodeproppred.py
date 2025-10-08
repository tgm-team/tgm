import torch
import torch.nn as nn
from torch.nn import Sequential


class NodePredictor(torch.nn.Module):
    r"""Encoder for node property prediction.

    Args:
        in_dim (int): Dimension of input
        out_dim (int): Dimension of output
        hidden_dim (int): Size of hidden embedding
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        nlayers: int = 2,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.model = Sequential()
        self.model.append(nn.Linear(in_dim, hidden_dim))
        self.model.append(nn.ReLU())

        for i in range(1, nlayers - 1):
            self.model.append(nn.Linear(hidden_dim, hidden_dim))
            self.model.append(nn.ReLU())

        self.model.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, z_node: torch.Tensor) -> torch.Tensor:
        r"""Forward pass.

        Args:
            z_node (torch.Tensor): embedding of a node
        """
        return self.model(z_node)
