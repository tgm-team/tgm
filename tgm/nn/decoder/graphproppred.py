import torch
import torch.nn as nn
from torch.nn import Sequential


def mean_pooling(z: torch.Tensor) -> torch.Tensor:
    r"""Default graph pooling: Mean pooling."""
    # @TODO: we can define this in different module and have a base class for this
    return torch.mean(z, dim=0).squeeze()


def sum_pooling(z: torch.Tensor) -> torch.Tensor:
    r"""Default graph pooling: Sunm pooling."""
    # @TODO: we can define this in different module and have a base class for this
    return torch.sum(z, dim=0).squeeze()


POOLING_OP = {'mean': mean_pooling, 'sum': sum_pooling}


class GraphPredictor(torch.nn.Module):
    r"""Perform pooling over provided node features and perform graph level task.

    Args:
        in_dim (int): Dimension of input
        out_dim (int): Dimension of output
        nlayers (int): Number of layers
        hidden_dim (int): Size of each hidden embeddings
        graph_pooling (str): graph pooling operation (mean, sum.)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        nlayers: int = 2,
        hidden_dim: int = 64,
        graph_pooling: str = 'mean',
    ) -> None:
        super().__init__()
        if graph_pooling not in POOLING_OP:
            raise ValueError(
                f'{graph_pooling} pooling operations is not supported. Please choose from {list(POOLING_OP.keys())}'
            )

        self.model = Sequential()
        self.model.append(nn.Linear(in_dim, hidden_dim))
        self.model.append(nn.ReLU())

        for i in range(1, nlayers - 1):
            self.model.append(nn.Linear(hidden_dim, hidden_dim))
            self.model.append(nn.ReLU())

        self.model.append(nn.Linear(hidden_dim, out_dim))
        self.graph_pooling = POOLING_OP[graph_pooling]

    def forward(self, z_nodes: torch.Tensor) -> torch.Tensor:
        r"""Forward pass.

        Args:
            z_nodes (torch.Tensor): embedding of nodes
        """
        z_graph = self.graph_pooling(z_nodes)
        return self.model(z_graph)
