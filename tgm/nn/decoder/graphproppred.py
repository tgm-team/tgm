from typing import Callable

import torch
import torch.nn as nn


def mean_pooling(z: torch.Tensor) -> torch.Tensor:
    r"""Default graph pooling: Mean pooling."""
    # @TODO: we can define this in different module and have a base class for this
    return torch.mean(z, dim=0).squeeze()


class GraphPredictor(torch.nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, graph_pooling: Callable = mean_pooling
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)
        self.graph_pooling = graph_pooling

    def forward(self, z_node: torch.Tensor) -> torch.Tensor:
        z_graph = self.graph_pooling(z_node)
        h = self.fc1(z_graph)
        h = h.relu()
        return self.fc2(h)
