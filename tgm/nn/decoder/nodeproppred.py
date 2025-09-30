import torch
import torch.nn as nn


class NodePredictor(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, z_node: torch.Tensor) -> torch.Tensor:
        h = self.fc1(z_node)
        h = h.relu()
        return self.fc2(h)
