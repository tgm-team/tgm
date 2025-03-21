import torch
import torch.nn as nn


class MergeLayer(nn.Module):
    def __init__(
        self, in_dim1: int, in_dim2: int, hidden_dim: int, output_dim: int
    ) -> None:
        """Merge two input layers via: in_dim1 + in_dim2 -> hidden_dim -> output_dim."""
        super().__init__()
        self.fc1 = nn.Linear(in_dim1 + in_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x1, x2], dim=1)  # (*, in_dim1 + in_dim2)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
