import numpy as np
import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    def __init__(self, time_dim: int, requires_grad: bool = True) -> None:
        """Time encoder representation.

        Args:
            time_dim (int): The dimension of time encodings.
            requires_grad (bool): Whether the time encoder needs gradient.
        """
        super().__init__()
        self.time_dim = time_dim
        self.w = nn.Linear(1, time_dim)

        # Initialization from: https://github.com/yule-BUAA/DyGLib/blob/master/models/modules.py
        w = np.linspace(0, 0.9, time_dim, dtype=np.float32).reshape(time_dim, -1)
        self.w.weight = nn.Parameter(torch.from_numpy(w), requires_grad=requires_grad)
        self.w.bias = nn.Parameter(torch.zeros(time_dim), requires_grad=requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(dim=-1)  # (batch_size, seq_len, 1)
        x = self.w(x)  # (batch_size, seq_len, tim_dim)
        x = torch.cos(x)
        return x
