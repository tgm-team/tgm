import numpy as np
import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    def __init__(self, time_dim: int) -> None:
        """Time encoder representation.

        Args:
            time_dim (int): The dimension of time encodings.
        """
        super().__init__()
        self.w = torch.nn.Linear(1, time_dim)

        # Initialization from: https://github.com/yule-BUAA/DyGLib/blob/master/models/modules.py
        w = (1 / 10 ** np.linspace(0, 9, time_dim)).reshape(time_dim, 1)
        self.w.weight = torch.nn.Parameter(torch.from_numpy(w).float())
        self.w.bias = torch.nn.Parameter(torch.zeros(time_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(dim=-1).float()  # (batch_size, seq_len, 1)
        return torch.cos(self.w(x))
