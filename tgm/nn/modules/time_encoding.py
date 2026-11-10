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
        self.time_dim = time_dim
        self.w = torch.nn.Linear(1, time_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initialization from: https://github.com/yule-BUAA/DyGLib/blob/master/models/modules.py
        w = (1 / 10 ** np.linspace(0, 9, self.time_dim)).reshape(self.time_dim, 1)
        w_tensor = torch.as_tensor(
            w,
            dtype=self.w.weight.dtype,
            device=self.w.weight.device,
        )
        with torch.no_grad():
            self.w.weight.copy_(w_tensor)
            self.w.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(dim=-1).float()  # (batch_size, seq_len, 1)
        return torch.cos(self.w(x))
