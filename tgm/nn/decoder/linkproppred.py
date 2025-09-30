from typing import Callable

import torch
import torch.nn as nn


def cat_merge(z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
    r"""Default merging operation: Concat."""
    # @TODO: we can define this in different module and have a base class for this
    return torch.cat([z_src, z_dst], dim=1)


class LinkPredictor(nn.Module):
    def __init__(self, dim: int, merge_op: Callable = cat_merge) -> None:
        super().__init__()
        self.merge_op = merge_op
        self.fc1 = nn.Linear(2 * dim, dim)
        self.fc2 = nn.Linear(dim, 1)

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        h = self.fc1(self.merge_op(z_src, z_dst))
        h = h.relu()
        return self.fc2(h).view(-1)
