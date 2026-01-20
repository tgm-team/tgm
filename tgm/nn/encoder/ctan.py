import torch
import torch.nn as nn
from torch_geometric.nn import AntiSymmetricConv, TransformerConv
from torch_geometric.nn.models.tgn import TimeEncoder


class CTAN(torch.nn.Module):
    def __init__(
        self,
        edge_dim: int,
        memory_dim: int,
        time_dim: int,
        node_dim: int,
        num_iters: int = 1,
        mean_delta_t: float = 0.0,
        std_delta_t: float = 1.0,
        epsilon: float = 0.1,
        gamma: float = 0.1,
    ) -> None:
        super().__init__()
        self.mean_delta_t = mean_delta_t
        self.std_delta_t = std_delta_t
        self.time_enc = TimeEncoder(time_dim)
        self.enc_x = nn.Linear(memory_dim + node_dim, memory_dim)

        phi = TransformerConv(
            memory_dim, memory_dim, edge_dim=edge_dim + time_dim, root_weight=False
        )
        self.aconv = AntiSymmetricConv(
            memory_dim, phi, num_iters=num_iters, epsilon=epsilon, gamma=gamma
        )

    def forward(
        self,
        x: torch.Tensor,
        last_update: torch.Tensor,
        edge_index: torch.Tensor,
        t: torch.Tensor,
        msg: torch.Tensor,
    ) -> torch.Tensor:
        rel_t = (last_update[edge_index[0]] - t).abs()
        rel_t = ((rel_t - self.mean_delta_t) / self.std_delta_t).to(x.dtype)
        enc_x = self.enc_x(x)
        edge_attr = torch.cat([msg, self.time_enc(rel_t)], dim=-1)
        z = self.aconv(enc_x, edge_index, edge_attr=edge_attr)
        z = torch.tanh(z)
        return z
