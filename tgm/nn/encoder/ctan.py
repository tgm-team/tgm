from typing import Callable, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import AntiSymmetricConv, TransformerConv
from torch_geometric.nn.inits import ones, zeros
from torch_geometric.nn.models.tgn import TimeEncoder
from torch_geometric.utils import scatter


class CTAN(torch.nn.Module):
    """An implementation of CTAN.

    Args:
        edge_dim (int): Dimension of edge features.
        memory_dim (int): Dimension of memory embeddings.
        time_dim (int): Dimension of time encodings.
        node_dim (int): Dimension of static/dynamic node features.
        num_iters (int): Number of AntiSymmetricConv layers.
        mean_delta_t (float): Mean delta time between edge events (used to normalize time signal).
        std_delta_t (float): Std delta time between edge events (used to normalize time signal).
        epsilon (float): Discretization step size for AntiSymmetricConv.
        gamma (float): The strength of the diffusion in the AntiSymmetricConv.

    Reference: https://arxiv.org/abs/2406.02740
    """

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
        """Forward pass.

        Args:
            x (PyTorch Float Tensor): Node features.
            last_update (PyTorch Tensor): Last memory update timestamps.
            edge_index (PyTorch Tensor): Graph edge indices.
            t (PyTorch Tensor): Graph edge timestamps.
            msg (PyTorch Tensor): Memory embeddings.

        Returns:
            (PyTorch Float Tensor): Embeddings for the batch of node ids.
        """
        rel_t = (last_update[edge_index[0]] - t).abs()
        rel_t = ((rel_t - self.mean_delta_t) / self.std_delta_t).to(x.dtype)
        enc_x = self.enc_x(x)
        edge_attr = torch.cat([msg, self.time_enc(rel_t)], dim=-1)
        z = self.aconv(enc_x, edge_index, edge_attr=edge_attr)
        z = torch.tanh(z)
        return z


class CTANMemory(torch.nn.Module):
    """The CTAN Memory model.

    Args:
        num_nodes (int): The number of nodes to save memories for.
        memory_dim (int): The hidden memory dimensionality.
        aggr_module (Callable): The message aggregator function
            which aggregates messages to the same destination into a single
            representation.
        init_time (int): Start time of the graph, used during memory reset.

    Reference: https://arxiv.org/abs/2406.02740
    """

    def __init__(
        self, num_nodes: int, memory_dim: int, aggr_module: Callable, init_time: int = 0
    ) -> None:
        super().__init__()

        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.init_time = init_time
        self.aggr_module = aggr_module

        self.register_buffer('memory', torch.zeros(num_nodes, memory_dim))
        self.register_buffer(
            'last_update', torch.ones(self.num_nodes, dtype=torch.long) * init_time
        )
        self.register_buffer('_assoc', torch.empty(num_nodes, dtype=torch.long))

    def reset_parameters(self) -> None:
        if hasattr(self.aggr_module, 'reset_parameters'):
            self.aggr_module.reset_parameters()
        self.reset_state()

    def reset_state(self) -> None:
        zeros(self.memory)
        ones(self.last_update)
        self.last_update *= self.init_time

    def detach(self) -> None:
        self.memory.detach_()

    def forward(self, n_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.memory[n_id], self.last_update[n_id]

    def update_state(
        self,
        src: torch.Tensor,
        pos_dst: torch.Tensor,
        t: torch.Tensor,
        src_emb: torch.Tensor,
        pos_dst_emb: torch.Tensor,
    ) -> None:
        idx = torch.cat([src, pos_dst], dim=0)
        _idx = idx.unique()
        self._assoc[_idx] = torch.arange(_idx.size(0), device=_idx.device)

        t = torch.cat([t, t], dim=0)
        last_update = scatter(t, self._assoc[idx], 0, _idx.size(0), reduce='max')

        emb = torch.cat([src_emb, pos_dst_emb], dim=0)
        aggr = self.aggr_module(emb, self._assoc[idx], t, _idx.size(0))

        self.last_update[_idx] = last_update
        self.memory[_idx] = aggr.detach()
