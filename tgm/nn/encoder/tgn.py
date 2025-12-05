import copy
from typing import Callable, Dict, Tuple, cast

import torch
from torch import Tensor
from torch.nn import GRUCell
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.inits import zeros
from torch_geometric.utils import scatter

from ..modules.time_encoding import Time2Vec


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        msg_dim: int,
        time_enc: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.time_dim
        self.conv = TransformerConv(
            in_channels, out_channels // 2, heads=2, dropout=0.1, edge_dim=edge_dim
        )

    def forward(
        self,
        x: torch.Tensor,
        last_update: torch.Tensor,
        edge_index: torch.Tensor,
        t: torch.Tensor,
        msg: torch.Tensor,
    ) -> torch.Tensor:
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LastAggregator(torch.nn.Module):
    def forward(
        self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int
    ) -> torch.Tensor:
        out = msg.new_zeros((dim_size, msg.size(-1)))

        if index.numel() > 0:
            scores = torch.full((dim_size, t.size(0)), float('-inf'), device=t.device)
            scores[index, torch.arange(t.size(0), device=t.device)] = t.float()
            argmax = scores.argmax(dim=1)
            valid = scores.max(dim=1).values > float('-inf')
            out[valid] = msg[argmax[valid]]

        return out


class MeanAggregator(torch.nn.Module):
    def forward(
        self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int
    ) -> torch.Tensor:
        return scatter(msg, index, dim=0, dim_size=dim_size, reduce='mean')


class IdentityMessage(torch.nn.Module):
    def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int) -> None:
        super().__init__()
        self.out_channels = raw_msg_dim + 2 * memory_dim + time_dim

    def forward(
        self, z_src: Tensor, z_dst: Tensor, raw_msg: Tensor, t_enc: Tensor
    ) -> torch.Tensor:
        return torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1)


TGNMessageStoreType = Dict[int, Tuple[Tensor, Tensor, Tensor, Tensor]]


class TGNMemory(torch.nn.Module):
    r"""The Temporal Graph Network (TGN) memory model from the
    `"Temporal Graph Networks for Deep Learning on Dynamic Graphs"
    <https://arxiv.org/abs/2006.10637>`_ paper.

    .. note::

        For an example of using TGN, see `examples/tgn.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        tgn.py>`_.

    Args:
        num_nodes (int): The number of nodes to save memories for.
        raw_msg_dim (int): The raw message dimensionality.
        memory_dim (int): The hidden memory dimensionality.
        time_dim (int): The time encoding dimensionality.
        message_module (torch.nn.Module): The message function which
            combines source and destination node memory embeddings, the raw
            message and the time encoding.
        aggregator_module (torch.nn.Module): The message aggregator function
            which aggregates messages to the same destination into a single
            representation.
    """

    def __init__(
        self,
        num_nodes: int,
        raw_msg_dim: int,
        memory_dim: int,
        time_dim: int,
        message_module: Callable,
        aggregator_module: Callable,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim

        self.msg_s_module = message_module
        self.msg_d_module = copy.deepcopy(message_module)
        self.aggr_module = aggregator_module
        self.time_enc = Time2Vec(time_dim=time_dim)
        self.memory_updater = GRUCell(message_module.out_channels, memory_dim)  # type: ignore

        self.register_buffer('memory', torch.empty(num_nodes, memory_dim))
        self.register_buffer('last_update', torch.empty(num_nodes, dtype=torch.long))
        self.register_buffer('_assoc', torch.empty(num_nodes, dtype=torch.long))

        self.msg_s_store: TGNMessageStoreType = {}
        self.msg_d_store: TGNMessageStoreType = {}

        self.reset_parameters()

    @property
    def device(self) -> torch.device:
        return self.time_enc.w.weight.device

    def reset_parameters(self) -> None:
        if hasattr(self.msg_s_module, 'reset_parameters'):
            self.msg_s_module.reset_parameters()
        if hasattr(self.msg_d_module, 'reset_parameters'):
            self.msg_d_module.reset_parameters()
        if hasattr(self.aggr_module, 'reset_parameters'):
            self.aggr_module.reset_parameters()
        self.memory_updater.reset_parameters()
        self.reset_state()

    def reset_state(self) -> None:
        zeros(self.memory)
        zeros(self.last_update)
        self._reset_message_store()

    def detach(self) -> None:
        self.memory.detach_()

    def forward(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        if self.training:
            memory, last_update = self._get_updated_memory(n_id)
        else:
            memory, last_update = self.memory[n_id], self.last_update[n_id]

        return memory, last_update

    def update_state(
        self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor
    ) -> None:
        n_id = torch.cat([src, dst]).unique()

        if self.training:
            self._update_memory(n_id)
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
        else:
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
            self._update_memory(n_id)

    def _reset_message_store(self) -> None:
        i = self.memory.new_empty((0,), device=self.device, dtype=torch.long)
        msg = self.memory.new_empty((0, self.raw_msg_dim), device=self.device)
        # Message store format: (src, dst, t, msg)
        self.msg_s_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}
        self.msg_d_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}

    def _update_memory(self, n_id: Tensor) -> None:
        memory, last_update = self._get_updated_memory(n_id)
        self.memory[n_id] = memory
        self.last_update[n_id] = last_update

    def _get_updated_memory(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        # Compute messages (src -> dst).
        msg_s, t_s, src_s, _ = self._compute_msg(
            n_id, self.msg_s_store, self.msg_s_module
        )

        # Compute messages (dst -> src).
        msg_d, t_d, src_d, _ = self._compute_msg(
            n_id, self.msg_d_store, self.msg_d_module
        )

        # Aggregate messages.
        idx = torch.cat([src_s, src_d], dim=0)
        msg = torch.cat([msg_s, msg_d], dim=0)
        t = torch.cat([t_s, t_d], dim=0)
        aggr = self.aggr_module(msg, self._assoc[idx], t, n_id.size(0))

        # Get local copy of updated memory.
        memory = self.memory_updater(aggr, self.memory[n_id])

        # Get local copy of updated `last_update`.
        dim_size = self.last_update.size(0)
        last_update = scatter(t, idx.long(), 0, dim_size, reduce='max')[n_id]
        return memory, last_update

    def _update_msg_store(
        self,
        src: Tensor,
        dst: Tensor,
        t: Tensor,
        raw_msg: Tensor,
        msg_store: TGNMessageStoreType,
    ) -> None:
        n_id, perm = src.sort()
        n_id, count = n_id.unique_consecutive(return_counts=True)
        for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
            msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx])

    def _compute_msg(
        self, n_id: Tensor, msg_store: TGNMessageStoreType, msg_module: Callable
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        data = [msg_store[i] for i in n_id.tolist()]
        src, dst, t, raw_msg = list(zip(*data))
        src = torch.cat(src, dim=0).to(self.device)  # type: ignore
        dst = torch.cat(dst, dim=0).to(self.device)  # type: ignore
        t = torch.cat(t, dim=0).to(self.device)  # type: ignore
        raw_msg = torch.cat(raw_msg, dim=0).to(self.device)  # type: ignore
        t_rel = t - self.last_update[src]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))  # type: ignore
        msg = msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc)
        return cast(Tuple[Tensor, Tensor, Tensor, Tensor], (msg, t, src, dst))

    def train(self, mode: bool = True) -> 'TGNMemory':
        if self.training and not mode:
            # Flush message store to memory in case we just entered eval mode.
            self._update_memory(torch.arange(self.num_nodes, device=self.memory.device))
            self._reset_message_store()
        super().train(mode)
        return self
