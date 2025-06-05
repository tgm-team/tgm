from __future__ import annotations

from collections import deque
from dataclasses import fields
from typing import Any, Deque, Dict, List, Protocol, Set

import torch

from opendg._storage import DGSliceTracker
from opendg.graph import DGBatch, DGraph


class DGHook(Protocol):
    requires: Set[str]
    produces: Set[str]

    r"""The behaviours to be executed on a DGraph before materializing."""

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch: ...


class HookManager:
    def __init__(self, hooks: List[DGHook], device: str = 'cpu') -> None:
        if device != 'cpu':
            hooks.append(PinMemoryHook())
            hooks.append(DeviceTransferHook(device))

        self.hooks = hooks
        self._validate_hook_dependencies()

    def __call__(self, dg: DGraph) -> DGBatch:
        batch = dg.materialize()
        for hook in self.hooks:
            batch = hook(dg, batch)
        return batch

    def _validate_hook_dependencies(self) -> None:
        produced: Set[str] = set()
        for hook in self.hooks:
            missing = hook.requires - produced
            if missing:
                raise ValueError(
                    f'{hook.__class__.__name__} is missing required fields: {missing}'
                )
            produced |= hook.produces


class PinMemoryHook:
    requires = set()
    produces = set()

    r"""Pin all tensors in the DGBatch to page-locked memory for faster async CPU-GPU transfers."""

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        for k in fields(batch):
            v = getattr(batch, k.name)
            if isinstance(v, torch.Tensor) and not v.is_cuda and not v.is_pinned():
                setattr(batch, k.name, v.pin_memory())
        return batch


class DeviceTransferHook:
    requires = set()
    produces = set()

    r"""Moves all tensors in the DGBatch to the specified device."""

    def __init__(self, device: str) -> None:
        self.device = device

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        for k in fields(batch):
            v = getattr(batch, k.name)
            if isinstance(v, torch.Tensor) and v.device != self.device:
                setattr(batch, k.name, v.to(device=self.device, non_blocking=True))
        return batch


class NegativeEdgeSamplerHook:
    r"""Sample negative edges for dynamic link prediction.

    Args:
        low (int): The minimum node id to sample
        high (int) : The maximum node id to sample
        neg_sampling_ratio (float): The ratio of sampled negative destination nodes
            to the number of positive destination nodes (default = 1.0).
    """

    def __init__(self, low: int, high: int, neg_sampling_ratio: float = 1.0) -> None:
        if not 0 < neg_sampling_ratio <= 1:
            raise ValueError('neg_sampling_ratio must be in (0, 1]')
        if not low < high:
            raise ValueError(f'low ({low}) must be strictly less than high ({high})')
        self.low = low
        self.high = high
        self.neg_sampling_ratio = neg_sampling_ratio

    # TODO: Historical vs. random
    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        size = (round(self.neg_sampling_ratio * batch.dst.size(0)),)
        batch.neg = torch.randint(self.low, self.high, size)  # type: ignore
        return batch


class NeighborSamplerHook:
    r"""Load data from DGraph using a memory based sampling function.

    Args:
        num_nbrs (List[int]): Number of neighbors to sample at each hop (-1 to keep all)

    Raises:
        ValueError: If the num_nbrs list is empty.
    """

    def __init__(self, num_nbrs: List[int]) -> None:
        if not len(num_nbrs):
            raise ValueError('num_nbrs must be non-empty')
        if not all([isinstance(x, int) and (x == -1 or x > 0) for x in num_nbrs]):
            raise ValueError('Each value in num_nbrs must be a positive integer or -1')
        self._num_nbrs = num_nbrs

    @property
    def num_nbrs(self) -> List[int]:
        return self._num_nbrs

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        # TODO: Compose hooks
        self.neg_sampling_ratio = 1.0
        self.low = 0
        self.high = dg.num_nodes
        size = (round(self.neg_sampling_ratio * batch.dst.size(0)),)
        batch.neg = torch.randint(self.low, self.high, size, dtype=torch.long)  # type: ignore

        batch.nids, batch.nbr_nids, batch.nbr_times, batch.nbr_feats, batch.nbr_mask = (  # type: ignore
            dg._storage.get_nbrs(
                seed_nodes=torch.cat([batch.src, batch.dst, batch.neg]),  # type: ignore
                num_nbrs=self.num_nbrs,
                slice=DGSliceTracker(end_idx=dg._slice.end_idx),
            )
        )
        return batch


class RecencyNeighborHook:
    r"""Load neighbors from DGraph using a recency sampling. Each node maintains a fixed number of recent neighbors.

    Args:
        num_nodes (int): Total number of nodes to track.
        num_nbrs (List[int]): Number of neighbors to sample at each hop (max neighbors to keep).

    Raises:
        ValueError: If the num_nbrs list is empty.
    """

    def __init__(self, num_nodes: int, num_nbrs: List[int]) -> None:
        if not len(num_nbrs):
            raise ValueError('num_nbrs must be non-empty')
        if not all([isinstance(x, int) and (x > 0) for x in num_nbrs]):
            raise ValueError('Each value in num_nbrs must be a positive integer')
        if len(num_nbrs) > 1:
            raise ValueError('RecencyNeighborSamplerHook only supports 1 hop for now')
        self._num_nbrs = num_nbrs
        self._nbrs: Dict[int, List[Deque[Any]]] = {}
        for node in range(num_nodes):
            self._nbrs[node] = [deque(maxlen=num_nbrs[0]) for _ in range(3)]

    @property
    def num_nbrs(self) -> List[int]:
        return self._num_nbrs

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        # TODO: Compose hooks
        self.neg_sampling_ratio = 1.0
        self.low = 0
        self.high = dg.num_nodes
        size = (round(self.neg_sampling_ratio * batch.dst.size(0)),)
        batch.neg = torch.randint(self.low, self.high, size, dtype=torch.long)  # type: ignore

        seed_nodes = torch.cat([batch.src, batch.dst, batch.neg])  # type: ignore
        unique, inverse_indices = seed_nodes.unique(return_inverse=True)

        batch_size = len(seed_nodes)
        nbr_nids = torch.empty(batch_size, self._num_nbrs[0], dtype=torch.long)
        nbr_times = torch.empty(batch_size, self._num_nbrs[0], dtype=torch.long)
        nbr_feats = torch.zeros(batch_size, self._num_nbrs[0], dg.edge_feats_dim)  # type: ignore
        nbr_mask = torch.zeros(batch_size, self._num_nbrs[0], dtype=torch.long)
        for i, node in enumerate(unique.tolist()):
            if nn := len(self._nbrs[node][0]):
                mask = inverse_indices == i
                nbr_nids[mask, :nn] = torch.LongTensor(self._nbrs[node][0])
                nbr_times[mask, :nn] = torch.LongTensor(self._nbrs[node][1])
                nbr_feats[mask, :nn] = torch.stack(list(self._nbrs[node][2]))  # TODO:
                nbr_mask[mask, :nn] = nn >= self._num_nbrs[0]

        batch.nids = [seed_nodes]  # type: ignore
        batch.nbr_nids = [nbr_nids]  # type: ignore
        batch.nbr_times = [nbr_times]  # type: ignore
        batch.nbr_feats = [nbr_feats]  # type: ignore
        batch.nbr_mask = [nbr_mask]  # type: ignore

        self._update(batch)
        return batch

    def _update(self, batch: DGBatch) -> None:
        for i in range(batch.src.size(0)):
            src_nbr = int(batch.src[i].item())
            dst_nbr = int(batch.dst[i].item())
            time = batch.time[i].item()

            self._nbrs[src_nbr][0].append(dst_nbr)
            self._nbrs[src_nbr][1].append(time)
            self._nbrs[dst_nbr][0].append(src_nbr)
            self._nbrs[dst_nbr][1].append(time)
            if batch.edge_feats is not None:
                self._nbrs[src_nbr][2].append(batch.edge_feats[i])
                self._nbrs[dst_nbr][2].append(batch.edge_feats[i])
