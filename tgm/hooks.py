from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Protocol, Set, runtime_checkable

import numpy as np
import torch

from tgm._storage import DGSliceTracker
from tgm.graph import DGBatch, DGraph


@runtime_checkable
class DGHook(Protocol):
    requires: Set[str]
    produces: Set[str]

    r"""The behaviours to be executed on a DGraph before materializing."""

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch: ...


class HookManager:
    def __init__(self, dg: DGraph, hooks: List[DGHook]) -> None:
        if not isinstance(hooks, list):
            raise TypeError(f'Invalid hook type: {type(hooks)}')
        if not all(isinstance(h, DGHook) for h in hooks):
            raise TypeError('All items in hook list must follow DGHook protocol')

        # Implicitly add dedup hook after all user-defined hooks and before device transfer
        hooks.append(DeduplicationHook())

        if dg.device.type != 'cpu':
            hooks.append(PinMemoryHook())
            hooks.append(DeviceTransferHook(dg.device))

        self.hooks = hooks
        self._validate_hook_dependencies()

    @classmethod
    def from_any(
        cls, dg: DGraph, hook_like: HookManager | DGHook | List[DGHook] | None
    ) -> HookManager:
        if isinstance(hook_like, cls):
            return hook_like
        elif hook_like is None:
            return cls(dg, hooks=[])
        elif isinstance(hook_like, DGHook):
            return cls(dg, hooks=[hook_like])
        elif isinstance(hook_like, list):
            return cls(dg, hooks=hook_like)
        else:
            raise TypeError(f'Invalid hook type: {type(hook_like)}')

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
    requires: Set[str] = set()
    produces: Set[str] = set()

    r"""Pin all tensors in the DGBatch to page-locked memory for faster async CPU-GPU transfers."""

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        for k, v in vars(batch).items():
            if isinstance(v, torch.Tensor) and not v.is_cuda and not v.is_pinned():
                setattr(batch, k, v.pin_memory())
        return batch


class DeviceTransferHook:
    requires: Set[str] = set()
    produces: Set[str] = set()

    r"""Moves all tensors in the DGBatch to the specified device."""

    def __init__(self, device: str | torch.device) -> None:
        self.device = torch.device(device)

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        for k, v in vars(batch).items():
            if isinstance(v, torch.Tensor) and v.device != self.device:
                setattr(batch, k, v.to(device=self.device, non_blocking=True))
        return batch


class DeduplicationHook:
    requires: Set[str] = set()
    produces = {
        'unique_nids',
        'nid_to_idx',
        'src_idx',
        'dst_idx',
        'neg_idx',
        'nbr_nids_idx',
    }

    r"""Deduplicate node IDs from batch fields and create index mappings to unique node embeddings.

    Note: Supports batches with or without negative samples and multi-hop neighbors.
    """

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        device = batch.src.device

        nids = [batch.src, batch.dst]
        if hasattr(batch, 'neg'):
            nids.append(batch.neg)
        if hasattr(batch, 'nbr_nids'):
            for hop_nids in batch.nbr_nids:
                nids.append(hop_nids.flatten())

        all_nids = torch.cat(nids, dim=0)
        unique_nids = torch.unique(all_nids)
        max_nid = int(unique_nids.max().item())
        nid_to_idx = torch.full((max_nid + 1,), -1, dtype=torch.long, device=device)
        nid_to_idx[unique_nids] = torch.arange(len(unique_nids), device=device)

        batch.unique_nids = unique_nids  # type: ignore
        batch.nid_to_idx = nid_to_idx  # type: ignore
        batch.src_idx = nid_to_idx[batch.src]  # type: ignore
        batch.dst_idx = nid_to_idx[batch.dst]  # type: ignore
        if hasattr(batch, 'neg'):
            batch.neg_idx = nid_to_idx[batch.neg]  # type: ignore
        if hasattr(batch, 'nbr_nids'):
            batch.nbr_nids_idx = [nid_to_idx[hop_nids] for hop_nids in batch.nbr_nids]  # type: ignore

        return batch


class NegativeEdgeSamplerHook:
    requires: Set[str] = set()
    produces = {'neg'}

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


class TGBNegativeEdgeSamplerHook:
    requires: Set[str] = set()
    produces = {'neg', 'neg_batch_list'}
    r"""Load data from DGraph using pre-generated TGB negative samples.
    Make sure to perform `dataset.load_val_ns()` or `dataset.load_test_ns()` before using this hook.

    Args:
        neg_sampler (object): The negative sampler object to use for sampling.
        split_mode (str): The split mode to use for sampling, either 'val' or 'test'.

    Raises:
        ValueError: If neg_sampler is not provided.
    """

    def __init__(self, neg_sampler: object, split_mode: str) -> None:
        if neg_sampler is None:
            raise ValueError('neg_sampler must be provided')
        if split_mode not in ['val', 'test']:
            raise ValueError('split_mode must be one of val, test')
        if neg_sampler.eval_set[split_mode] is None:  # type: ignore
            raise ValueError(
                f'please run load_{split_mode}_ns() before using this hook'
            )

        self.neg_sampler = neg_sampler
        self.split_mode = split_mode

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        # this might complain if the edge is not found in the negative sampler, which could happen if the user is not using the correct version of dataset
        neg_batch_list = self.neg_sampler.query_batch(  # type: ignore
            batch.src, batch.dst, batch.time, split_mode=self.split_mode
        )
        queries = []
        tensor_batch_list = []
        for neg_batch in neg_batch_list:
            queries.append(neg_batch)
            tensor_batch_list.append(torch.tensor(neg_batch, dtype=torch.long))
        unique_neg = np.unique(np.concatenate(queries))
        batch.neg = torch.tensor(unique_neg, dtype=torch.long)  # type: ignore
        batch.neg_batch_list = tensor_batch_list  # type: ignore
        return batch


class NeighborSamplerHook:
    requires: Set[str] = set()
    produces = {'nids', 'nbr_nids', 'nbr_times', 'nbr_feats', 'nbr_mask'}

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
        device = dg.device
        if hasattr(batch, 'neg'):
            batch.neg = batch.neg.to(device)
            seed_nodes = torch.cat([batch.src, batch.dst, batch.neg])
        else:
            seed_nodes = torch.cat([batch.src, batch.dst])

        # TODO: Storage needs to use the right device
        batch.nids, batch.nbr_nids, batch.nbr_times, batch.nbr_feats, batch.nbr_mask = (  # type: ignore
            dg._storage.get_nbrs(
                seed_nodes=seed_nodes,
                num_nbrs=self.num_nbrs,
                slice=DGSliceTracker(end_idx=dg._slice.end_idx),
            )
        )
        return batch


class RecencyNeighborHook:
    requires: Set[str] = set()
    produces = {'nids', 'nbr_nids', 'nbr_times', 'nbr_feats', 'nbr_mask'}

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
        device = dg.device
        if hasattr(batch, 'neg'):
            # TODO: Because we materialize DGraph on device, this means
            # batch.src, batch.dst are already on the correct device.
            # However, batch.neg is not. And we cannot wait until the device
            # hook runs since the correct device must already be handled here.
            # This questions whether we should materialize on device, since user
            # custom hooks will also need to handle this behaviour. Alternatively,
            # we make the hook manager smart, it 'injects' the device hook in the
            # list of hooks depending on their 'requires' set.
            batch.neg = batch.neg.to(device)
            seed_nodes = torch.cat([batch.src, batch.dst, batch.neg])
        else:
            seed_nodes = torch.cat([batch.src, batch.dst])

        unique, inverse_indices = seed_nodes.unique(return_inverse=True)

        batch_size = len(seed_nodes)
        nbr_nids = torch.empty(
            batch_size, self._num_nbrs[0], dtype=torch.long, device=device
        )
        nbr_times = torch.empty(
            batch_size, self._num_nbrs[0], dtype=torch.long, device=device
        )
        nbr_feats = torch.zeros(
            batch_size,
            self._num_nbrs[0],
            dg.edge_feats_dim,  # type: ignore
            device=device,
        )
        nbr_mask = torch.zeros(
            batch_size,
            self._num_nbrs[0],
            dtype=torch.long,
            device=device,
        )
        for i, node in enumerate(unique.tolist()):
            if nn := len(self._nbrs[node][0]):
                mask = inverse_indices == i
                nbr_nids[mask, :nn] = torch.tensor(
                    self._nbrs[node][0], device=device, dtype=torch.long
                )
                nbr_times[mask, :nn] = torch.tensor(
                    self._nbrs[node][1], device=device, dtype=torch.long
                )
                nbr_feats[mask, :nn] = (
                    torch.stack(list(self._nbrs[node][2])).float().to(device)
                )
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
