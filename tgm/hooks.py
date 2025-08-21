from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import is_dataclass
from typing import Any, List, Protocol, Set, runtime_checkable

import numpy as np
import torch

from tgm import DGBatch, DGraph
from tgm._storage import DGSliceTracker


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
        bad_hook_names = [type(h).__name__ for h in hooks if not isinstance(h, DGHook)]
        if len(bad_hook_names):
            raise TypeError(
                f'These hooks do not correctly implement the DGHook protocol: {bad_hook_names}, '
                'ensure there is a __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch implemented'
            )

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
        pin_if_needed = (
            lambda x: x.pin_memory() if not x.is_cuda and not x.is_pinned() else x
        )

        _apply_to_tensors_inplace(batch, pin_if_needed)
        return batch


class DeviceTransferHook:
    requires: Set[str] = set()
    produces: Set[str] = set()

    r"""Moves all tensors in the DGBatch to the specified device."""

    def __init__(self, device: str | torch.device) -> None:
        self.device = torch.device(device)

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        move_if_needed = (
            lambda x: x.to(device=self.device, non_blocking=True)
            if x.device != self.device
            else x
        )

        _apply_to_tensors_inplace(batch, move_if_needed)
        return batch


class DeduplicationHook:
    requires: Set[str] = set()
    produces = {'unique_nids', 'global_to_local'}

    r"""Deduplicate node IDs from batch fields and create index mappings to unique node embeddings.

    Note: Supports batches with or without negative samples and multi-hop neighbors.
    """

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        nids = [batch.src, batch.dst]
        if hasattr(batch, 'neg'):
            batch.neg = batch.neg.to(batch.src.device)
            nids.append(batch.neg)
        if hasattr(batch, 'nbr_nids'):
            for hop in range(len(batch.nbr_nids)):
                hop_nids, hop_mask = batch.nbr_nids[hop], batch.nbr_mask[hop].bool()  # type: ignore
                valid_hop_nids = hop_nids[hop_mask].to(batch.src.device)
                nids.append(valid_hop_nids)

        all_nids = torch.cat(nids, dim=0)
        unique_nids = torch.unique(all_nids, sorted=True)

        batch.unique_nids = unique_nids  # type: ignore
        batch.global_to_local = lambda x: torch.searchsorted(unique_nids, x)  # type: ignore

        return batch


class NegativeEdgeSamplerHook:
    requires: Set[str] = set()
    produces = {'neg'}

    r"""Sample negative edges for dynamic link prediction.

    Args:
        low (int): The minimum node id to sample
        high (int) : The maximum node id to sample
        neg_ratio (float): The ratio of sampled negative destination nodes
            to the number of positive destination nodes (default = 1.0).
    """

    def __init__(self, low: int, high: int, neg_ratio: float = 1.0) -> None:
        if not 0 < neg_ratio <= 1:
            raise ValueError(f'neg_ratio must be in (0, 1], got: {neg_ratio}')
        if not low < high:
            raise ValueError(f'low ({low}) must be strictly less than high ({high})')
        self.low = low
        self.high = high
        self.neg_ratio = neg_ratio

    # TODO: Historical vs. random
    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        size = (round(self.neg_ratio * batch.dst.size(0)),)
        batch.neg = torch.randint(  # type: ignore
            self.low, self.high, size, dtype=torch.long, device=dg.device
        )
        batch.neg = 888 * torch.ones_like(batch.src).to(dg.device)
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
            raise ValueError(f'split_mode must be "val" or "test", got: {split_mode}')
        if neg_sampler.eval_set[split_mode] is None:  # type: ignore
            raise ValueError(
                f'please run load_{split_mode}_ns() before using this hook'
            )

        self.neg_sampler = neg_sampler
        self.split_mode = split_mode

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        # this might complain if the edge is not found in the negative sampler, which could happen if the user is not using the correct version of dataset
        neg_batch_list = self.neg_sampler.query_batch(  # type: ignore
            np.array([batch.src[0]]) - 1,
            np.array([batch.dst[0]]) - 1,
            np.array([batch.time[0]]),
            split_mode=self.split_mode,
        )
        queries = []
        tensor_batch_list = []
        for neg_batch in neg_batch_list:
            neg_batch = [x + 1 for x in neg_batch]
            queries.append(neg_batch)
            tensor_batch_list.append(
                torch.tensor(neg_batch, dtype=torch.long, device=dg.device)
            )
        unique_neg = np.unique(np.concatenate(queries))
        batch.neg = torch.tensor(unique_neg, dtype=torch.long, device=dg.device)  # type: ignore
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

        batch.nids, batch.times = [], []  # type: ignore
        batch.nbr_nids, batch.nbr_times = [], []  # type: ignore
        batch.nbr_feats, batch.nbr_mask = [], []  # type: ignore

        for hop, num_nbrs in enumerate(self.num_nbrs):
            if hop == 0:
                seed = [batch.src, batch.dst]
                times = [batch.time.repeat(2)]  # Real link times
                if hasattr(batch, 'neg'):
                    batch.neg = batch.neg.to(device)
                    seed.append(batch.neg)

                    # This is a heuristic. For our fake (negative) link times,
                    # we pick random time stamps within [batch.start_time, batch.end_time].
                    # Using random times on the whole graph will likely produce information
                    # leakage, making the prediction easier than it should be.

                    # Use generator to locall constrain rng for reproducability
                    gen = torch.Generator(device=device)
                    gen.manual_seed(0)
                    fake_times = torch.randint(
                        int(batch.time.min().item()),
                        int(batch.time.max().item()) + 1,
                        (batch.neg.size(0),),
                        device=device,
                        generator=gen,
                    )
                    times.append(fake_times)
                seed_nodes = torch.cat(seed)
                seed_times = torch.cat(times)
            else:
                mask = batch.nbr_mask[hop - 1].bool()  # type: ignore
                seed_nodes = batch.nbr_nids[hop - 1][mask].flatten()  # type: ignore
                seed_times = batch.nbr_times[hop - 1][mask].flatten()  # type: ignore

            # TODO: Storage needs to use the right device

            # We slice on batch.start_time so that we only consider neighbor events
            # that occurred strictly before this batch
            nbr_nids, nbr_times, nbr_feats, nbr_mask = dg._storage.get_nbrs(
                seed_nodes,
                num_nbrs=num_nbrs,
                slice=DGSliceTracker(end_time=int(batch.time.min())),
            )

            batch.nids.append(seed_nodes)  # type: ignore
            batch.times.append(seed_times)  # type: ignore
            batch.nbr_nids.append(nbr_nids)  # type: ignore
            batch.nbr_times.append(nbr_times)  # type: ignore
            batch.nbr_feats.append(nbr_feats)  # type: ignore
            batch.nbr_mask.append(nbr_mask)  # type: ignore

        return batch


class RecencyNeighborHook:
    requires: Set[str] = set()
    produces = {'nids', 'nbr_nids', 'times', 'nbr_times', 'nbr_feats', 'nbr_mask'}

    r"""Load neighbors from DGraph using a recency sampling. Each node maintains a fixed number of recent neighbors.

    Args:
        num_nodes (int): Total number of nodes to track.
        num_nbrs (List[int]): Number of neighbors to sample at each hop (max neighbors to keep).
        edge_feats_dim (int): Edge feature dimension on the dynamic graph.

    Raises:
        ValueError: If the num_nbrs list is empty.
    """

    def __init__(
        self, num_nodes: int, num_nbrs: List[int], edge_feats_dim: int
    ) -> None:
        if not len(num_nbrs):
            raise ValueError('num_nbrs must be non-empty')
        if not all([isinstance(x, int) and (x > 0) for x in num_nbrs]):
            raise ValueError('Each value in num_nbrs must be a positive integer')

        self._num_nbrs = num_nbrs
        self._max_nbrs = max(num_nbrs)
        self._edge_feats_dim = edge_feats_dim

        # For each node: list of (nbr_id, time, feat)
        self._history = defaultdict(lambda: deque())

        self._device = torch.device('cpu')

    @property
    def num_nbrs(self) -> List[int]:
        return self._num_nbrs

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        device = dg.device
        self._move_queues_to_device_if_needed(device)  # No-op after first batch
        self._update(batch)

        batch.nids, batch.times = [], []  # type: ignore
        batch.nbr_nids, batch.nbr_times = [], []  # type: ignore
        batch.nbr_feats, batch.nbr_mask = [], []  # type: ignore

        for hop, num_nbrs in enumerate(self.num_nbrs):
            if hop == 0:
                seed_nodes = torch.cat([batch.src, batch.dst])
                seed_times = batch.time.repeat(2)

                if hasattr(batch, 'neg'):
                    batch.neg = batch.neg.to(device)
                    seed_nodes = torch.cat([seed_nodes, batch.neg])

                    fake_times = batch.time.repeat(len(batch.neg))
                    seed_times = torch.cat([seed_times, fake_times])
            else:
                batch.nbr_mask[hop - 1].bool()
                # TODO:: figure out mask
                # seed_nodes = batch.nbr_nids[hop - 1][mask].flatten()
                # seed_times = batch.nbr_times[hop - 1][mask].flatten()
                seed_nodes = batch.nbr_nids[hop - 1].flatten()
                seed_times = batch.nbr_times[hop - 1].flatten()

            nbr_nids, nbr_times, nbr_feats, nbr_mask = self._get_recency_neighbors(
                seed_nodes, seed_times, num_nbrs
            )

            batch.nids.append(seed_nodes)  # type: ignore
            batch.times.append(seed_times)  # type: ignore
            batch.nbr_nids.append(nbr_nids)  # type: ignore
            batch.nbr_times.append(nbr_times)  # type: ignore
            batch.nbr_feats.append(nbr_feats)  # type: ignore
            batch.nbr_mask.append(nbr_mask)  # type: ignore

        return batch

    def _get_recency_neighbors(
        self, node_ids: torch.Tensor, query_times: torch.Tensor, k: int
    ):
        num_nodes = node_ids.size(0)
        device = node_ids.device

        nbr_nids = torch.zeros((num_nodes, k), dtype=torch.long, device=device)
        nbr_times = torch.zeros((num_nodes, k), dtype=torch.long, device=device)
        nbr_feats = torch.zeros((num_nodes, k, self._edge_feats_dim), device=device)
        nbr_mask = torch.zeros((num_nodes, k), dtype=torch.bool, device=device)

        for i, (nid, qtime) in enumerate(zip(node_ids.tolist(), query_times.tolist())):
            history = self._history[nid]
            # Filter neighbors strictly before the query time
            valid = [(nbr, t, f) for (nbr, t, f) in history if t < qtime]
            if not valid:
                continue
            # Take the most recent k
            valid = valid[-k:]

            # Fill in from the back
            nbr_nids[i, -len(valid) :] = torch.tensor(
                [x[0] for x in valid], dtype=torch.long, device=device
            )
            nbr_times[i, -len(valid) :] = torch.tensor(
                [x[1] for x in valid], dtype=torch.long, device=device
            )
            nbr_feats[i, -len(valid) :] = torch.stack([x[2] for x in valid])
            nbr_mask[i, -len(valid) :] = True

        return nbr_nids, nbr_times, nbr_feats, nbr_mask

    def _update(self, batch):
        src, dst, time = batch.src.tolist(), batch.dst.tolist(), batch.time.tolist()
        if batch.edge_feats is None:
            edge_feats = torch.zeros(
                (len(src), self._edge_feats_dim), device=self._device
            )
        else:
            edge_feats = batch.edge_feats

        for s, d, t, f in zip(src, dst, time, edge_feats):
            self._history[s].append((d, t, f.clone()))
            self._history[d].append((s, t, f.clone()))  # undirected

    def _move_queues_to_device_if_needed(self, device: torch.device) -> None:
        self._device = device


def _apply_to_tensors_inplace(obj: Any, fn: Any) -> Any:
    if torch.is_tensor(obj):
        return fn(obj)
    elif is_dataclass(obj):
        for k, v in vars(obj).items():
            setattr(obj, k, _apply_to_tensors_inplace(v, fn))
        return obj
    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = _apply_to_tensors_inplace(obj[i], fn)
        return obj
    elif isinstance(obj, tuple):
        # Tuples are immutable, so return a new tuple
        return tuple(_apply_to_tensors_inplace(x, fn) for x in obj)
    elif isinstance(obj, dict):
        for k in obj:
            obj[k] = _apply_to_tensors_inplace(obj[k], fn)
        return obj
    else:
        return obj
