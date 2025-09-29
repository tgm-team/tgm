from __future__ import annotations

from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, List, Set, Tuple

import torch

from tgm import DGBatch, DGraph
from tgm._storage import DGSliceTracker
from tgm.constants import PADDED_NODE_ID
from tgm.hooks import StatefulHook, StatelessHook


class PinMemoryHook(StatelessHook):
    """Pin all tensors in the DGBatch to page-locked memory for faster async CPU-GPU transfers."""

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        pin_if_needed = (
            lambda x: x.pin_memory() if not x.is_cuda and not x.is_pinned() else x
        )

        _apply_to_tensors_inplace(batch, pin_if_needed)
        return batch


class DeviceTransferHook(StatelessHook):
    """Moves all tensors in the DGBatch to the specified device."""

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


class DeduplicationHook(StatelessHook):
    """Deduplicate node IDs from batch fields and create index mappings to unique node embeddings.

    Note: Supports batches with or without negative samples and multi-hop neighbors.
    """

    requires: Set[str] = set()
    produces = {'unique_nids', 'global_to_local'}

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        nids = [batch.src, batch.dst]
        if hasattr(batch, 'neg'):
            batch.neg = batch.neg.to(batch.src.device)
            nids.append(batch.neg)
        if hasattr(batch, 'nbr_nids'):
            for hop in range(len(batch.nbr_nids)):
                nids.append(batch.nbr_nids[hop].to(batch.src.device))
        nids.append(
            batch.node_ids.to(batch.src.device)
        ) if batch.node_ids is not None else None

        all_nids = torch.cat(nids, dim=0)
        unique_nids = torch.unique(all_nids, sorted=True)

        batch.unique_nids = unique_nids  # type: ignore
        batch.global_to_local = lambda x: torch.searchsorted(unique_nids, x).int()  # type: ignore

        return batch


class NegativeEdgeSamplerHook(StatelessHook):
    """Sample negative edges for dynamic link prediction.

    Args:
        low (int): The minimum node id to sample
        high (int) : The maximum node id to sample
        neg_ratio (float): The ratio of sampled negative destination nodes
            to the number of positive destination nodes (default = 1.0).
    """

    requires: Set[str] = set()
    produces = {'neg', 'neg_time'}

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
        if size[0] == 0:
            batch.neg = torch.empty(size, dtype=torch.int32, device=dg.device)  # type: ignore
            batch.neg_time = torch.empty(size, dtype=torch.int64, device=dg.device)  # type: ignore
        else:
            batch.neg = torch.randint(  # type: ignore
                self.low, self.high, size, dtype=torch.int32, device=dg.device
            )
            batch.neg_time = batch.time.clone()  # type: ignore
        return batch


class TGBNegativeEdgeSamplerHook(StatelessHook):
    """Load data from DGraph using pre-generated TGB negative samples.
    Make sure to perform `dataset.load_val_ns()` or `dataset.load_test_ns()` before using this hook.

    Args:
        neg_sampler (object): The negative sampler object to use for sampling.
        split_mode (str): The split mode to use for sampling, either 'val' or 'test'.

    Raises:
        ValueError: If neg_sampler is not provided.
    """

    requires: Set[str] = set()
    produces = {'neg', 'neg_batch_list', 'neg_time'}

    def __init__(self, dataset_name: str, split_mode: str) -> None:
        if split_mode not in ['val', 'test']:
            raise ValueError(f'split_mode must be "val" or "test", got: {split_mode}')

        try:
            from tgb.linkproppred.negative_sampler import NegativeEdgeSampler
            from tgb.utils.info import DATA_VERSION_DICT, PROJ_DIR
        except ImportError:
            raise ImportError(
                f'TGB required for {self.__class__.__name__}, try `pip install py-tgb`'
            )

        neg_sampler = NegativeEdgeSampler(dataset_name=dataset_name)

        # Load evaluation sets
        root = Path(PROJ_DIR + 'datasets') / dataset_name.replace('-', '_')
        if dataset_name in DATA_VERSION_DICT:
            version_suffix = f'_v{DATA_VERSION_DICT[dataset_name]}'
        else:
            version_suffix = ''

        val_ns_fname = root / f'{dataset_name}_val_ns{version_suffix}.pkl'
        test_ns_fname = root / f'{dataset_name}_test_ns{version_suffix}.pkl'
        neg_sampler.load_eval_set(fname=str(val_ns_fname), split_mode='val')
        neg_sampler.load_eval_set(fname=str(test_ns_fname), split_mode='test')

        self.neg_sampler = neg_sampler
        self.split_mode = split_mode

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        if batch.src.size(0) == 0:
            batch.neg = torch.empty(  # type: ignore
                batch.src.size(0), dtype=torch.int32, device=dg.device
            )
            batch.neg_time = torch.empty(  # type: ignore
                batch.src.size(0), dtype=torch.int64, device=dg.device
            )
            batch.neg_batch_list = []  # type: ignore
            return batch  # empty batch
        try:
            neg_batch_list = self.neg_sampler.query_batch(
                batch.src, batch.dst, batch.time, split_mode=self.split_mode
            )
        except ValueError as e:
            raise ValueError(
                f'Negative sampling failed for split_mode={self.split_mode}. Try updating your TGB package: `pip install --upgrade py-tgb`'
            ) from e

        batch.neg_batch_list = [  # type: ignore
            torch.tensor(neg_batch, dtype=torch.int32, device=dg.device)
            for neg_batch in neg_batch_list
        ]
        batch.neg = torch.unique(torch.cat(batch.neg_batch_list))  # type: ignore

        # This is a heuristic. For our fake (negative) link times,
        # we pick random time stamps within [batch.start_time, batch.end_time].
        # Using random times on the whole graph will likely produce information
        # leakage, making the prediction easier than it should be.

        # Use generator to local constrain rng for reproducibility
        gen = torch.Generator(device=dg.device)
        gen.manual_seed(0)
        batch.neg_time = torch.randint(  # type: ignore
            int(batch.time.min().item()),
            int(batch.time.max().item()) + 1,
            (batch.neg.size(0),),  # type: ignore
            device=dg.device,
            generator=gen,
        )
        return batch


class NeighborSamplerHook(StatelessHook):
    """Load data from DGraph using a memory based sampling function.

    Args:
        num_nbrs (List[int]): Number of neighbors to sample at each hop (-1 to keep all)
        directed (bool): If true, aggregates interactions in src->dst direction only (default=False).

    Raises:
        ValueError: If the num_nbrs list is empty.
    """

    requires: Set[str] = set()
    produces = {'nids', 'nbr_nids', 'nbr_times', 'nbr_feats'}

    def __init__(self, num_nbrs: List[int], directed: bool = False) -> None:
        if not len(num_nbrs):
            raise ValueError('num_nbrs must be non-empty')
        if not all([isinstance(x, int) and (x > 0) for x in num_nbrs]):
            raise ValueError('Each value in num_nbrs must be a positive integer')
        self._num_nbrs = num_nbrs
        self._directed = directed

    @property
    def num_nbrs(self) -> List[int]:
        return self._num_nbrs

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        device = dg.device

        batch.nids, batch.times = [], []  # type: ignore
        batch.nbr_nids, batch.nbr_times = [], []  # type: ignore
        batch.nbr_feats = []  # type: ignore

        for hop, num_nbrs in enumerate(self.num_nbrs):
            if hop == 0:
                seed = [batch.src, batch.dst]
                times = [batch.time.repeat(2)]  # Real link times
                if hasattr(batch, 'neg'):
                    batch.neg = batch.neg.to(device)
                    seed.append(batch.neg)
                    times.append(batch.neg_time)  # type: ignore

                seed_nodes = torch.cat(seed)
                seed_times = torch.cat(times)
                if seed_nodes.numel() == 0:
                    for hop in range(len(self.num_nbrs)):
                        batch.nids.append(torch.empty(0, dtype=torch.int32))  # type: ignore
                        batch.times.append(torch.empty(0, dtype=torch.int64))  # type: ignore
                        batch.nbr_nids.append(torch.empty(0, dtype=torch.int32))  # type: ignore
                        batch.nbr_times.append(torch.empty(0, dtype=torch.int64))  # type: ignore
                        batch.nbr_feats.append(  # type: ignore
                            torch.empty(0, dg.edge_feats_dim).float()  # type: ignore
                        )
                    return batch
            else:
                seed_nodes = batch.nbr_nids[hop - 1].flatten()  # type: ignore
                seed_times = batch.nbr_times[hop - 1].flatten()  # type: ignore

            # TODO: Storage needs to use the right device

            # We slice on batch.start_time so that we only consider neighbor events
            # that occurred strictly before this batch
            nbr_nids, nbr_times, nbr_feats = dg._storage.get_nbrs(
                seed_nodes,
                num_nbrs=num_nbrs,
                slice=DGSliceTracker(end_time=int(batch.time.min()) - 1),
                directed=self._directed,
            )

            batch.nids.append(seed_nodes)  # type: ignore
            batch.times.append(seed_times)  # type: ignore
            batch.nbr_nids.append(nbr_nids)  # type: ignore
            batch.nbr_times.append(nbr_times)  # type: ignore
            batch.nbr_feats.append(nbr_feats)  # type: ignore

        return batch


class RecencyNeighborHook(StatefulHook):
    requires: Set[str] = set()
    produces = {'nids', 'nbr_nids', 'times', 'nbr_times', 'nbr_feats'}

    """Load neighbors from DGraph using a recency sampling. Each node maintains a fixed number of recent neighbors.

    Args:
        num_nodes (int): Total number of nodes to track.
        num_nbrs (List[int]): Number of neighbors to sample at each hop (max neighbors to keep).
        directed (bool): If true, aggregates interactions in src->dst direction only (default=False).
        seed_nodes_key (str, optional): the str to identify the initial seed nodes to sample for.

    Raises:
        ValueError: If the num_nbrs list is empty.
    """

    def __init__(
        self,
        num_nodes: int,
        num_nbrs: List[int],
        directed: bool = False,
        seed_nodes_key: str = None,  # type: ignore
    ) -> None:
        if not len(num_nbrs):
            raise ValueError('num_nbrs must be non-empty')
        if not all([isinstance(x, int) and (x > 0) for x in num_nbrs]):
            raise ValueError('Each value in num_nbrs must be a positive integer')

        self._num_nodes = num_nodes
        self._num_nbrs = num_nbrs
        self._max_nbrs = max(num_nbrs)
        self._directed = directed
        self._device = torch.device('cpu')
        self._seed_nodes_key = seed_nodes_key

        self._nbr_ids = torch.full(
            (num_nodes, self._max_nbrs), PADDED_NODE_ID, dtype=torch.int32
        )
        self._nbr_times = torch.zeros((num_nodes, self._max_nbrs), dtype=torch.int64)
        self._write_pos = torch.zeros(num_nodes, dtype=torch.int32)

        # Wait until first __call__ to infer the edge_feats_dim on the underlying graph
        self._need_to_initialize_nbr_feats = True
        self._edge_feats_dim = None
        self._nbr_feats = None

    @property
    def num_nbrs(self) -> List[int]:
        return self._num_nbrs

    def reset_state(self) -> None:
        self._nbr_ids.fill_(PADDED_NODE_ID)
        self._nbr_times.zero_()
        self._write_pos.zero_()

        if self._nbr_feats is not None:
            self._nbr_feats.zero_()

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        self._initialize_nbr_feats_if_needed(dg)
        device = dg.device
        self._move_queues_to_device_if_needed(device)  # No-op after first batch

        batch.nids, batch.times = [], []  # type: ignore
        batch.nbr_nids, batch.nbr_times = [], []  # type: ignore
        batch.nbr_feats = []  # type: ignore

        for hop, num_nbrs in enumerate(self.num_nbrs):
            if hop == 0:
                if self._seed_nodes_key is None:
                    seed = [batch.src, batch.dst]
                    times = [batch.time.repeat(2)]  # Real link times
                    if hasattr(batch, 'neg'):
                        batch.neg = batch.neg.to(device)
                        seed.append(batch.neg)
                        times.append(batch.neg_time)
                    seed_nodes = torch.cat(seed)
                    seed_times = torch.cat(times)
                    if seed_nodes.numel() == 0:
                        for hop in range(len(self.num_nbrs)):
                            batch.nids.append(torch.empty(0, dtype=torch.int32))
                            batch.times.append(torch.empty(0, dtype=torch.int64))
                            batch.nbr_nids.append(torch.empty(0, dtype=torch.int32))
                            batch.nbr_times.append(torch.empty(0, dtype=torch.int64))
                            batch.nbr_feats.append(
                                torch.empty(0, self._edge_feats_dim).float()
                            )
                        return batch
                else:
                    seed_nodes = getattr(batch, self._seed_nodes_key)
                    if seed_nodes is None:
                        return batch
                    else:
                        seed_nodes = seed_nodes.to(device)
                        seed_times = batch.node_times.to(device)  # type: ignore
            else:
                seed_nodes = batch.nbr_nids[hop - 1].flatten()  # type: ignore
                seed_times = batch.nbr_times[hop - 1].flatten()  # type: ignore

            nbr_nids, nbr_times, nbr_feats = self._get_recency_neighbors(
                seed_nodes, seed_times, num_nbrs
            )

            batch.nids.append(seed_nodes)  # type: ignore
            batch.times.append(seed_times)  # type: ignore
            batch.nbr_nids.append(nbr_nids)  # type: ignore
            batch.nbr_times.append(nbr_times)  # type: ignore
            batch.nbr_feats.append(nbr_feats)  # type: ignore

        if batch.src.numel():
            self._update(batch)
        return batch

    def _get_recency_neighbors(
        self, node_ids: torch.Tensor, query_times: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, ...]:
        assert self._nbr_feats is not None  # For mypy
        B = self._max_nbrs  # buffer size

        nbr_nids = self._nbr_ids[node_ids]  # (N, B)
        nbr_times = self._nbr_times[node_ids]  # (N, B)
        nbr_feats = self._nbr_feats[node_ids]  # (N, B, edge_dim)
        write_pos = self._write_pos[node_ids]  # (N,)

        # Unroll indices to all buffers, so that last write is at index -1
        # If we had no query_time constraint, we would just take the last k entries
        candidate_idx = write_pos[:, None] - torch.arange(B, 0, -1, device=self._device)
        candidate_idx %= B  # (N, B) with oldest ... newest

        # Read the neighbor times in that unrolled order, and get query_times mask
        candidate_times = torch.gather(nbr_times, 1, candidate_idx)  # (N, B)
        time_mask = candidate_times < query_times[:, None]  # (N, B)
        time_mask[torch.gather(nbr_nids, 1, candidate_idx) == PADDED_NODE_ID] = False

        # For each node, find the rightmost valid entry, i.e the last index which
        # satisfies the time mask. We will read k slots backwards from there.
        # Since we write out buffers chronologically, we just search for rightmost valid entry
        pos = torch.arange(B, device=self._device)
        last_valid_pos = (time_mask * pos).amax(dim=1)  # (N,)
        N = len(node_ids)
        last_valid_pos = torch.where(
            time_mask.any(dim=1),
            (time_mask * torch.arange(B, device=self._device)).amax(dim=1),
            torch.full((N,), -1, device=self._device),
        )

        # We figured out the last time constraint valid position, now build the k-window
        # ending at last_valid_pos (with wraparound). Since last_valid_pos is relative to
        # the unrolled buffer ordering, we can read backwards k slots, and know that any
        # negative entries imply that less than k entries satisfies time mask. We clamp those to -1.
        offset = torch.arange(k - 1, -1, -1, device=self._device)  # [k - 1, ..., 0]
        gather_pos = last_valid_pos[:, None] - offset[:, None].T  # (N, k)
        gather_pos = torch.clamp(gather_pos, min=-1)  # Map all invalid entries to -1

        # For each node, we have something like [-1, -1, -1, 0, 1, 2, ..., x]
        # where the whole array is of size k, the x + 1 non-negative entries refer
        # to indices in the unrolled buffer which satisfy the time constraint.
        # We need to now map back to the original buffer indices (not unrolled)
        # Here, we take the actual circular buffer indices corresponding to the columns
        # we want. We temporarily clamp gather_pos negatives (invalid time entries) to 0
        # but they get replace with -1 according to the torch.where mask gather_pos >= 0
        out_idx = torch.where(
            gather_pos >= 0,
            torch.gather(candidate_idx, 1, gather_pos.clamp(min=0)),
            torch.full_like(gather_pos, -1),
        )

        # Crate a mask of valid indices, and clamp out_idx for safe gather. We'll make sure to
        # only write the entries at positions where valid_mask is True
        valid_mask = out_idx >= 0
        safe_idx = out_idx.clamp(min=0)

        # Gather out tensors
        out_nbrs = torch.gather(nbr_nids, 1, safe_idx)
        out_times = torch.gather(nbr_times, 1, safe_idx)
        out_feats = torch.gather(
            nbr_feats, 1, safe_idx.unsqueeze(-1).expand(-1, -1, self._edge_feats_dim)
        )

        # Overwrite invalid positions in-place
        out_nbrs[~valid_mask] = PADDED_NODE_ID
        out_times[~valid_mask] = 0
        out_feats[~valid_mask] = 0.0

        return out_nbrs, out_times, out_feats

    def _update(self, batch: DGBatch) -> None:
        assert self._nbr_feats is not None  # For mypy
        if batch.edge_feats is None:
            edge_feats = torch.zeros(
                (len(batch.src), self._edge_feats_dim), device=self._device
            )
        else:
            edge_feats = batch.edge_feats

        if self._directed:
            node_ids, nbr_nids, times = batch.src, batch.dst, batch.time
        else:
            # It's fine that times is out-of-order here since we sort below
            node_ids = torch.cat([batch.src, batch.dst])
            nbr_nids = torch.cat([batch.dst, batch.src])
            times = torch.cat([batch.time, batch.time])
            edge_feats = torch.cat([edge_feats, edge_feats])

        # Lexicographical sort by node id and time. Duplicate nodes will be adjacent.
        # Each nodes events will be sorted chronologically
        max_time = times.max() + 1
        composite_key = node_ids * max_time + times
        perm = torch.argsort(composite_key, stable=True)

        sorted_nodes = node_ids[perm]
        sorted_nbr_ids = nbr_nids[perm]
        sorted_times = times[perm]
        sorted_feats = edge_feats[perm]

        # All the tensors we need to write are properly sorted and groupbed by node.
        # However, in order for deterministic scatter on multi-dimensional arrays, we
        # cannot afford to have multiple tensors written at same buffer position.
        # E.g. This occurs if we have more node events than the buffer capacity.
        # Therefore, we do another index select that retains only the last B entries
        # for each node. This guarantees at most one write per buffer position, and
        # will still be sorted chronologically, with grouping by nodes.
        B = self._max_nbrs
        _, inv, cnts = torch.unique_consecutive(
            sorted_nodes, return_inverse=True, return_counts=True
        )
        cumcnts = torch.cat(
            [torch.tensor([0], device=self._device), cnts.cumsum(0)[:-1]]
        )
        pos_in_group = (
            torch.arange(len(sorted_nodes), device=self._device) - cumcnts[inv]
        )
        mask = pos_in_group >= (cnts[inv] - B)

        sorted_nodes = sorted_nodes[mask]
        sorted_nbr_ids = sorted_nbr_ids[mask]
        sorted_times = sorted_times[mask]
        sorted_feats = sorted_feats[mask]

        # Count number of writes per node as a cumulative offset
        _, inv, cnts = torch.unique_consecutive(
            sorted_nodes, return_inverse=True, return_counts=True
        )
        cum_cnts = torch.cat(
            [torch.tensor([0], device=self._device), cnts[:-1]]
        ).cumsum(dim=0)
        offsets = torch.arange(len(sorted_nodes), device=self._device) - cum_cnts[inv]

        # Compute write indices using current write position and offsets
        write_idx = (self._write_pos[sorted_nodes] + offsets) % self._max_nbrs

        # Scatter into buffers. Correct "last write wins" for features, since we have at most B writes
        self._nbr_ids[sorted_nodes, write_idx] = sorted_nbr_ids
        self._nbr_times[sorted_nodes, write_idx] = sorted_times
        self._nbr_feats[sorted_nodes, write_idx, :] = sorted_feats

        # Increment write_pos per node
        num_writes = torch.ones_like(sorted_nodes, device=self._device)
        self._write_pos.scatter_add_(0, sorted_nodes.long(), num_writes)

    def _move_queues_to_device_if_needed(self, device: torch.device) -> None:
        assert self._nbr_feats is not None  # For mypy
        if device != self._device:
            self._device = device
            self._nbr_ids = self._nbr_ids.to(device)
            self._nbr_times = self._nbr_times.to(device)
            self._nbr_feats = self._nbr_feats.to(device)
            self._write_pos = self._write_pos.to(device)

    def _initialize_nbr_feats_if_needed(self, dg: DGraph) -> None:
        if self._need_to_initialize_nbr_feats:
            self._edge_feats_dim = dg.edge_feats_dim or 0  # type: ignore
            self._nbr_feats = torch.zeros(
                (self._num_nodes, self._max_nbrs, self._edge_feats_dim)  # type: ignore
            )
            self._need_to_initialize_nbr_feats = False


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
