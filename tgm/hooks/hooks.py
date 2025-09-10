from __future__ import annotations

from dataclasses import is_dataclass
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

        all_nids = torch.cat(nids, dim=0)
        unique_nids = torch.unique(all_nids, sorted=True)

        batch.unique_nids = unique_nids  # type: ignore
        batch.global_to_local = lambda x: torch.searchsorted(unique_nids, x)  # type: ignore

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
        batch.neg = torch.randint(  # type: ignore
            self.low, self.high, size, dtype=torch.long, device=dg.device
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
            batch.src, batch.dst, batch.time, split_mode=self.split_mode
        )
        batch.neg_batch_list = [  # type: ignore
            torch.tensor(neg_batch, dtype=torch.long, device=dg.device)
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
        edge_feats_dim (int): Edge feature dimension on the dynamic graph.
        directed (bool): If true, aggregates interactions in src->dst direction only (default=False).

    Raises:
        ValueError: If the num_nbrs list is empty.
    """

    def __init__(
        self,
        num_nodes: int,
        num_nbrs: List[int],
        edge_feats_dim: int,
        directed: bool = False,
    ) -> None:
        if not len(num_nbrs):
            raise ValueError('num_nbrs must be non-empty')
        if not all([isinstance(x, int) and (x > 0) for x in num_nbrs]):
            raise ValueError('Each value in num_nbrs must be a positive integer')

        self._num_nbrs = num_nbrs
        self._max_nbrs = max(num_nbrs)
        self._directed = directed
        self._edge_feats_dim = edge_feats_dim
        self._device = torch.device('cpu')

        self._nbr_ids = torch.full(
            (num_nodes, self._max_nbrs), PADDED_NODE_ID, dtype=torch.long
        )
        self._nbr_times = torch.zeros((num_nodes, self._max_nbrs), dtype=torch.long)
        self._nbr_feats = torch.zeros((num_nodes, self._max_nbrs, edge_feats_dim))
        self._write_pos = torch.zeros(num_nodes, dtype=torch.long)

    @property
    def num_nbrs(self) -> List[int]:
        return self._num_nbrs

    def reset_state(self) -> None:
        self._nbr_ids.fill_(PADDED_NODE_ID)
        self._nbr_times.zero_()
        self._nbr_feats.zero_()
        self._write_pos.zero_()

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        device = dg.device
        self._move_queues_to_device_if_needed(device)  # No-op after first batch

        batch.nids, batch.times = [], []  # type: ignore
        batch.nbr_nids, batch.nbr_times = [], []  # type: ignore
        batch.nbr_feats = []  # type: ignore

        def print_for_node(y):
            print(
                f'Circular buffer checking for node {y}: nbrs = {self._nbr_ids[y]}, times = {self._nbr_times[y]}, feats: {self._nbr_feats[y]}'
            )

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

        # print_for_node(290)
        # print(batch.src, batch.edge_feats.float())
        self._update(batch)
        # print_for_node(290)
        # input()
        return batch

    def _get_recency_neighbors(
        self, node_ids: torch.Tensor, query_times: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, ...]:
        log = node_ids.tolist() == [233, 9, 8400, 8235]
        log = False
        B = self._max_nbrs  # buffer size

        nbr_nids = self._nbr_ids[node_ids]  # (N, B)
        nbr_times = self._nbr_times[node_ids]  # (N, B)
        nbr_feats = self._nbr_feats[node_ids]  # (N, B, edge_dim)
        write_pos = self._write_pos[node_ids]  # (N,)

        if log:
            print('Nbr nids: ', nbr_nids)
            print('Nbr times: ', nbr_times)
            print('Write pos: ', write_pos)

        # Unroll indices to all buffers, so that last write is at index -1
        # If we had no query_time constraint, we would just take the last k entries
        # Shape (N, B) with oldest ... newest
        # candidate_idx = write_pos[:, None] - torch.arange(1, B + 1, device=self._device)
        candidate_idx = write_pos[:, None] - torch.arange(B, 0, -1, device=self._device)
        candidate_idx %= B

        # Read the neighbor times in that unrolled order, and get query_times mask
        candidate_times = torch.gather(nbr_times, 1, candidate_idx)  # (N, B)
        time_mask = candidate_times < query_times[:, None]  # (N, B)
        time_mask[torch.gather(nbr_nids, 1, candidate_idx) == PADDED_NODE_ID] = False

        if log:
            print('Candidate idx: ', candidate_idx)
            print('Candidate times: ', candidate_times)
            print('Time mask: ', time_mask)

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

        if log:
            print('Last valid pos: ', last_valid_pos)

        # We figured out the last time constraint valid position, now build the k-window
        # ending at last_valid_pos (with wraparound). Since last_valid_pos is relative to
        # the unrolled buffer ordering, we can read backwards k slots, and know that any
        # negative entries imply that less than k entries satisfies time mask. We clamp those to -1.
        offset = torch.arange(k - 1, -1, -1, device=self._device)  # [k - 1, ..., 0]
        gather_pos = last_valid_pos[:, None] - offset[:, None].T  # (N, k)
        gather_pos = torch.clamp(gather_pos, min=-1)  # Map all invalid entries to -1

        if log:
            print('Offset: ', offset)
            print('Gather pos: ', gather_pos)

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

        if log:
            print('Out idx: ', out_idx)

        # Prepare output tensors filled with PAD / zeros
        out_nbrs = torch.full_like(out_idx, PADDED_NODE_ID)
        out_times = torch.zeros_like(out_idx)
        out_feats = torch.zeros(
            (out_idx.size(0), out_idx.size(1), self._edge_feats_dim),
            device=self._device,
            dtype=nbr_feats.dtype,
        )

        # Mask of valid indices
        valid_mask = out_idx >= 0

        # Clamp to 0 to safely gather
        safe_idx = out_idx.clamp(min=0)

        # Gather all entries
        gathered_nbrs = torch.gather(nbr_nids, 1, safe_idx)
        gathered_times = torch.gather(nbr_times, 1, safe_idx)
        gathered_feats = torch.gather(
            nbr_feats, 1, safe_idx.unsqueeze(-1).expand(-1, -1, self._edge_feats_dim)
        )

        # Fill only valid positions
        out_nbrs[valid_mask] = gathered_nbrs[valid_mask]
        out_times[valid_mask] = gathered_times[valid_mask]
        out_feats[valid_mask] = gathered_feats[valid_mask]

        if log:
            print('out nbrs:', out_nbrs)
            print('out times:', out_times)
            input()

        return out_nbrs, out_times, out_feats

    def _update(self, batch: DGBatch) -> None:
        if batch.edge_feats is None:
            edge_feats = torch.zeros(
                (len(batch.src), self._edge_feats_dim), device=self._device
            )
        else:
            edge_feats = batch.edge_feats.float()

        if self._directed:
            node_ids, nbr_nids, times = batch.src, batch.dst, batch.time
        else:
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

        # print(
        #    f'Circular buffer sorted node ids: {sorted_nodes}, sorted nbrs: {sorted_nbr_ids} and sorted node times : {sorted_times}'
        # )

        # Count number of writes per node as a cumulative offset
        _, inv, cnts = torch.unique_consecutive(
            sorted_nodes, return_inverse=True, return_counts=True
        )
        cum_cnts = torch.cat(
            [torch.tensor([0], device=self._device), cnts[:-1]]
        ).cumsum(dim=0)
        offsets = torch.arange(len(sorted_nodes), device=self._device) - cum_cnts[inv]

        # print(f'Circular buffer cumsumcounts: {cum_cnts}')
        # print(f'Circular buffer offsets: {offsets}')

        # Compute write indices using current write position and offets
        write_idx = (self._write_pos[sorted_nodes] + offsets) % self._max_nbrs
        # print(f'Circular buffer idx: {idx}')

        # Scatter updates into buffers
        # def print_for_node(y):
        #    print(
        #        f'Circular buffer checking for node {y}: nbrs = {self._nbr_ids[y]}, times = {self._nbr_times[y]}, feats: {self._nbr_feats[y]}'
        #    )

        # print('---- Pre scatter ----')
        # print_for_node(0)
        # print_for_node(1)
        # print_for_node(2)

        # 5. Scatter into buffers
        self._nbr_ids[sorted_nodes, write_idx] = sorted_nbr_ids
        self._nbr_times[sorted_nodes, write_idx] = sorted_times
        self._nbr_feats[sorted_nodes, write_idx, :] = sorted_feats

        # self._nbr_ids[sorted_nodes, write_idx] = sorted_nbr_ids
        # self._nbr_times[sorted_nodes, write_idx] = sorted_times
        # self._nbr_feats[sorted_nodes, write_idx, :] = sorted_feats

        # print('---- Post scatter ----')
        # print_for_node(0)
        # print_for_node(1)
        # print_for_node(2)

        # Increment write_pos per node
        num_writes = torch.ones_like(sorted_nodes, device=self._device)
        self._write_pos.scatter_add_(0, sorted_nodes, num_writes)

    def _move_queues_to_device_if_needed(self, device: torch.device) -> None:
        if device != self._device:
            self._device = device
            self._nbr_ids = self._nbr_ids.to(device)
            self._nbr_times = self._nbr_times.to(device)
            self._nbr_feats = self._nbr_feats.to(device)
            self._write_pos = self._write_pos.to(device)


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
