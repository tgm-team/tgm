from __future__ import annotations

import warnings
from typing import List, Set, Tuple

import torch

from tgm import DGBatch, DGraph
from tgm.constants import PADDED_NODE_ID
from tgm.core._storage import DGSliceTracker
from tgm.hooks import StatefulHook, StatelessHook
from tgm.util.logging import _get_logger

logger = _get_logger(__name__)


class NeighborSamplerHook(StatelessHook):
    """Load data from DGraph using a memory based sampling function.

    Args:
        num_nbrs (List[int]): Number of neighbors to sample at each hop (-1 to keep all)
        directed (bool): If true, aggregates interactions in src->dst direction only (default=False).
        seed_nodes_keys ([List[str]): List of batch attribute keys to identify the initial seed nodes to sample for.
        seed_times_keys ([List[str]): List of batch attribute keys to identify the initial seed times to sample for.

    Note:
        The order of the output tensors respect the order of seed_nodes_keys.
        For instance, for seed node keys ['src', 'dst', 'neg'] will have the first output index (hop 0) contain the concatenation
        of batch.src, batch.dst, batch.neg (in that order). The next index (hop 1) will contain first-hop neighbors of batch.src
        followed by first-hop neighbors of batch.dst, and then those of batch.neg. This pattern repeats for deeper hops.

    Raises:
        ValueError: If the num_nbrs list is empty or has non-positive entries.
        ValueError: If len(seed_nodes_keys) != len(seed_times_keys).
    """

    requires: Set[str] = set()
    produces = {'nids', 'nbr_nids', 'nbr_times', 'nbr_feats', 'seed_node_nbr_mask'}

    def __init__(
        self,
        num_nbrs: List[int],
        seed_nodes_keys: List[str],
        seed_times_keys: List[str],
        directed: bool = False,
    ) -> None:
        if not len(num_nbrs):
            raise ValueError('num_nbrs must be non-empty')
        if not all([isinstance(x, int) and (x > 0) for x in num_nbrs]):
            raise ValueError('Each value in num_nbrs must be a positive integer')
        self._num_nbrs = num_nbrs
        self._directed = directed

        if len(seed_nodes_keys) != len(seed_times_keys):
            raise ValueError(
                f'len(seed_nodes_keys) ({len(seed_nodes_keys)}) '
                f'!= len(seed_times_keys) ({len(seed_times_keys)})\n'
                f'seed_nodes_keys={seed_nodes_keys}, '
                f'seed_times_keys={seed_times_keys}'
            )
        self._seed_nodes_keys = seed_nodes_keys
        self._seed_times_keys = seed_times_keys
        logger.debug(
            'Seed nodes keys: %s, Seed times keys: %s',
            self._seed_nodes_keys,
            self._seed_times_keys,
        )
        self._warned_seed_None = False

    @property
    def num_nbrs(self) -> List[int]:
        return self._num_nbrs

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        batch.nids, batch.times = [], []  # type: ignore
        batch.nbr_nids, batch.nbr_times = [], []  # type: ignore
        batch.nbr_feats = []  # type: ignore

        def _append_empty_hop() -> None:
            batch.nids.append(torch.empty(0, dtype=torch.int32))  # type: ignore
            batch.times.append(torch.empty(0, dtype=torch.int64))  # type: ignore
            batch.nbr_nids.append(torch.empty(0, dtype=torch.int32))  # type: ignore
            batch.nbr_times.append(torch.empty(0, dtype=torch.int64))  # type: ignore
            batch.nbr_feats.append(  # type: ignore
                torch.empty(0, dg.edge_feats_dim).float()  # type: ignore
            )

        seed_nodes, seed_times, seed_node_nbr_mask = self._get_seed_tensors(batch)
        if not seed_nodes.numel():
            logger.debug('No seed_nodes found, appending empty hop information')
            for _ in self.num_nbrs:
                _append_empty_hop()
            return batch

        for hop, num_nbrs in enumerate(self.num_nbrs):
            if hop > 0:
                seed_nodes = batch.nbr_nids[hop - 1].flatten()  # type: ignore
                seed_times = batch.nbr_times[hop - 1].flatten()  # type: ignore

            # TODO: Storage needs to use the right device

            # We slice on batch.start_time so that we only consider neighbor events
            # that occurred strictly before this batch
            logger.debug(
                'Getting uniform nbrs for hop %d with %d seed nodes',
                hop,
                seed_nodes.numel(),
            )
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

        batch.seed_node_nbr_mask = seed_node_nbr_mask  # type: ignore
        return batch

    def _get_seed_tensors(
        self, batch: DGBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        device = batch.src.device
        seeds, times = [], []
        seed_node_mask = dict()

        offset = 0
        for node_attr, time_attr in zip(self._seed_nodes_keys, self._seed_times_keys):
            missing = [
                attr for attr in (node_attr, time_attr) if not hasattr(batch, attr)
            ]
            if missing:
                raise ValueError(f'Missing seed attributes {missing} on batch')

            seed = getattr(batch, node_attr)
            time = getattr(batch, time_attr)

            for name, tensor in [(node_attr, seed), (time_attr, time)]:
                # We recover from tensor = None, since the current batch could just
                # be missing certain attributes (e.g. dynamic node events), but for
                # non-Tensor and non-None attrs we explicitly raise
                if tensor is None:
                    logger.debug(
                        'Seed attribute %s is None on this batch, skipping', name
                    )
                    if not self._warned_seed_None:
                        warnings.warn(
                            f'Seed attribute {name} is None on this batch, skipping this batch. '
                            'Future occurrences will also be skipped but the warning will be suppressed',
                            UserWarning,
                        )
                        self._warned_seed_None = True
                    break
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError(f'{name} must be a Tensor, got {type(tensor)}')
                if tensor.ndim != 1:
                    raise ValueError(f'{name} must be 1-D, got shape {tensor.shape}')

                # Bounds checks
                # TODO: Infer self._num_nodes from underlying graph
                self._num_nodes = float('inf')
                if name == node_attr:
                    if (tensor < 0).any() or (tensor >= self._num_nodes).any():
                        raise ValueError(
                            f'Seed nodes in {name} must satisfy 0 <= x < {self._num_nodes}, '
                            f'got values in range [{tensor.min().item()}, {tensor.max().item()}]'
                        )
                    seeds.append(seed.to(device))
                    num_seed_nodes = tensor.shape[0]
                    seed_node_mask[name] = torch.arange(offset, num_seed_nodes).to(
                        device
                    )
                    offset += num_seed_nodes
                elif name == time_attr:
                    if (tensor < 0).any():
                        raise ValueError(
                            f'Seed times in {name} must be >= 0, got min value: {tensor.min().item()}'
                        )
                    times.append(time.to(device))

        if seeds and times:
            seed_nodes, seed_times = torch.cat(seeds), torch.cat(times)
        else:
            seed_nodes = torch.empty(0, dtype=torch.int32, device=device)
            seed_times = torch.empty(0, dtype=torch.int64, device=device)
        return seed_nodes, seed_times, seed_node_mask


class RecencyNeighborHook(StatefulHook):
    requires: Set[str] = set()
    produces = {
        'nids',
        'nbr_nids',
        'times',
        'nbr_times',
        'nbr_feats',
        'seed_node_nbr_mask',
    }

    """Load neighbors from DGraph using a recency sampling. Each node maintains a fixed number of recent neighbors.

    Args:
        num_nodes (int): Total number of nodes to track.
        num_nbrs (List[int]): Number of neighbors to sample at each hop (max neighbors to keep).
        directed (bool): If true, aggregates interactions in src->dst direction only (default=False).
                                               If not specified, defaults to batch edges: ['src', 'dst']
                                               If not specified, defaults to batch times: ['time', 'time']
        seed_nodes_keys ([List[str]): List of batch attribute keys to identify the initial seed nodes to sample for.
        seed_times_keys ([List[str]): List of batch attribute keys to identify the initial seed times to sample for.

    Note:
        The order of the output tensors respect the order of seed_nodes_keys.
        For instance, for seed node keys ['src', 'dst', 'neg'] will have the first output index (hop 0) contain the concatenation
        of batch.src, batch.dst, batch.neg (in that order). The next index (hop 1) will contain first-hop neighbors of batch.src
        followed by first-hop neighbors of batch.dst, and then those of batch.neg. This pattern repeats for deeper hops.

    Raises:
        ValueError: If the num_nbrs list is empty or has non-positive entries.
        ValueError: If len(seed_nodes_keys) != len(seed_times_keys).
    """

    def __init__(
        self,
        num_nodes: int,
        num_nbrs: List[int],
        seed_nodes_keys: List[str],
        seed_times_keys: List[str],
        directed: bool = False,
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

        if len(seed_nodes_keys) != len(seed_times_keys):
            raise ValueError(
                f'len(seed_nodes_keys) ({len(seed_nodes_keys)}) '
                f'!= len(seed_times_keys) ({len(seed_times_keys)})\n'
                f'seed_nodes_keys={seed_nodes_keys}, '
                f'seed_times_keys={seed_times_keys}'
            )
        self._seed_nodes_keys = seed_nodes_keys
        self._seed_times_keys = seed_times_keys
        logger.debug(
            'Seed nodes keys: %s, Seed times keys: %s',
            self._seed_nodes_keys,
            self._seed_times_keys,
        )
        self._warned_seed_None = False

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
        self._move_queues_to_device_if_needed(dg.device)

        batch.nids, batch.times = [], []  # type: ignore
        batch.nbr_nids, batch.nbr_times = [], []  # type: ignore
        batch.nbr_feats = []  # type: ignore

        def _append_empty_hop() -> None:
            batch.nids.append(torch.empty(0, dtype=torch.int32))  # type: ignore
            batch.times.append(torch.empty(0, dtype=torch.int64))  # type: ignore
            batch.nbr_nids.append(torch.empty(0, dtype=torch.int32))  # type: ignore
            batch.nbr_times.append(torch.empty(0, dtype=torch.int64))  # type: ignore
            batch.nbr_feats.append(  # type: ignore
                torch.empty(0, dg.edge_feats_dim).float()  # type: ignore
            )

        seed_nodes, seed_times, seed_node_mask = self._get_seed_tensors(batch)
        if not seed_nodes.numel():
            logger.debug('No seed_nodes found, appending empty hop information')
            for _ in self.num_nbrs:
                _append_empty_hop()
            return batch

        for hop, num_nbrs in enumerate(self.num_nbrs):
            if hop > 0:
                seed_nodes = batch.nbr_nids[hop - 1].flatten()  # type: ignore
                seed_times = batch.nbr_times[hop - 1].flatten()  # type: ignore

            logger.debug(
                'Getting last %d nbrs for hop %d with %d seed nodes',
                num_nbrs,
                hop,
                seed_nodes.numel(),
            )
            nbr_nids, nbr_times, nbr_feats = self._get_recency_neighbors(
                seed_nodes, seed_times, num_nbrs
            )

            batch.nids.append(seed_nodes)  # type: ignore
            batch.times.append(seed_times)  # type: ignore
            batch.nbr_nids.append(nbr_nids)  # type: ignore
            batch.nbr_times.append(nbr_times)  # type: ignore
            batch.nbr_feats.append(nbr_feats)  # type: ignore

        batch.seed_node_nbr_mask = seed_node_mask  # type: ignore
        if batch.src.numel():
            logger.debug('Updating circular buffers')
            self._update(batch)
        return batch

    def _get_seed_tensors(
        self, batch: DGBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        device = batch.src.device
        seeds, times = [], []
        seed_node_mask = dict()

        offset = 0
        for node_attr, time_attr in zip(self._seed_nodes_keys, self._seed_times_keys):
            missing = [
                attr for attr in (node_attr, time_attr) if not hasattr(batch, attr)
            ]
            if missing:
                raise ValueError(f'Missing seed attributes {missing} on batch')

            seed = getattr(batch, node_attr)
            time = getattr(batch, time_attr)

            for name, tensor in [(node_attr, seed), (time_attr, time)]:
                # We recover from tensor = None, since the current batch could just
                # be missing certain attributes (e.g. dynamic node events), but for
                # non-Tensor and non-None attrs we explicitly raise
                if tensor is None:
                    logger.debug(
                        'Seed attribute %s is None on this batch, skipping', name
                    )
                    if not self._warned_seed_None:
                        warnings.warn(
                            f'Seed attribute {name} is None on this batch, skipping this batch. '
                            'Future occurrences will also be skipped but the warning will be suppressed',
                            UserWarning,
                        )
                        self._warned_seed_None = True
                    break
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError(f'{name} must be a Tensor, got {type(tensor)}')
                if tensor.ndim != 1:
                    raise ValueError(f'{name} must be 1-D, got shape {tensor.shape}')

                # Bounds checks
                if name == node_attr:
                    if (tensor < 0).any() or (tensor >= self._num_nodes).any():
                        raise ValueError(
                            f'Seed nodes in {name} must satisfy 0 <= x < {self._num_nodes}, '
                            f'got values in range [{tensor.min().item()}, {tensor.max().item()}]'
                        )
                    seeds.append(seed.to(device))
                    num_seed_nodes = tensor.shape[0]
                    seed_node_mask[name] = torch.arange(offset, num_seed_nodes).to(
                        device
                    )
                    offset += num_seed_nodes
                elif name == time_attr:
                    if (tensor < 0).any():
                        raise ValueError(
                            f'Seed times in {name} must be >= 0, got min value: {tensor.min().item()}'
                        )
                    times.append(time.to(device))

        if seeds and times:
            seed_nodes, seed_times = torch.cat(seeds), torch.cat(times)
        else:
            seed_nodes = torch.empty(0, dtype=torch.int32, device=device)
            seed_times = torch.empty(0, dtype=torch.int64, device=device)
        return seed_nodes, seed_times, seed_node_mask

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


class NeighborCooccurrenceHook(StatelessHook):
    requires = {'nbr_nids', 'nbr_idx_map'}
    produces = {'neighbours_coocurence'}

    def __init__(self, seed_nodes_pairs_keys: List[Tuple[str]]):
        all_pairs = True
        for nodes_pairs in seed_nodes_pairs_keys:
            all_pairs &= len(nodes_pairs) == 2

        if not all_pairs:
            raise ValueError(
                'Neighbour co-occurence is computed according to a pair of nodes'
            )
        self._seed_nodes_pairs_keys = seed_nodes_pairs_keys

    def _count_nodes_freq(
        self, src_nbrs: torch.Tensor, dst_nbrs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert src_nbrs.ndim == 2 and src_nbrs.shape == dst_nbrs.shape

        # cross occurrences count
        cross_mask = src_nbrs.unsqueeze(dim=1) == dst_nbrs.unsqueeze(dim=-1)

        # self occurrences count
        src_mask = src_nbrs.unsqueeze(dim=1) == src_nbrs.unsqueeze(dim=-1)
        dst_mask = dst_nbrs.unsqueeze(dim=1) == dst_nbrs.unsqueeze(dim=-1)

        src_freq = torch.stack([src_mask.sum(1), cross_mask.sum(1)], dim=2).float()
        dst_freq = torch.stack([dst_mask.sum(1), cross_mask.sum(2)], dim=2).float()

        # Mask out PADDED_NODE_ID
        src_freq[src_nbrs == PADDED_NODE_ID] = 0.0
        dst_freq[dst_nbrs == PADDED_NODE_ID] = 0.0
        return src_freq, dst_freq

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        device = batch.nbr_nids.device  # type: ignore
        missing = []
        for src, dst in self._seed_nodes_pairs_keys:  # type: ignore
            if not hasattr(batch, src):
                missing.append(src)
            if not hasattr(batch, dst):
                missing.append(dst)

        if missing:
            raise ValueError(f'Missing seed attributes {missing} on batch')

        neighbours = batch.nbr_nids[0]  # type: ignore
        all_pairs_coocurence = []
        for src_key, dst_key in self._seed_nodes_pairs_keys:  # type: ignore
            src = getattr(batch, src_key)
            dst = getattr(batch, dst_key)

            src.shape[0]
            src_neighbours = neighbours[batch.nbr_idx_map[src_key]]  # type: ignore
            dst_neighbours = neighbours[batch.nbr_idx_map[dst_key]]  # type: ignore

            # include seed nodes are neighbors themselves
            src_neighbours = torch.cat([src[:, None], src_neighbours], dim=1)
            dst_neighbours = torch.cat([dst[:, None], dst_neighbours], dim=1)

            if src.shape[0] != dst.shape[0]:
                src = torch.repeat_interleave(
                    src, repeats=dst.shape[0], dim=0
                )  # This only works when (src,neg) is passed. If (neg,src) is passed. It will break the logic
                dst = dst.repeat(src.shape[0])

                src_neighbours = torch.repeat_interleave(
                    src_neighbours, repeats=dst.shape[0], dim=0
                )
                dst_neighbours = dst_neighbours.repeat(src.shape[0], 1)
            src_coocurence, dst_coocurence = self._count_nodes_freq(
                src_neighbours, dst_neighbours
            )
            pair_coocurence = torch.stack([src_coocurence, dst_coocurence], dim=0)
            all_pairs_coocurence.append(pair_coocurence)

        batch.neighbours_coocurence = torch.stack(all_pairs_coocurence, dim=0).to(  # type: ignore
            device
        )
        return batch
