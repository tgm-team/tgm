from __future__ import annotations

import warnings
from typing import List, Tuple

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
        directed (bool): If true, aggregates interactions in edge_src->edge_dst direction only (default=False).
        seed_nodes_keys ([List[str]): List of batch attribute keys to identify the initial seed nodes to sample for.
        seed_times_keys ([List[str]): List of batch attribute keys to identify the initial seed times to sample for.

    Note:
        The order of the output tensors respect the order of seed_nodes_keys.
        For instance, for seed node keys ['edge_src', 'edge_dst', 'neg'] will have the first output index (hop 0) contain the concatenation
        of batch.edge_src, batch.edge_dst, batch.neg (in that order). The next index (hop 1) will contain first-hop neighbors of batch.edge_src
        followed by first-hop neighbors of batch.edge_dst, and then those of batch.neg. This pattern repeats for deeper hops.

    Raises:
        ValueError: If the num_nbrs list is empty or has non-positive entries.
        ValueError: If len(seed_nodes_keys) != len(seed_times_keys).
    """

    requires = {'edge_src', 'edge_dst', 'edge_time'}
    produces = {
        'seed_nids',
        'seed_times',
        'nbr_nids',
        'nbr_edge_time',
        'nbr_edge_x',
        'seed_node_nbr_mask',
    }

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
        batch.seed_nids, batch.seed_times = [], []  # type: ignore
        batch.nbr_nids, batch.nbr_edge_time = [], []  # type: ignore
        batch.nbr_edge_x = []  # type: ignore

        def _append_empty_hop() -> None:
            batch.seed_nids.append(torch.empty(0, dtype=torch.int32))  # type: ignore
            batch.seed_times.append(torch.empty(0, dtype=torch.int64))  # type: ignore
            batch.nbr_nids.append(torch.empty(0, dtype=torch.int32))  # type: ignore
            batch.nbr_edge_time.append(torch.empty(0, dtype=torch.int64))  # type: ignore
            batch.nbr_edge_x.append(  # type: ignore
                torch.empty(0, dg.edge_x_dim).float()  # type: ignore
            )

        seed_nodes, seed_times, seed_node_nbr_mask = self._get_seed_tensors(batch)
        if not seed_nodes.numel():
            logger.debug('No seed_nodes found, appending empty hop information')
            for _ in self.num_nbrs:
                _append_empty_hop()
            batch.seed_node_nbr_mask = seed_node_nbr_mask  # type: ignore
            return batch

        for hop, num_nbrs in enumerate(self.num_nbrs):
            if hop > 0:
                seed_nodes = batch.nbr_nids[hop - 1].flatten()  # type: ignore
                seed_times = batch.nbr_edge_time[hop - 1].flatten()  # type: ignore

            # TODO: Storage needs to use the right device

            # We slice on batch.start_time so that we only consider neighbor events
            # that occurred strictly before this batch
            logger.debug(
                'Getting uniform nbrs for hop %d with %d seed nodes',
                hop,
                seed_nodes.numel(),
            )
            nbr_nids, nbr_edge_time, nbr_edge_x = dg._storage.get_nbrs(
                seed_nodes,
                num_nbrs=num_nbrs,
                slice=DGSliceTracker(end_time=int(batch.edge_time.min()) - 1),
                directed=self._directed,
            )

            batch.seed_nids.append(seed_nodes)  # type: ignore
            batch.seed_times.append(seed_times)  # type: ignore
            batch.nbr_nids.append(nbr_nids)  # type: ignore
            batch.nbr_edge_time.append(nbr_edge_time)  # type: ignore
            batch.nbr_edge_x.append(nbr_edge_x)  # type: ignore

        batch.seed_node_nbr_mask = seed_node_nbr_mask  # type: ignore
        return batch

    def _get_seed_tensors(
        self, batch: DGBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        device = batch.edge_src.device
        seeds, seed_times = [], []
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
                    seed_node_mask[name] = torch.arange(
                        offset, offset + num_seed_nodes, device=device
                    )
                    offset += num_seed_nodes
                elif name == time_attr:
                    if (tensor < 0).any():
                        raise ValueError(
                            f'Seed times in {name} must be >= 0, got min value: {tensor.min().item()}'
                        )
                    seed_times.append(time.to(device))

        if seeds and seed_times:
            seed_nodes, seed_times = torch.cat(seeds), torch.cat(seed_times)  # type: ignore
        else:
            seed_nodes = torch.empty(0, dtype=torch.int32, device=device)
            seed_times = torch.empty(0, dtype=torch.int64, device=device)  # type: ignore
        return seed_nodes, seed_times, seed_node_mask  # type: ignore


class RecencyNeighborHook(StatefulHook):
    requires = {'edge_src', 'edge_dst', 'edge_time'}
    produces = {
        'seed_nids',
        'seed_times',
        'nbr_nids',
        'nbr_edge_time',
        'nbr_edge_x',
        'seed_node_nbr_mask',
    }

    """Load neighbors from DGraph using a recency sampling. Each node maintains a fixed number of recent neighbors.

    Args:
        num_nodes (int): Total number of nodes to track.
        num_nbrs (List[int]): Number of neighbors to sample at each hop (max neighbors to keep).
        directed (bool): If true, aggregates interactions in edge_src->edge_dst direction only (default=False).
                                               If not specified, defaults to batch edges: ['edge_src', 'edge_dst']
                                               If not specified, defaults to batch seed_times: ['time', 'time']
        seed_nodes_keys ([List[str]): List of batch attribute keys to identify the initial seed nodes to sample for.
        seed_times_keys ([List[str]): List of batch attribute keys to identify the initial seed times to sample for.

    Note:
        The order of the output tensors respect the order of seed_nodes_keys.
        For instance, for seed node keys ['edge_src', 'edge_dst', 'neg'] will have the first output index (hop 0) contain the concatenation
        of batch.edge_src, batch.edge_dst, batch.neg (in that order). The next index (hop 1) will contain first-hop neighbors of batch.edge_src
        followed by first-hop neighbors of batch.edge_dst, and then those of batch.neg. This pattern repeats for deeper hops.

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

        # Wait until first __call__ to infer the edge_x_dim on the underlying graph
        self._need_to_initialize_nbr_feats = True
        self._edge_x_dim = None
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

        batch.seed_nids, batch.seed_times = [], []  # type: ignore
        batch.nbr_nids, batch.nbr_edge_time = [], []  # type: ignore
        batch.nbr_edge_x = []  # type: ignore

        def _append_empty_hop() -> None:
            batch.seed_nids.append(torch.empty(0, dtype=torch.int32))  # type: ignore
            batch.seed_times.append(torch.empty(0, dtype=torch.int64))  # type: ignore
            batch.nbr_nids.append(torch.empty(0, dtype=torch.int32))  # type: ignore
            batch.nbr_edge_time.append(torch.empty(0, dtype=torch.int64))  # type: ignore
            batch.nbr_edge_x.append(  # type: ignore
                torch.empty(0, dg.edge_x_dim).float()  # type: ignore
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
                seed_times = batch.nbr_edge_time[hop - 1].flatten()  # type: ignore

            logger.debug(
                'Getting last %d nbrs for hop %d with %d seed nodes',
                num_nbrs,
                hop,
                seed_nodes.numel(),
            )
            nbr_nids, nbr_edge_time, nbr_edge_x = self._get_recency_neighbors(
                seed_nodes, seed_times, num_nbrs
            )

            batch.seed_nids.append(seed_nodes)  # type: ignore
            batch.seed_times.append(seed_times)  # type: ignore
            batch.nbr_nids.append(nbr_nids)  # type: ignore
            batch.nbr_edge_time.append(nbr_edge_time)  # type: ignore
            batch.nbr_edge_x.append(nbr_edge_x)  # type: ignore

        batch.seed_node_nbr_mask = seed_node_mask  # type: ignore
        if batch.edge_src.numel():
            logger.debug('Updating circular buffers')
            self._update(batch)
        return batch

    def _get_seed_tensors(
        self, batch: DGBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        device = batch.edge_src.device
        seeds, seed_times = [], []
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
                    seed_node_mask[name] = torch.arange(
                        offset, offset + num_seed_nodes
                    ).to(device)
                    offset += num_seed_nodes
                elif name == time_attr:
                    if (tensor < 0).any():
                        raise ValueError(
                            f'Seed times in {name} must be >= 0, got min value: {tensor.min().item()}'
                        )
                    seed_times.append(time.to(device))

        if seeds and seed_times:
            seed_nodes, seed_times = torch.cat(seeds), torch.cat(seed_times)  # type: ignore
        else:
            seed_nodes = torch.empty(0, dtype=torch.int32, device=device)
            seed_times = torch.empty(0, dtype=torch.int64, device=device)  # type: ignore
        return seed_nodes, seed_times, seed_node_mask  # type: ignore

    def _get_recency_neighbors(
        self, node_ids: torch.Tensor, query_times: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, ...]:
        assert self._nbr_feats is not None  # For mypy
        B = self._max_nbrs  # buffer size

        nbr_nids = self._nbr_ids[node_ids]  # (N, B)
        nbr_edge_time = self._nbr_times[node_ids]  # (N, B)
        nbr_edge_x = self._nbr_feats[node_ids]  # (N, B, edge_dim)
        write_pos = self._write_pos[node_ids]  # (N,)

        # Unroll indices to all buffers, so that last write is at index -1
        # If we had no query_time constraint, we would just take the last k entries
        candidate_idx = write_pos[:, None] - torch.arange(B, 0, -1, device=self._device)
        candidate_idx %= B  # (N, B) with oldest ... newest

        # Read the neighbor seed_times in that unrolled order, and get query_times mask
        candidate_times = torch.gather(nbr_edge_time, 1, candidate_idx)  # (N, B)
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
        out_times = torch.gather(nbr_edge_time, 1, safe_idx)
        out_feats = torch.gather(
            nbr_edge_x, 1, safe_idx.unsqueeze(-1).expand(-1, -1, self._edge_x_dim)
        )

        # Overwrite invalid positions in-place
        out_nbrs[~valid_mask] = PADDED_NODE_ID
        out_times[~valid_mask] = 0
        out_feats[~valid_mask] = 0.0

        return out_nbrs, out_times, out_feats

    def _update(self, batch: DGBatch) -> None:
        assert self._nbr_feats is not None  # For mypy
        if batch.edge_x is None:
            edge_feats = torch.zeros(
                (len(batch.edge_src), self._edge_x_dim), device=self._device
            )
        else:
            edge_feats = batch.edge_x

        if self._directed:
            node_ids, nbr_nids, seed_times = (
                batch.edge_src,
                batch.edge_dst,
                batch.edge_time,
            )
        else:
            # It's fine that seed_times is out-of-order here since we sort below
            node_ids = torch.cat([batch.edge_src, batch.edge_dst])
            nbr_nids = torch.cat([batch.edge_dst, batch.edge_src])
            seed_times = torch.cat([batch.edge_time, batch.edge_time])
            edge_feats = torch.cat([edge_feats, edge_feats])

        # Lexicographical sort by node id and time. Duplicate nodes will be adjacent.
        # Each nodes events will be sorted chronologically
        max_time = seed_times.max() + 1
        composite_key = node_ids * max_time + seed_times
        perm = torch.argsort(composite_key, stable=True)

        sorted_nodes = node_ids[perm]
        sorted_nbr_ids = nbr_nids[perm]
        sorted_times = seed_times[perm]
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
            self._edge_x_dim = dg.edge_x_dim or 0  # type: ignore
            self._nbr_feats = torch.zeros(
                (self._num_nodes, self._max_nbrs, self._edge_x_dim)  # type: ignore
            )
            self._need_to_initialize_nbr_feats = False
