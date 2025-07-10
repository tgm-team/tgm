import random
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import torch
from torch import Tensor

from tgm.data import DGData
from tgm.timedelta import TimeDeltaDG

from ..base import DGSliceTracker, DGStorageBase


class DGStorageArrayBackend(DGStorageBase):
    r"""Array backed implementation of temporal graph storage engine."""

    def __init__(self, data: DGData) -> None:
        self._data = data

        # Binary search caches for finding timestamps in event array
        self._lb_cache: Dict[Optional[int], int] = {}
        self._ub_cache: Dict[Optional[int], int] = {}

    def discretize(
        self,
        old_time_granularity: TimeDeltaDG,
        new_time_granularity: TimeDeltaDG,
        reduce_op: Literal['first'],
    ) -> 'DGStorageBase':
        # TODO: Vectorize this as a groupby([bucket, edge_ids].reduce(reduce_op))

        # We can assume that new granularity is coarser than old time granularity
        # since we checked args in the caller.
        time_factor = old_time_granularity.convert(new_time_granularity)
        buckets = (self._data.timestamps.float() * time_factor).floor().long()

        if reduce_op != 'first':
            raise NotImplementedError(f'No reduction implemented for op: {reduce_op}')

        def _get_keep_indices(event_idx: Tensor, ids: Tensor) -> Tensor:
            event_buckets = buckets[event_idx]
            seen: Set[Any] = set()
            keep, prev_bucket = [], None

            for i in range(event_idx.numel()):
                bucket = int(event_buckets[i])
                node_or_edge = tuple(ids[i].tolist()) if ids.ndim > 1 else int(ids[i])
                if bucket != prev_bucket:
                    seen.clear()
                    prev_bucket = bucket
                if node_or_edge not in seen:
                    seen.add(node_or_edge)
                    keep.append(i)
            return torch.tensor(keep, dtype=torch.long)

        # Edge events
        edge_mask = _get_keep_indices(self._data.edge_event_idx, self._data.edge_index)
        edge_timestamps = buckets[self._data.edge_event_idx][edge_mask]
        edge_index = self._data.edge_index[edge_mask]
        edge_feats = None
        if self._data.edge_feats is not None:
            edge_feats = self._data.edge_feats[edge_mask]

        # Node events
        node_timestamps, node_ids, dynamic_node_feats = None, None, None
        if self._data.node_event_idx is not None:
            node_mask = _get_keep_indices(
                self._data.node_event_idx,
                self._data.node_ids,  # type: ignore
            )
            node_timestamps = buckets[self._data.node_event_idx][node_mask]
            node_ids = self._data.node_ids[node_mask]  # type: ignore
            dynamic_node_feats = None
            if self._data.dynamic_node_feats is not None:
                dynamic_node_feats = self._data.dynamic_node_feats[node_mask]

        static_node_feats = None
        if self._data.static_node_feats is not None:  # Need a deep copy
            static_node_feats = self._data.static_node_feats.clone()

        new_data = DGData.from_raw(
            edge_timestamps=edge_timestamps,
            edge_index=edge_index,
            edge_feats=edge_feats,
            node_timestamps=node_timestamps,
            node_ids=node_ids,
            dynamic_node_feats=dynamic_node_feats,
            static_node_feats=static_node_feats,
        )
        return DGStorageArrayBackend(new_data)

    def get_start_time(self, slice: DGSliceTracker) -> Optional[int]:
        lb_idx, ub_idx = self._binary_search(slice)
        if lb_idx >= ub_idx:
            return None
        return int(self._data.timestamps[lb_idx].item())

    def get_end_time(self, slice: DGSliceTracker) -> Optional[int]:
        lb_idx, ub_idx = self._binary_search(slice)
        if lb_idx >= ub_idx:
            return None
        return int(self._data.timestamps[ub_idx - 1].item())

    def get_nodes(self, slice: DGSliceTracker) -> Set[int]:
        all_nodes: Set[int] = set()
        lb_idx, ub_idx = self._binary_search(slice)

        edge_mask = (self._data.edge_event_idx >= lb_idx) & (
            self._data.edge_event_idx < ub_idx
        )
        edge_event_nodes = self._data.edge_index[edge_mask].unique().tolist()
        all_nodes.update(edge_event_nodes)

        if self._data.node_event_idx is not None:
            node_mask = (self._data.node_event_idx >= lb_idx) & (
                self._data.node_event_idx < ub_idx
            )
            node_event_nodes = self._data.node_ids[node_mask].unique().tolist()  # type: ignore
            all_nodes.update(node_event_nodes)
        return all_nodes

    def get_edges(self, slice: DGSliceTracker) -> Tuple[Tensor, Tensor, Tensor]:
        lb_idx, ub_idx = self._binary_search(slice)
        edge_mask = (self._data.edge_event_idx >= lb_idx) & (
            self._data.edge_event_idx < ub_idx
        )
        edges = self._data.edge_index[edge_mask]
        src, dst = edges[:, 0], edges[:, 1]
        time = self._data.timestamps[self._data.edge_event_idx[edge_mask]]

        src, dst, time = src.contiguous(), dst.contiguous(), time.contiguous()
        return src, dst, time

    def get_num_timestamps(self, slice: DGSliceTracker) -> int:
        lb_idx, ub_idx = self._binary_search(slice)
        return len(self._data.timestamps[lb_idx:ub_idx].unique())

    def get_num_events(self, slice: DGSliceTracker) -> int:
        lb_idx, ub_idx = self._binary_search(slice)
        return ub_idx - lb_idx

    def get_nbrs(
        self,
        seed_nodes: Tensor,
        num_nbrs: int,
        slice: DGSliceTracker,
    ) -> Tuple[Tensor, ...]:
        # TODO: Take in a sample_func to enable more than uniform sampling
        device = seed_nodes.device
        unique_nodes, inverse_indices = seed_nodes.unique(return_inverse=True)
        seed_nodes_set = set(unique_nodes.tolist())

        lb_idx, ub_idx = self._binary_search(slice)
        edge_mask = (self._data.edge_event_idx >= lb_idx) & (
            self._data.edge_event_idx < ub_idx
        )
        edges = self._data.edge_index[edge_mask]
        event_ids = self._data.edge_event_idx[edge_mask]

        src_list = edges[:, 0].tolist()
        dst_list = edges[:, 1].tolist()
        eid_list = event_ids.tolist()

        # This is a loop over the entire graph up to (not including) the current batch end time
        # which results in quadratic cost for a single epoch. Consider raises a warning to let the
        # user know that this is not the right backend for this.
        nbrs: Dict[int, List[Tuple[int, int]]] = {node: [] for node in seed_nodes_set}
        for s, d, i in zip(src_list, dst_list, eid_list):
            if s in nbrs:
                nbrs[s].append((i, d))
            if d in nbrs:
                nbrs[d].append((i, s))

        B = len(seed_nodes)
        nbr_nids = torch.full((B, num_nbrs), -1, dtype=torch.long, device=device)
        nbr_times = torch.zeros(B, num_nbrs, dtype=torch.long, device=device)
        nbr_feats = torch.zeros(B, num_nbrs, self.get_edge_feats_dim(), device=device)  # type: ignore
        nbr_mask = torch.zeros(B, num_nbrs, dtype=torch.long, device=device)

        for i, node in enumerate(unique_nodes.tolist()):
            node_nbrs = nbrs[node]
            if not len(node_nbrs):
                continue

            # Subsample if we have more neighbours than was queried
            if num_nbrs != -1 and len(node_nbrs) > num_nbrs:
                node_nbrs = random.sample(node_nbrs, k=num_nbrs)

            nbr_ids, times, feats = [], [], []
            for eid, nbr_id in node_nbrs:
                nbr_ids.append(nbr_id)
                times.append(self._data.timestamps[eid])
                if self._data.edge_feats is not None:
                    feats.append(self._data.edge_feats[eid])

            nn = len(nbr_ids)
            mask = inverse_indices == i
            nbr_nids[mask, :nn] = torch.tensor(nbr_ids, dtype=torch.long, device=device)
            nbr_times[mask, :nn] = torch.tensor(times, dtype=torch.long, device=device)
            if self._data.edge_feats is not None:
                nbr_feats[mask, :nn] = torch.stack(feats).to(device).float()
            nbr_mask[mask, :nn] = 1

        return nbr_nids, nbr_times, nbr_feats, nbr_mask

    def get_static_node_feats(self) -> Optional[Tensor]:
        return self._data.static_node_feats

    def get_dynamic_node_feats(self, slice: DGSliceTracker) -> Optional[Tensor]:
        if self._data.dynamic_node_feats is None:
            return None
        assert self._data.node_event_idx is not None  # for mypy
        assert self._data.node_ids is not None  # for mypy

        lb_idx, ub_idx = self._binary_search(slice)
        node_mask = (self._data.node_event_idx >= lb_idx) & (
            self._data.node_event_idx < ub_idx
        )
        if node_mask.sum() == 0:
            return None

        time = self._data.timestamps[self._data.node_event_idx[node_mask]]
        nodes = self._data.node_ids[node_mask]
        indices = torch.stack([time, nodes], dim=0)
        values = self._data.dynamic_node_feats[node_mask]

        max_node_id = nodes.max()
        edge_mask = (self._data.edge_event_idx >= lb_idx) & (
            self._data.edge_event_idx < ub_idx
        )
        if edge_mask.sum() != 0 and len(self._data.edge_index[edge_mask]):
            max_node_id = max(max_node_id, self._data.edge_index[edge_mask].max())  # type: ignore

        max_time = slice.end_time or self._data.timestamps[ub_idx - 1]
        node_feats_dim = self.get_dynamic_node_feats_dim()
        shape = (max_time + 1, max_node_id + 1, node_feats_dim)
        return torch.sparse_coo_tensor(indices, values, shape)  # type: ignore

    def get_edge_feats(self, slice: DGSliceTracker) -> Optional[Tensor]:
        if self._data.edge_feats is None:
            return None

        lb_idx, ub_idx = self._binary_search(slice)
        edge_mask = (self._data.edge_event_idx >= lb_idx) & (
            self._data.edge_event_idx < ub_idx
        )
        if edge_mask.sum() == 0:
            return None

        time = self._data.timestamps[self._data.edge_event_idx[edge_mask]]
        edges = self._data.edge_index[edge_mask]
        indices = torch.stack([time, edges[:, 0], edges[:, 1]], dim=0)
        values = self._data.edge_feats[edge_mask]

        max_node_id = edges.max()
        if self._data.node_event_idx is not None:
            node_mask = (self._data.node_event_idx >= lb_idx) & (
                self._data.node_event_idx < ub_idx
            )
            if len(self._data.node_ids[node_mask]):  # type: ignore
                max_node_id = max(max_node_id, self._data.node_ids[node_mask].max())  # type: ignore

        max_time = slice.end_time or self._data.timestamps[ub_idx - 1]
        edge_feats_dim = self.get_edge_feats_dim()
        shape = (max_time + 1, max_node_id + 1, max_node_id + 1, edge_feats_dim)
        return torch.sparse_coo_tensor(indices, values, shape)  # type: ignore

    def get_static_node_feats_dim(self) -> Optional[int]:
        if self._data.static_node_feats is None:
            return None
        return self._data.static_node_feats.shape[1]

    def get_dynamic_node_feats_dim(self) -> Optional[int]:
        if self._data.dynamic_node_feats is None:
            return None
        return self._data.dynamic_node_feats.shape[1]

    def get_edge_feats_dim(self) -> Optional[int]:
        if self._data.edge_feats is None:
            return None
        return self._data.edge_feats.shape[1]

    def _binary_search(self, slice: DGSliceTracker) -> Tuple[int, int]:
        ts = self._data.timestamps
        if slice.start_time not in self._lb_cache:
            t = ts[0] if slice.start_time is None else slice.start_time
            self._lb_cache[slice.start_time] = int(torch.searchsorted(ts, t))
        if slice.end_time not in self._ub_cache:
            t = ts[-1] if slice.end_time is None else slice.end_time
            self._ub_cache[slice.end_time] = int(
                torch.searchsorted(ts, t, side='right')
            )
        lb = self._lb_cache[slice.start_time]
        ub = self._ub_cache[slice.end_time]

        # Additional clamping on possible index constraints
        clamp = lambda x, lo, hi: max(lo, min(hi, x))
        lb = clamp(lb, slice.start_idx or 0, slice.end_idx or len(ts))
        ub = clamp(ub, slice.start_idx or 0, slice.end_idx or len(ts))
        return lb, ub
