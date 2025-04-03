import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch import Tensor

from opendg.events import EdgeEvent, Event, NodeEvent

from ..base import DGSliceTracker, DGStorageBase


class DGStorageArrayBackend(DGStorageBase):
    r"""Array backed implementation of temporal graph storage engine."""

    def __init__(self, events: List[Event]) -> None:
        if not len(events):
            raise ValueError(f'Tried to init {self.__class__.__name__} with empty data')
        self._events = self._sort_events_list_if_needed(events[:])  # Make a copy
        self._node_feats_dim, self._edge_feats_dim = self._check_feature_dims(events)
        self._ts = np.array([event.t for event in self._events])

        # Binary search caches for finding timestamps in event array
        self._lb_cache: Dict[Optional[int], int] = {}
        self._ub_cache: Dict[Optional[int], int] = {}

    def to_events(self, slice: DGSliceTracker) -> List[Event]:
        lb_idx, ub_idx = self._binary_search(slice)
        if slice.node_slice is None:
            return self._events[lb_idx:ub_idx]

        events = []
        for i in range(lb_idx, ub_idx):
            event = self._events[i]
            if any(x in slice.node_slice for x in self._nodes_in_event(event)):
                events.append(event)
        return events

    def get_start_time(self, slice: DGSliceTracker) -> Optional[int]:
        for i in range(*self._binary_search(slice)):
            event = self._events[i]
            nodes = self._nodes_in_event(event)
            if slice.node_slice is None or any(x in slice.node_slice for x in nodes):
                return event.t
        return None

    def get_end_time(self, slice: DGSliceTracker) -> Optional[int]:
        lb_idx, ub_idx = self._binary_search(slice)
        for i in range(ub_idx - 1, lb_idx - 1, -1):
            event = self._events[i]
            nodes = self._nodes_in_event(event)
            if slice.node_slice is None or any(x in slice.node_slice for x in nodes):
                return event.t
        return None

    def get_nodes(self, slice: DGSliceTracker) -> Set[int]:
        all_nodes: Set[int] = set()
        for i in range(*self._binary_search(slice)):
            event = self._events[i]
            nodes = self._nodes_in_event(event)
            if slice.node_slice is None or any(x in slice.node_slice for x in nodes):
                all_nodes.update(nodes)
        return all_nodes

    def get_edges(self, slice: DGSliceTracker) -> Tuple[Tensor, Tensor, Tensor]:
        src, dst, time = [], [], []
        for i in range(*self._binary_search(slice)):
            event = self._events[i]
            if isinstance(event, EdgeEvent):
                edge = event.edge
                if slice.node_slice is None or any(x in slice.node_slice for x in edge):
                    src.append(event.src)
                    dst.append(event.dst)
                    time.append(event.t)

        src_tensor = torch.LongTensor(src)
        dst_tensor = torch.LongTensor(dst)
        time_tensor = torch.LongTensor(time)
        return src_tensor, dst_tensor, time_tensor

    def get_num_timestamps(self, slice: DGSliceTracker) -> int:
        timestamps = set()
        for i in range(*self._binary_search(slice)):
            event = self._events[i]
            nodes = self._nodes_in_event(event)
            if slice.node_slice is None or any(x in slice.node_slice for x in nodes):
                timestamps.add(event.t)
        return len(timestamps)

    def get_num_events(self, slice: DGSliceTracker) -> int:
        num_events = 0
        for i in range(*self._binary_search(slice)):
            event = self._events[i]
            nodes = self._nodes_in_event(event)
            if slice.node_slice is None or any(x in slice.node_slice for x in nodes):
                num_events += 1
        return num_events

    def get_nbrs(
        self,
        seed_nodes: Tensor,
        num_nbrs: List[int],
        slice: DGSliceTracker,
    ) -> Tuple[List[Tensor], ...]:
        # TODO: Take in a sample_func to enable more than uniform sampling
        if len(num_nbrs) > 1:
            raise NotImplementedError(f'Multi-hop not implemented')
        n_nbrs = num_nbrs[0]
        unique, inverse_indices = seed_nodes.unique(return_inverse=True)
        seed_nodes_set = set(unique.tolist())

        nbrs: Dict[int, Set[Tuple[int, int]]] = {node: set() for node in seed_nodes_set}
        for i in range(*self._binary_search(slice)):
            event = self._events[i]
            if isinstance(event, EdgeEvent):
                edge = event.edge
                if slice.node_slice is None or all(x in slice.node_slice for x in edge):
                    # Use 0/1 flag to denote dst/src neighbor, respectively
                    if event.src in seed_nodes_set:
                        nbrs[event.src].add((i, 1))
                    if event.dst in seed_nodes_set:
                        nbrs[event.dst].add((i, 0))

        # TODO: Node feats
        batch_size = len(seed_nodes)
        nbr_nids = torch.empty(batch_size, n_nbrs, dtype=torch.long)
        nbr_times = torch.empty(batch_size, n_nbrs, dtype=torch.long)
        nbr_feats = torch.zeros(batch_size, n_nbrs, self._edge_feats_dim)  # type: ignore
        nbr_mask = torch.zeros(batch_size, n_nbrs, dtype=torch.long)
        for i, nbrs_set in enumerate(nbrs.values()):
            node_nbrs = list(nbrs_set)
            if n_nbrs != -1 and len(node_nbrs) > n_nbrs:
                node_nbrs = random.sample(node_nbrs, k=n_nbrs)

            nbr_nids_, nbr_times_, nbr_feats_ = [], [], []
            for event_idx, edge_idx in node_nbrs:
                nbr_nids_.append(self._events[event_idx].edge[edge_idx])  # type: ignore
                nbr_times_.append(self._events[event_idx].t)
                if self._events[event_idx].features is not None:
                    nbr_feats_.append(self._events[event_idx].features)
                else:
                    raise NotImplementedError('Some edges do not have features')

            nn = len(node_nbrs)
            mask = inverse_indices == i
            nbr_nids[mask, :nn] = torch.tensor(nbr_nids_)
            nbr_times[mask, :nn] = torch.tensor(nbr_times_)
            nbr_feats[mask, :nn] = torch.stack(nbr_feats_)  # type: ignore
            nbr_mask[mask, :nn] = 1
        return [seed_nodes], [nbr_nids], [nbr_times], [nbr_feats], [nbr_mask]

    def get_node_feats(self, slice: DGSliceTracker) -> Optional[Tensor]:
        max_time, max_node_id = -1, -1  # Assuming these are both non-negative
        indices, values = [], []
        for i in range(*self._binary_search(slice)):
            event = self._events[i]
            nodes = self._nodes_in_event(event)
            if slice.node_slice is None or any(x in slice.node_slice for x in nodes):
                max_time = max(max_time, event.t)
                max_node_id = max(max_node_id, *nodes)
                if isinstance(event, NodeEvent) and event.features is not None:
                    indices.append([event.t, event.src])
                    values.append(event.features)

        if not len(values):
            return None

        # If the end_time is given, then it determines the dimension of the temporal axis
        # even if there are no events at the end time (could be the case after calling slice_time)
        max_time = slice.end_time or max_time
        assert self._node_feats_dim is not None

        # https://pytorch.org/docs/stable/sparse.html#construction
        values_tensor = torch.stack(values)
        indices_tensor = torch.tensor(indices).t()
        shape = (max_time + 1, max_node_id + 1, self._node_feats_dim)
        return torch.sparse_coo_tensor(indices_tensor, values_tensor, shape)

    def get_edge_feats(self, slice: DGSliceTracker) -> Optional[Tensor]:
        max_time, max_node_id = -1, -1  # Assuming these are both non-negative
        indices, values = [], []
        for i in range(*self._binary_search(slice)):
            event = self._events[i]
            nodes = self._nodes_in_event(event)
            if slice.node_slice is None or any(x in slice.node_slice for x in nodes):
                max_time = max(max_time, event.t)
                max_node_id = max(max_node_id, *nodes)
                if isinstance(event, EdgeEvent) and event.features is not None:
                    indices.append([event.t, event.src, event.dst])
                    values.append(event.features)

        if not len(values):
            return None

        # If the end_time is given, then it determines the dimension of the temporal axis
        # even if there are no events at the end time (could be the case after calling slice_time)
        max_time = slice.end_time or max_time
        assert self._edge_feats_dim is not None

        # https://pytorch.org/docs/stable/sparse.html#construction
        values_tensor = torch.stack(values)
        indices_tensor = torch.tensor(indices).t()
        shape = (max_time + 1, max_node_id + 1, max_node_id + 1, self._edge_feats_dim)
        return torch.sparse_coo_tensor(indices_tensor, values_tensor, shape)

    def get_node_feats_dim(self) -> Optional[int]:
        return self._node_feats_dim

    def get_edge_feats_dim(self) -> Optional[int]:
        return self._edge_feats_dim

    @staticmethod
    def _nodes_in_event(event: Event) -> Tuple[int, ...]:
        return (event.src,) if isinstance(event, NodeEvent) else event.edge

    def _binary_search(self, slice: DGSliceTracker) -> Tuple[int, int]:
        if slice.start_time not in self._lb_cache:
            t = self._ts[0] if slice.start_time is None else slice.start_time
            self._lb_cache[slice.start_time] = int(np.searchsorted(self._ts, t))
        if slice.end_time not in self._ub_cache:
            t = self._ts[-1] if slice.end_time is None else slice.end_time
            self._ub_cache[slice.end_time] = int(
                np.searchsorted(self._ts, t, side='right')
            )
        lb = self._lb_cache[slice.start_time]
        ub = self._ub_cache[slice.end_time]

        # Additional clamping on possible index constraints
        clamp = lambda x, lo, hi: lo if x < lo else hi if x > hi else x
        lb = clamp(lb, slice.start_idx or 0, slice.end_idx or len(self._ts))
        ub = clamp(ub, slice.start_idx or 0, slice.end_idx or len(self._ts))
        return lb, ub
