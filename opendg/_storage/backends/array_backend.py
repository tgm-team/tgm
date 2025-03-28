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
        # TODO: Be aware of slice idx
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

    def get_nbrs(
        self,
        seed_nodes: Set[int],
        num_nbrs: List[int],
        slice: DGSliceTracker,
    ) -> Dict[int, List[List[Tuple[int, int]]]]:
        # TODO: Take in a sample_func to enable more than uniform sampling
        if len(num_nbrs) > 1:
            raise NotImplementedError(f'Multi-hop not implemented')

        T = Tuple[int, int]
        nbrs: Dict[int, List[Set[T]]] = {node: [set()] for node in seed_nodes}
        sampled_nbrs: Dict[int, List[List[T]]] = {node: [[]] for node in seed_nodes}
        if not len(self._events):
            return sampled_nbrs

        hop = 0
        for i in range(*self._binary_search(slice)):
            event = self._events[i]
            if isinstance(event, EdgeEvent):
                edge = event.edge
                if slice.node_slice is None or all(x in slice.node_slice for x in edge):
                    if event.src in seed_nodes:
                        nbrs[event.src][hop].add((event.dst, event.t))
                    if event.dst in seed_nodes:
                        nbrs[event.dst][hop].add((event.src, event.t))

        for node, nbrs_list in nbrs.items():
            node_nbrs = list(nbrs_list[hop])
            if num_nbrs[hop] != -1 and len(node_nbrs) > num_nbrs[hop]:
                node_nbrs = random.sample(node_nbrs, k=num_nbrs[hop])
            sampled_nbrs[node] = [node_nbrs]
        return sampled_nbrs

    def get_node_feats(self, slice: DGSliceTracker) -> Optional[Tensor]:
        # TODO: Be aware of slice idx
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
        # TODO: Be aware of slice idx
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
        return self._lb_cache[slice.start_time], self._ub_cache[slice.end_time]
