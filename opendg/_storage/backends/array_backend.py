import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch import Tensor

from opendg.events import EdgeEvent, Event, NodeEvent

from ..base import DGStorageBase


class DGStorageArrayBackend(DGStorageBase):
    r"""Array backed implementation of temporal graph storage engine."""

    def __init__(self, events: List[Event]) -> None:
        self._node_feats_shape = self._check_node_feature_shapes(events)
        self._edge_feats_shape = self._check_edge_feature_shapes(events)
        self._events = self._sort_events_list_if_needed(events[:])  # Make a copy
        self._ts = np.array([event.t for event in self._events])

        # TODO: try to bypass functools lru cache restrictions on ndarrays
        self._lb_idx_cache: Dict[Optional[int], int] = {}
        self._ub_idx_cache: Dict[Optional[int], int] = {}

    def to_events(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        node_slice: Optional[Set[int]] = None,
    ) -> List[Event]:
        if not len(self._events):
            return []

        lb_idx, ub_idx = self._lb_time_idx(start_time), self._ub_time_idx(end_time)
        if node_slice is None:
            return self._events[lb_idx:ub_idx]

        events = []
        for i in range(lb_idx, ub_idx):
            event = self._events[i]
            if event.src in node_slice:
                events.append(event)
            elif isinstance(event, EdgeEvent) and event.dst in node_slice:
                events.append(event)
        return events

    def get_start_time(self, node_slice: Optional[Set[int]] = None) -> Optional[int]:
        start_time = None
        for event in self._events:
            event_nodes = (event.src,) if isinstance(event, NodeEvent) else event.edge
            if node_slice is None or len(node_slice.intersection(event_nodes)):
                if start_time is None or event.t < start_time:
                    start_time = event.t
        return start_time

    def get_end_time(self, node_slice: Optional[Set[int]] = None) -> Optional[int]:
        end_time = None
        for event in self._events:
            event_nodes = (event.src,) if isinstance(event, NodeEvent) else event.edge
            if node_slice is None or len(node_slice.intersection(event_nodes)):
                if end_time is None or event.t > end_time:
                    end_time = event.t
        return end_time

    def get_nodes(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        node_slice: Optional[Set[int]] = None,
    ) -> Set[int]:
        if not len(self._events):
            return set()

        lb_idx, ub_idx = self._lb_time_idx(start_time), self._ub_time_idx(end_time)

        nodes: Set[int] = set()
        for i in range(lb_idx, ub_idx):
            event = self._events[i]
            event_nodes = (event.src,) if isinstance(event, NodeEvent) else event.edge
            if node_slice is None or len(node_slice.intersection(event_nodes)):
                nodes.update(event_nodes)
        return nodes

    def get_num_edges(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        node_slice: Optional[Set[int]] = None,
    ) -> int:
        if not len(self._events):
            return 0

        lb_idx, ub_idx = self._lb_time_idx(start_time), self._ub_time_idx(end_time)

        edges = set()
        for i in range(lb_idx, ub_idx):
            event = self._events[i]
            if isinstance(event, EdgeEvent):
                if node_slice is None or len(node_slice.intersection(event.edge)):
                    edges.add((event.t, event.edge))
        return len(edges)

    def get_num_timestamps(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        node_slice: Optional[Set[int]] = None,
    ) -> int:
        if not len(self._events):
            return 0

        lb_idx, ub_idx = self._lb_time_idx(start_time), self._ub_time_idx(end_time)

        timestamps = set()
        for i in range(lb_idx, ub_idx):
            event = self._events[i]
            event_nodes = (event.src,) if isinstance(event, NodeEvent) else event.edge
            if node_slice is None or len(node_slice.intersection(event_nodes)):
                timestamps.add(event.t)
        return len(timestamps)

    def get_nbrs(
        self,
        seed_nodes: List[int],
        num_nbrs: List[int],
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        node_slice: Optional[Set[int]] = None,
    ) -> Dict[int, List[List[Tuple[int, int]]]]:
        # TODO: Take in a sample_func to enable more than uniform sampling
        if len(num_nbrs) > 1:
            raise NotImplementedError(
                f'Multi-hop not impemented for {self.__class__.__name__}'
            )

        seed_nodes_set = set(seed_nodes)

        nbrs: Dict[int, List[Set[Tuple[int, int]]]] = {
            node: [set()] for node in seed_nodes
        }
        sampled_nbrs: Dict[int, List[List[Tuple[int, int]]]] = {
            node: [[]] for node in seed_nodes_set
        }
        if not len(self._events):
            return sampled_nbrs

        lb_idx, ub_idx = self._lb_time_idx(start_time), self._ub_time_idx(end_time)

        hop = 0
        for i in range(lb_idx, ub_idx):
            event = self._events[i]
            if isinstance(event, EdgeEvent):
                if node_slice is None or len(node_slice.intersection(event.edge)):
                    src, dst, t = event.src, event.dst, event.t
                    if src in seed_nodes_set:
                        nbrs[src][hop].add((dst, t))
                    if dst in seed_nodes_set:
                        nbrs[dst][hop].add((src, t))

        for node, nbrs_list in nbrs.items():
            node_nbrs = list(nbrs_list[hop])
            if num_nbrs[hop] != -1 and len(node_nbrs) > num_nbrs[hop]:
                node_nbrs = random.sample(node_nbrs, k=num_nbrs[hop])
            sampled_nbrs[node] = [node_nbrs]
        return sampled_nbrs

    def get_node_feats(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        node_slice: Optional[Set[int]] = None,
    ) -> Optional[Tensor]:
        if not len(self._events):
            return None

        lb_idx, ub_idx = self._lb_time_idx(start_time), self._ub_time_idx(end_time)

        # Assuming these are both non-negative
        max_time, max_node_id = -1, -1

        indices, values = [], []
        for i in range(lb_idx, ub_idx):
            event = self._events[i]
            event_nodes = (event.src,) if isinstance(event, NodeEvent) else event.edge
            if node_slice is None or len(node_slice.intersection(event_nodes)):
                max_time = max(max_time, event.t)
                max_node_id = max(max_node_id, *event_nodes)
                if isinstance(event, NodeEvent) and event.msg is not None:
                    indices.append([event.t, event.src])
                    values.append(event.msg)

        if not len(values):
            return None

        # If the end_time is given, then it determines the dimension of the temporal axis
        # even if there are no events at the end time (could be the case after calling slice_time)
        max_time = end_time if end_time is not None else max_time

        # https://pytorch.org/docs/stable/sparse.html#construction
        values_tensor = torch.stack(values)
        indices_tensor = torch.tensor(indices).t()
        assert self._node_feats_shape is not None
        shape = (max_time + 1, max_node_id + 1, *self._node_feats_shape)

        return torch.sparse_coo_tensor(indices_tensor, values_tensor, shape)

    def get_edge_feats(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        node_slice: Optional[Set[int]] = None,
    ) -> Optional[Tensor]:
        if not len(self._events):
            return None

        lb_idx, ub_idx = self._lb_time_idx(start_time), self._ub_time_idx(end_time)

        # Assuming these are both non-negative
        max_time, max_node_id = -1, -1

        indices, values = [], []
        for i in range(lb_idx, ub_idx):
            event = self._events[i]
            event_nodes = (event.src,) if isinstance(event, NodeEvent) else event.edge
            if node_slice is None or len(node_slice.intersection(event_nodes)):
                max_time = max(max_time, event.t)
                max_node_id = max(max_node_id, *event_nodes)
                if isinstance(event, EdgeEvent) and event.msg is not None:
                    indices.append([event.t, event.src, event.dst])
                    values.append(event.msg)

        if not len(values):
            return None

        # If the end_time is given, then it determines the dimension of the temporal axis
        # even if there are no events at the end time (could be the case after calling slice_time)
        max_time = end_time if end_time is not None else max_time

        # https://pytorch.org/docs/stable/sparse.html#construction
        values_tensor = torch.stack(values)
        indices_tensor = torch.tensor(indices).t()
        assert self._edge_feats_shape is not None
        shape = (
            max_time + 1,
            max_node_id + 1,
            max_node_id + 1,
            *self._edge_feats_shape,
        )

        return torch.sparse_coo_tensor(indices_tensor, values_tensor, shape)

    def _lb_time_idx(self, t: Optional[int]) -> int:
        if t not in self._lb_idx_cache:
            tt = self._ts[0] if t is None else t
            self._lb_idx_cache[t] = int(np.searchsorted(self._ts, tt))
        return self._lb_idx_cache[t]

    def _ub_time_idx(self, t: Optional[int]) -> int:
        if t not in self._ub_idx_cache:
            tt = self._ts[-1] if t is None else t
            self._ub_idx_cache[t] = int(np.searchsorted(self._ts, tt, side='right'))
        return self._ub_idx_cache[t]
