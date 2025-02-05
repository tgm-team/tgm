from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from torch import Tensor

from opendg.events import EdgeEvent, Event, NodeEvent
from opendg.timedelta import TimeDeltaDG

from ..base import DGStorageBase


class DGStorageDictBackend(DGStorageBase):
    r"""Dictionary implementation of temporal graph storage engine."""

    def __init__(self, events: List[Event], time_delta: TimeDeltaDG) -> None:
        self._node_feats_shape = self._check_node_feature_shapes(events)
        self._edge_feats_shape = self._check_edge_feature_shapes(events)

        self._events_dict: Dict[int, List[Event]] = defaultdict(list)
        for event in events:
            self._events_dict[event.time].append(event)

        self._time_delta = time_delta  # Note: will need to cache when temporal_coarsening is implemented

        # Cached Values
        self._start_time: Optional[int] = None
        self._end_time: Optional[int] = None
        self._num_nodes: Optional[int] = None
        self._num_edges: Optional[int] = None
        self._num_timestamps: Optional[int] = None

        # Lazy ops
        # TODO: There are all cache misses
        self._max_start_time: int = self.start_time  # type: ignore
        self._min_end_time: int = self.end_time  # type: ignore
        self._node_slice: Set[int] = set(range(self.num_nodes))

    def materialize(self, override_self: bool = False) -> 'DGStorageBase':
        events_dict: Dict[int, List[Event]] = {}
        for time, events in self._events_dict.items():
            if self._max_start_time <= time <= self._min_end_time:
                for e in events:
                    if isinstance(e, NodeEvent) and e.node_id in self._node_slice:
                        events_dict[time].append(e)
                    elif isinstance(e, EdgeEvent) and len(
                        set(e.edge).intersection(self._node_slice)
                    ):
                        events_dict[time].append(e)

        if override_self:
            # This refreshes the caches
            self._events_dict = events_dict
            self._max_start_time = self.start_time  # type: ignore
            self._min_end_time = self.end_time  # type: ignore
            self._node_slice = set(range(self.num_nodes))
            return self

        # TODO: Make a fast copy
        return DGStorageDictBackend(self.to_events(), self.time_delta)

    def get_nbrs(self, nodes: List[int]) -> Dict[int, List[Tuple[int, int]]]:
        # TODO: Cache this
        self.materialize(override_self=True)
        nbrs_dict = defaultdict(list)
        for t, events in self._events_dict.items():
            for event in events:
                if isinstance(event, EdgeEvent):
                    u, v = event.edge
                    if u in nodes:
                        nbrs_dict[u].append((v, t))
                    if v in nodes:
                        nbrs_dict[v].append((u, t))
        return dict(nbrs_dict)

    def to_events(self) -> List[Event]:
        self.materialize(override_self=True)
        events: List[Event] = []
        for t_events in self._events_dict.values():
            events += t_events
        return events

    def slice_time(self, start_time: int, end_time: int) -> 'DGStorageBase':
        self._check_slice_time_args(start_time, end_time)

        if start_time > self._max_start_time:
            self._max_start_time = start_time

        if end_time < self._min_end_time:
            self._min_end_time = end_time

        # TODO Cache check
        self._invalidate_cache()
        return self

    def slice_nodes(self, nodes: List[int]) -> 'DGStorageBase':
        self._node_slice &= set(nodes)

        # TODO Cache check
        self._invalidate_cache()
        return self

    def append(self, events: Union[Event, List[Event]]) -> 'DGStorageBase':
        self.materialize(override_self=True)  # TODO: Must we?

        if not isinstance(events, list):
            events = [events]

        # Check that the new events have matching feature dimension
        if len(self):
            # Node/edge feature shape must match our current feature shape
            exp_node_feats_shape = self._node_feats_shape
            exp_edge_feats_shape = self._edge_feats_shape
        else:
            # Except if our storage is empty, in which case the new event feature
            # shapes need not match previous events. This could happen if we had a
            # non-empty storage which was sliced to empty, and then appended to.
            exp_node_feats_shape = None
            exp_edge_feats_shape = None

        # We update our node/edge feature shapes in case they were previously None
        self._node_feats_shape = self._check_node_feature_shapes(
            events, expected_shape=exp_node_feats_shape
        )
        self._edge_feats_shape = self._check_edge_feature_shapes(
            events, expected_shape=exp_edge_feats_shape
        )

        for event in events:
            self._events_dict[event.time].append(event)
        self._invalidate_cache()
        return self

    def temporal_coarsening(
        self, time_delta: TimeDeltaDG, agg_func: str = 'sum'
    ) -> 'DGStorageBase':
        raise NotImplementedError('Temporal Coarsening is not implemented')

    @property
    def start_time(self) -> Optional[int]:
        if self._max_start_time is not None:
            self._start_time = self._max_start_time
        elif self._start_time is None and len(self._events_dict):
            self._start_time = min(self._events_dict)
        return self._start_time

    @property
    def end_time(self) -> Optional[int]:
        if self._min_end_time is not None:
            self._end_time = self._min_end_time
        elif self._end_time is None and len(self._events_dict):
            self._end_time = max(self._events_dict)
        return self._end_time

    @property
    def time_delta(self) -> TimeDeltaDG:
        return self._time_delta

    @property
    def num_nodes(self) -> int:
        if self._num_nodes is not None:
            # This means node_slice was initialized and we are safe to use it moving forward
            return len(self._node_slice)

        # TODO: Use lazy ops
        max_node_id = -1  # We assume the ids are >= 0
        for events in self._events_dict.values():
            for event in events:
                if isinstance(event, NodeEvent):
                    max_node_id = max(max_node_id, event.node_id)
                elif isinstance(event, EdgeEvent):
                    max_node_id = max(max_node_id, event.edge[0], event.edge[1])

        self._num_nodes = max_node_id + 1
        return self._num_nodes

    @property
    def num_edges(self) -> int:
        # TODO: Use lazy ops
        self.materialize(override_self=True)
        if self._num_edges is None:
            edges = set()
            for events in self._events_dict.values():
                for event in events:
                    if isinstance(event, EdgeEvent):
                        edges.add((event.time, event.edge))
            self._num_edges = len(edges)
        return self._num_edges

    @property
    def num_timestamps(self) -> int:
        # TODO: Use lazy ops
        self.materialize(override_self=True)
        if self._num_timestamps is None:
            self._num_timestamps = len(self._events_dict)
        return self._num_timestamps

    @property
    def node_feats(self) -> Optional[Tensor]:
        self.materialize(override_self=True)

        indices, values = [], []
        for events in self._events_dict.values():
            for event in events:
                if isinstance(event, NodeEvent) and event.features is not None:
                    indices.append([event.time, event.node_id])
                    values.append(event.features)

        if not len(values):
            return None

        # This assert is here to make mypy happy
        assert self._node_feats_shape is not None
        assert self.end_time is not None

        values_tensor = torch.stack(values)
        indices_tensor = torch.tensor(
            indices
        ).t()  # https://pytorch.org/docs/stable/sparse.html#construction

        shape = (self.end_time + 1, self.num_nodes, *self._node_feats_shape)

        return torch.sparse_coo_tensor(indices_tensor, values_tensor, shape)

    @property
    def edge_feats(self) -> Optional[Tensor]:
        self.materialize(override_self=True)

        indices, values = [], []
        for events in self._events_dict.values():
            for event in events:
                if isinstance(event, EdgeEvent) and event.features is not None:
                    indices.append([event.time, event.edge[0], event.edge[1]])
                    values.append(event.features)

        if not len(values):
            return None

        # This assert is here to make mypy happy
        assert self._edge_feats_shape is not None
        assert self.end_time is not None

        values_tensor = torch.stack(values)
        indices_tensor = torch.tensor(
            indices
        ).t()  # https://pytorch.org/docs/stable/sparse.html#construction

        shape = (
            self.end_time + 1,
            self.num_nodes,
            self.num_nodes,
            *self._edge_feats_shape,
        )
        return torch.sparse_coo_tensor(indices_tensor, values_tensor, shape)

    def _invalidate_cache(self) -> None:
        self._start_time = None
        self._end_time = None
        self._num_nodes = None
        self._num_edges = None
        self._num_timestamps = None
