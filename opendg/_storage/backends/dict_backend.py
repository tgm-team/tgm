from collections import defaultdict
from typing import Dict, List, Optional, Set, Union

import torch
from torch import Tensor

from opendg.events import EdgeEvent, Event, NodeEvent
from opendg.timedelta import TimeDeltaDG

from ..base import DGStorageBase


class DGStorageDictBackend(DGStorageBase):
    r"""Dictionary implementation of temporal graph storage engine."""

    def __init__(self, events: List[Event]) -> None:
        self._node_feats_shape = self._check_node_feature_shapes(events)
        self._edge_feats_shape = self._check_edge_feature_shapes(events)

        self._events_dict: Dict[int, List[Event]] = defaultdict(list)
        for event in events:
            self._events_dict[event.time].append(event)

    def to_events(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        node_slice: Optional[Set[int]] = None,
    ) -> List[Event]:
        events: List[Event] = []
        for t, t_events in self._events_dict.items():
            for event in t_events:
                if self._valid_slice(t, event, start_time, end_time, node_slice):
                    events.append(event)
        return events

    def get_start_time(self, node_slice: Optional[Set[int]] = None) -> Optional[int]:
        start_time = None
        for t, t_events in self._events_dict.items():
            for event in t_events:
                if self._valid_slice(t, event, node_slice=node_slice):
                    if start_time is None or t < start_time:
                        start_time = t
                    break
        return start_time

    def get_end_time(self, node_slice: Optional[Set[int]] = None) -> Optional[int]:
        end_time = None
        for t, t_events in self._events_dict.items():
            for event in t_events:
                if self._valid_slice(t, event, node_slice=node_slice):
                    if end_time is None or t > end_time:
                        end_time = t
                    break
        return end_time

    def get_nodes(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> Set[int]:
        nodes = set()
        for t, t_events in self._events_dict.items():
            for event in t_events:
                if self._valid_slice(
                    t, event, start_time=start_time, end_time=end_time
                ):
                    if isinstance(event, NodeEvent):
                        nodes.add(event.node_id)
                    elif isinstance(event, EdgeEvent):
                        nodes.union(event.edge)
        return nodes

    def get_num_edges(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        node_slice: Optional[Set[int]] = None,
    ) -> int:
        edges = set()
        for t, t_events in self._events_dict.items():
            for event in t_events:
                if isinstance(event, EdgeEvent) and self._valid_slice(
                    t,
                    event,
                    start_time=start_time,
                    end_time=end_time,
                    node_slice=node_slice,
                ):
                    edges.add((event.time, event.edge))
        return len(edges)

    def get_num_timestamps(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        node_slice: Optional[Set[int]] = None,
    ) -> int:
        timestamps = set()
        for t, t_events in self._events_dict.items():
            for event in t_events:
                if self._valid_slice(
                    t,
                    event,
                    start_time=start_time,
                    end_time=end_time,
                    node_slice=node_slice,
                ):
                    timestamps.add(t)
                    break
        return len(timestamps)

    def append(self, events: Union[Event, List[Event]]) -> 'DGStorageBase':
        if not isinstance(events, list):
            events = [events]

        # Check that the new events have matching feature dimension
        if len(self._events_dict):
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
        return self

    def temporal_coarsening(
        self, time_delta: TimeDeltaDG, agg_func: str = 'sum'
    ) -> 'DGStorageBase':
        raise NotImplementedError('Temporal Coarsening is not implemented')

    def get_node_feats(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        node_slice: Optional[Set[int]] = None,
    ) -> Optional[Tensor]:
        # Assuming these are both non-negative
        max_time, max_node_id = -1, -1

        indices, values = [], []
        for t, t_events in self._events_dict.items():
            for event in t_events:
                if isinstance(event, NodeEvent) and self._valid_slice(
                    t,
                    event,
                    start_time=start_time,
                    end_time=end_time,
                    node_slice=node_slice,
                ):
                    indices.append([event.time, event.node_id])
                    values.append(event.features)

                    max_time = max(max_time, t)
                    max_node_id = max(max_node_id, event.node_id)

        if not len(values):
            return None

        values_tensor = torch.stack(values)
        indices_tensor = torch.tensor(
            indices
        ).t()  # https://pytorch.org/docs/stable/sparse.html#construction

        assert self._node_feats_shape is not None
        shape = (max_time + 1, max_node_id + 1, *self._node_feats_shape)

        return torch.sparse_coo_tensor(indices_tensor, values_tensor, shape)

    def get_edge_feats(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        node_slice: Optional[Set[int]] = None,
    ) -> Optional[Tensor]:
        # Assuming these are both non-negative
        max_time, max_node_id = -1, -1

        indices, values = [], []
        for t, t_events in self._events_dict.items():
            for event in t_events:
                if isinstance(event, EdgeEvent) and self._valid_slice(
                    t,
                    event,
                    start_time=start_time,
                    end_time=end_time,
                    node_slice=node_slice,
                ):
                    indices.append([event.time, event.edge[0], event.edge[1]])
                    values.append(event.features)

                    max_time = max(max_time, t)
                    max_node_id = max(max_node_id, event.edge[0], event.edge[1])

        if not len(values):
            return None

        values_tensor = torch.stack(values)
        indices_tensor = torch.tensor(
            indices
        ).t()  # https://pytorch.org/docs/stable/sparse.html#construction

        assert self._edge_feats_shape is not None

        shape = (
            max_time + 1,
            max_node_id + 1,
            max_node_id + 1,
            *self._edge_feats_shape,
        )

        return torch.sparse_coo_tensor(indices_tensor, values_tensor, shape)

    def _valid_slice(
        self,
        time: int,
        event: Event,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        node_slice: Optional[Set[int]] = None,
    ) -> bool:
        lb_time = float('-inf') if start_time is None else start_time
        ub_time = float('inf') if end_time is None else end_time

        time_valid = lb_time <= time < ub_time
        node_valid = (
            node_slice is None
            or (isinstance(event, NodeEvent) and event.node_id in node_slice)
            or (
                isinstance(event, EdgeEvent)
                and len(set(event.edge).intersection(node_slice)) > 0
            )
        )
        # TODO: This can be optimized by returning these seperately, and hence early
        # returning out of the event loop if we already know the timestamp is not valid
        return time_valid and node_valid
