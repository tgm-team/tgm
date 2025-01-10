from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from opendg.events import EdgeEvent, Event, NodeEvent
from opendg.typing import TimeDelta

from ..base import DGStorageBase


class DGStorageDictBackend(DGStorageBase):
    r"""Dictionary implementation of temporal graph storage engine."""

    def __init__(self, events: List[Event]) -> None:
        self._events_dict: Dict[int, List[Event]] = defaultdict(list)
        for event in events:
            self._events_dict[event.time].append(event)

        # Cached Values
        self._start_time: Optional[int] = None
        self._end_time: Optional[int] = None
        self._num_nodes: Optional[int] = None
        self._num_edges: Optional[int] = None
        self._num_timestamps: Optional[int] = None
        self._time_granularity: Optional[TimeDelta] = None

    def to_events(self) -> List[Event]:
        events: List[Event] = []
        for t_events in self._events_dict.values():
            events += t_events
        return events

    def slice_time(self, start_time: int, end_time: int) -> 'DGStorageBase':
        self._check_slice_time_args(start_time, end_time)
        self._invalidate_cache()
        self._events_dict = {
            k: v for k, v in self._events_dict.items() if start_time <= k < end_time
        }
        return self

    def slice_nodes(self, nodes: List[int]) -> 'DGStorageBase':
        self._invalidate_cache()

        events_dict: Dict[int, List[Event]] = defaultdict(list)
        for t, events in self._events_dict.items():
            for event in events:
                if isinstance(event, NodeEvent) and event.node_id in nodes:
                    events_dict[t].append(event)
                elif isinstance(event, EdgeEvent) and len(
                    set(event.edge).intersection(nodes)
                ):
                    events_dict[t].append(event)
        self._events_dict = events_dict
        return self

    def get_nbrs(self, nodes: List[int]) -> Dict[int, List[Tuple[int, int]]]:
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

    def append(self, events: Union[Event, List[Event]]) -> 'DGStorageBase':
        self._invalidate_cache()
        if not isinstance(events, list):
            events = [events]
        for event in events:
            self._events_dict[event.time].append(event)
        return self

    def temporal_coarsening(
        self, time_delta: TimeDelta, agg_func: str = 'sum'
    ) -> 'DGStorageBase':
        self._check_temporal_coarsening_args(time_delta, agg_func)

        # This assert is here to make mypy happy
        assert self.start_time is not None
        assert self.end_time is not None
        total_time = self.end_time - self.start_time

        # TODO: Support other time_delta formats
        interval_size = total_time // time_delta

        events_dict = defaultdict(list)
        for t, event in self._events_dict.items():
            bin_t = -((t - self.start_time) // -interval_size)  # Ceiling division

            # TODO: Use the agg_func
            events_dict[bin_t].extend(event)

        self._events_dict = events_dict
        self._invalidate_cache()
        return self

    @property
    def start_time(self) -> Optional[int]:
        if self._start_time is None and len(self._events_dict):
            self._start_time = min(self._events_dict)
        return self._start_time

    @property
    def end_time(self) -> Optional[int]:
        if self._end_time is None and len(self._events_dict):
            self._end_time = max(self._events_dict)
        return self._end_time

    @property
    def time_granularity(self) -> Optional[TimeDelta]:
        if self._time_granularity is None and len(self) > 1:
            # TODO: Validate and construct a proper TimeDelta object
            ts = list(self._events_dict.keys())
            self._time_granularity = min(
                [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
            )
        return self._time_granularity

    @property
    def num_nodes(self) -> int:
        if self._num_nodes is None:
            nodes = set()
            for events in self._events_dict.values():
                for event in events:
                    if isinstance(event, NodeEvent):
                        nodes.add(event.node_id)
                    elif isinstance(event, EdgeEvent):
                        nodes.update(event.edge)
            self._num_nodes = len(nodes)
        return self._num_nodes

    @property
    def num_edges(self) -> int:
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
        if self._num_timestamps is None:
            self._num_timestamps = len(self._events_dict)
        return self._num_timestamps

    @property
    def node_feats(self) -> Optional[Tensor]:
        feats = []
        for events in self._events_dict.values():
            for event in events:
                if isinstance(event, NodeEvent) and event.features is not None:
                    feats.append(event.features)

        if not len(feats):
            return None

        return torch.cat(feats)

    @property
    def edge_feats(self) -> Optional[Tensor]:
        feats = []
        for events in self._events_dict.values():
            for event in events:
                if isinstance(event, EdgeEvent) and event.features is not None:
                    feats.append(event.features)

        if not len(feats):
            return None

        return torch.cat(feats)

    def _invalidate_cache(self) -> None:
        self._start_time = None
        self._end_time = None
        self._num_nodes = None
        self._num_edges = None
        self._num_timestamps = None
        self._time_granularity = None
