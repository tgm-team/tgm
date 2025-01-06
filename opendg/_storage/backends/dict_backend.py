from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

from opendg.typing import Event, EventsDict

from ..base import DGStorageBase


class DGStorageDictBackend(DGStorageBase):
    r"""Dictionary implementation of temporal graph storage engine."""

    def __init__(self, events_dict: EventsDict) -> None:
        self._events_dict: EventsDict = events_dict

        # Cached Values
        self._start_time: Optional[int] = None
        self._end_time: Optional[int] = None
        self._num_nodes: Optional[int] = None
        self._num_edges: Optional[int] = None
        self._num_timestamps: Optional[int] = None

    @classmethod
    def from_events(cls, events: List[Event]) -> 'DGStorageBase':
        events_dict = defaultdict(list)
        for t, u, v in events:
            events_dict[t].append((u, v))
        return cls(events_dict)

    def to_events(self) -> List[Event]:
        events: List[Event] = []
        for t, edges in self._events_dict.items():
            for u, v in edges:
                events.append((t, u, v))
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

        # TODO: Fix events
        self._events_dict = {
            k: v
            for k, v in self._events_dict.items()
            if len(set(v).intersection(nodes))
        }
        return self

    def get_nbrs(self, nodes: List[int]) -> Dict[int, List[Tuple[int, int]]]:
        nbrs_dict = defaultdict(list)
        for t, edges in self._events_dict.items():
            for u, v in edges:
                if u in nodes:
                    nbrs_dict[u].append((v, t))
                if v in nodes:
                    nbrs_dict[v].append((u, t))
        return dict(nbrs_dict)

    def update(self, events: Union[Event, List[Event]]) -> 'DGStorageBase':
        self._invalidate_cache()
        if not isinstance(events, list):
            events = [events]
        for t, u, v in events:
            self._events_dict[t].append((u, v))
        return self

    def temporal_coarsening(
        self, time_delta: int, agg_func: str = 'sum'
    ) -> 'DGStorageBase':
        self._check_temporal_coarsening_args(time_delta, agg_func)

        # This assert is here to make mypy happy
        assert self.start_time is not None
        assert self.end_time is not None
        total_time = self.end_time - self.start_time

        # TODO: Support other time_delta formats
        interval_size = total_time // time_delta

        events_dict = defaultdict(list)
        for t, edges in self._events_dict.items():
            for u, v in edges:
                bin_t = -((t - self.start_time) // -interval_size)  # Ceiling division
                events_dict[bin_t].append((u, v))

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
    def num_nodes(self) -> int:
        # TODO: Fix events
        if self._num_nodes is None:
            self._num_nodes = len(set(sum(map(list, self._events_dict.values()), [])))
        return self._num_nodes

    @property
    def num_edges(self) -> int:
        # TODO: Fix events
        if self._num_edges is None:
            self._num_edges = len(set(self._events_dict.values()))
        return self._num_edges

    @property
    def num_timestamps(self) -> int:
        if self._num_timestamps is None:
            self._num_timestamps = len(self._events_dict)
        return self._num_timestamps

    def _invalidate_cache(self) -> None:
        self._start_time = None
        self._end_time = None
        self._num_nodes = None
        self._num_edges = None
        self._num_timestamps = None
