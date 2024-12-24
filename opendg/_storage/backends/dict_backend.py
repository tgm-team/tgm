from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

from opendg.typing import Event

from ..base import DGStorageBase


class DGStorageDictBackend(DGStorageBase):
    r"""Dictionary implementation of temporal graph storage engine."""

    def __init__(self, events_dict: Dict[int, Tuple[int, int]]) -> None:
        self._events_dict: Dict[int, Tuple[int, int]] = events_dict

        # Cached Values
        self._start_time: Optional[int] = None
        self._end_time: Optional[int] = None
        self._num_nodes: Optional[int] = None
        self._num_edges: Optional[int] = None
        self._num_timestamps: Optional[int] = None

    @classmethod
    def from_events(cls, events: List[Event]) -> 'DGStorageBase':
        events_dict = {}
        for t, u, v in events:
            events_dict[t] = (u, v)
        return cls(events_dict)

    def to_events(self) -> List[Event]:
        events: List[Event] = []
        for t, (u, v) in self._events_dict.items():
            events.append((t, u, v))
        return events

    def slice_time(self, start_time: int, end_time: int) -> 'DGStorageBase':
        self._check_slice_time_args(start_time, end_time)
        self._invalid_cache()
        self._events_dict = {
            k: v for k, v in self._events_dict.items() if start_time <= k < end_time
        }
        return self

    def slice_nodes(self, nodes: List[int]) -> 'DGStorageBase':
        self._invalid_cache()
        self._events_dict = {
            k: v
            for k, v in self._events_dict.items()
            if len(set(v).intersection(nodes))
        }
        return self

    def get_nbrs(self, nodes: List[int]) -> Dict[int, List[Tuple[int, int]]]:
        nbrs_dict = defaultdict(list)
        for t, (u, v) in self._events_dict.items():
            if u in nodes:
                nbrs_dict[u].append((v, t))
            if v in nodes:
                nbrs_dict[v].append((u, t))

        return dict(nbrs_dict)

    def update(self, events: Union[Event, List[Event]]) -> 'DGStorageBase':
        if not isinstance(events, list):
            events = [events]
        for t, u, v in events:
            self._events_dict[t] = (u, v)
        return self

    def temporal_coarsening(
        self, time_delta: int, agg_func: str = 'sum'
    ) -> 'DGStorageBase':
        raise NotImplementedError()

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
        if self._num_nodes is None:
            self._num_nodes = len(set(sum(map(list, self._events_dict.values()), [])))
        return self._num_nodes

    @property
    def num_edges(self) -> int:
        if self._num_edges is None:
            self._num_edges = len(set(self._events_dict.values()))
        return self._num_edges

    @property
    def num_timestamps(self) -> int:
        if self._num_timestamps is None:
            self._num_timestamps = len(self._events_dict)
        return self._num_timestamps

    def _invalid_cache(self) -> None:
        self._start_time = None
        self._end_time = None
        self._num_nodes = None
        self._num_edges = None
        self._num_timestamps = None
