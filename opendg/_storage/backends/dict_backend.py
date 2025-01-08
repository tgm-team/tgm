import itertools
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

from torch import Tensor

from opendg.typing import Event, EventsDict, TimeDelta

from ..base import DGStorageBase


class DGStorageDictBackend(DGStorageBase):
    r"""Dictionary implementation of temporal graph storage engine."""

    def __init__(
        self,
        events_dict: EventsDict,
        node_feats: Optional[Dict[int, Dict[int, Tensor]]] = None,
        edge_feats: Optional[Dict[int, Tensor]] = None,
    ) -> None:
        self._events_dict: EventsDict = events_dict
        self._node_feats: Optional[Dict[int, Dict[int, Tensor]]] = node_feats
        self._edge_feats: Optional[Dict[int, Tensor]] = edge_feats

        # Cached Values
        self._start_time: Optional[int] = None
        self._end_time: Optional[int] = None
        self._num_nodes: Optional[int] = None
        self._num_edges: Optional[int] = None
        self._num_timestamps: Optional[int] = None
        self._time_granularity: Optional[TimeDelta] = None

    @classmethod
    def from_events(cls, events: List[Event]) -> 'DGStorageBase':
        # TODO: Figure out API for node/edge feats here
        events_dict = defaultdict(list)
        for t, u, v in events:
            events_dict[t].append((u, v))
        return cls(events_dict)

    def to_events(self) -> List[Event]:
        # TODO: Figure out API for node/edge feats here
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

        if self._node_feats is not None:
            self._node_feats = {
                k: v for k, v in self._node_feats.items() if start_time <= k < end_time
            }

        if self._edge_feats is not None:
            self._edge_feats = {
                k: v for k, v in self._edge_feats.items() if start_time <= k < end_time
            }

        return self

    def slice_nodes(self, nodes: List[int]) -> 'DGStorageBase':
        self._invalidate_cache()

        events_dict = defaultdict(list)
        edge_feats: Optional[Dict[int, Tensor]] = {}

        for t, edges in self._events_dict.items():
            for edge in edges:
                if len(set(edge).intersection(nodes)):
                    # TODO: Update edge_feats
                    events_dict[t].append(edge)
        self._events_dict = events_dict

        if self._node_feats is not None:
            node_feats: Optional[Dict[int, Dict[int, Tensor]]] = {}
            for t, node_feats_t in self._node_feats.items():
                for node in node_feats_t:
                    if node in nodes:
                        # TODO Update node feats
                        continue
            self._node_feats = node_feats

        if self._edge_feats is not None:
            self._edge_feats = edge_feats

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

    def append(self, events: Union[Event, List[Event]]) -> 'DGStorageBase':
        # TODO: Figure out API for node/edge feats here
        self._invalidate_cache()
        if not isinstance(events, list):
            events = [events]
        for t, u, v in events:
            self._events_dict[t].append((u, v))
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
        for t, edges in self._events_dict.items():
            bin_t = -((t - self.start_time) // -interval_size)  # Ceiling division

            # TODO: Use the agg_func
            events_dict[bin_t].extend(edges)

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
            edges = set(itertools.chain.from_iterable(self._events_dict.values()))
            self._num_nodes = len(set(itertools.chain.from_iterable(edges)))
        return self._num_nodes

    @property
    def num_edges(self) -> int:
        if self._num_edges is None:
            edges = set(itertools.chain.from_iterable(self._events_dict.values()))
            self._num_edges = len(edges)
        return self._num_edges

    @property
    def num_timestamps(self) -> int:
        if self._num_timestamps is None:
            self._num_timestamps = len(self._events_dict)
        return self._num_timestamps

    @property
    def node_feats(self) -> Optional[Tensor]:
        if self._node_feats is None:
            return None

        # TODO
        # self._node_feats: Optional[Dict[int, Dict[int, Tensor]]] = node_feats
        return None

    @property
    def edge_feats(self) -> Optional[Tensor]:
        if self._edge_feats is None:
            return None

        # TODO
        # self._edge_feats: Optional[Dict[int, Tensor]] = edge_feats
        return None

    def _invalidate_cache(self) -> None:
        self._start_time = None
        self._end_time = None
        self._num_nodes = None
        self._num_edges = None
        self._num_timestamps = None
        self._time_granularity = None
