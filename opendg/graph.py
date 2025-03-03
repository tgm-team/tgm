import copy
from typing import Any, List, Optional, Set, Union

import torch

from opendg._io import read_csv, write_csv
from opendg._storage import DGStorage, DGStorageBase
from opendg.events import EdgeEvent, Event, NodeEvent
from opendg.timedelta import TimeDeltaDG


class DGraph:
    r"""The Dynamic Graph Object. Provides a 'view' over an internal DGStorage backend.

    Args:
        events (List[event]): The list of temporal events (node/edge events) that define the dynamic graph.
        time_delta (Optional[TimeDeltaDG]): Describes the time granularity associated with the event stream.
            If None, then the events are assumed 'ordered', with no specific time unit.
    """

    def __init__(
        self,
        events: Optional[List[Event]] = None,
        time_delta: Optional[TimeDeltaDG] = None,
        _storage: Optional[DGStorageBase] = None,
    ) -> None:
        if _storage is not None:
            if events is not None or time_delta is not None:
                raise ValueError(
                    'Cannot simultaneously initialize a DGraph with _storage and events/time_delta.'
                )

            self._storage = _storage
        else:
            events_list = [] if events is None else events
            self._storage = DGStorage(events_list)

        if time_delta is None:
            self._time_delta = TimeDeltaDG('r')  # Default to ordered granularity
        else:
            self._time_delta = time_delta

        self._check_node_feature_shapes()
        self._check_edge_feature_shapes()

        # Cached values
        self._start_time: Optional[int] = None
        self._end_time: Optional[int] = None
        self._num_nodes: Optional[int] = None
        self._num_edges: Optional[int] = None
        self._num_timestamps: Optional[int] = None
        self._node_slice: Optional[Set[int]] = None

    @classmethod
    def from_csv(
        cls,
        file_path: str,
        time_delta: Optional[TimeDeltaDG] = None,
        *args: Any,
        **kwargs: Any,
    ) -> 'DGraph':
        r"""Load a Dynamic Graph from a csv_file.

        Args:
            file_path (str): The os.pathlike object to read from.
            time_delta (Optional[TimeDeltaDG]): Describes the time granularity associated with the event stream.
                If None, then the events are assumed 'ordered', with no specific time unit.
            args (Any): Optional positional arguments.
            kwargs (Any): Optional keyword arguments.

        Returns:
           DGraph: The newly constructed dynamic graph.
        """
        events = read_csv(file_path, *args, **kwargs)
        return cls(events, time_delta)

    def to_csv(self, file_path: str, *args: Any, **kwargs: Any) -> None:
        r"""Write a Dynamic Graph to a csv_file.

        Args:
            file_path (str): The os.pathlike object to write to.
            args (Any): Optional positional arguments.
            kwargs (Any): Optional keyword arguments.
        """
        # TODO: materialize
        events = self._storage.to_events(
            self._start_time, self._end_time, self._node_slice
        )
        write_csv(events, file_path, *args, **kwargs)

    def slice_time(self, start_time: int, end_time: int) -> 'DGraph':
        r"""Extract temporal slice of the dynamic graph between start_time and end_time.

        Args:
            start_time (int): The start of the temporal slice.
            end_time (int): The end of the temporal slice (exclusive).

        Returns:
            DGraph view of events between start and end_time.
        """
        # TODO: Check args here
        dg = copy.copy(self)
        force_cache_refresh = False

        if self.start_time is not None and start_time > self.start_time:
            dg._start_time = start_time
            force_cache_refresh = True

        if self.end_time is not None and end_time < self.end_time:
            dg._end_time = end_time
            force_cache_refresh = True

        if force_cache_refresh:
            dg._invalidate_cache()

        return dg

    def slice_nodes(self, nodes: List[int]) -> 'DGraph':
        r"""Extract topological slice of the dynamcic graph given the list of nodes.

        Args:
            nodes (List[int]): The list of node ids to slice from.

        Returns:
            DGraph copy of events related to the input nodes.
        """
        dg = copy.copy(self)
        dg._invalidate_cache()

        if self._node_slice is None:
            self._node_slice = set(range(self.num_nodes))
        dg._node_slice = self._node_slice & set(nodes)

        return dg

    def append(self, events: Union[Event, List[Event]]) -> None:
        r"""Append events to the temporal end of the dynamic graph.

        Args:
            events (Union[Event, List[Event]]): The event of list of events to add to the temporal graph.

        """
        raise NotImplementedError

    def temporal_coarsening(
        self, time_delta: TimeDeltaDG, agg_func: str = 'sum'
    ) -> None:
        r"""Re-index the temporal axis of the dynamic graph.

        Args:
            time_delta (TimeDeltaDG): The time granularity to use.
            agg_func (Union[str, Callable]): The aggregation / reduction function to apply.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        r"""Returns the number of temporal length of the dynamic graph."""
        return self.num_timestamps

    def __str__(self) -> str:
        r"""Returns summary properties of the dynamic graph."""
        return f'Dynamic Graph Storage Engine ({self._storage.__class__.__name__}), Start Time: {self.start_time}, End Time: {self.end_time}, Nodes: {self.num_nodes}, Edges: {self.num_edges}, Timestamps: {self.num_timestamps}, Time Delta: {self.time_delta}'

    @property
    def start_time(self) -> Optional[int]:
        r"""The start time of the dynamic graph. None if the graph is empty."""
        if self._start_time is None:
            self._start_time = self._storage.get_start_time(self._node_slice)
        return self._start_time

    @property
    def end_time(self) -> Optional[int]:
        r"""The end time of the dynamic graph. None, if the graph is empty."""
        if self._end_time is None:
            self._end_time = self._storage.get_end_time(self._node_slice)
        return self._end_time

    @property
    def time_delta(self) -> TimeDeltaDG:
        r"""The time granularity of the dynamic graph."""
        return self._time_delta

    @property
    def num_nodes(self) -> int:
        r"""The total number of unique nodes encountered over the dynamic graph."""
        if self._num_nodes is None:
            if self._node_slice is None:
                self._node_slice = self._storage.get_nodes(
                    self._start_time, self._end_time
                )
            self._num_nodes = max(self._node_slice) + 1
        return self._num_nodes

    @property
    def num_edges(self) -> int:
        r"""The total number of unique edges encountered over the dynamic graph."""
        if self._num_edges is None:
            self._num_edges = self._storage.get_num_edges(
                self._start_time, self._end_time, self._node_slice
            )
        return self._num_edges

    @property
    def num_timestamps(self) -> int:
        r"""The total number of unique timestamps encountered over the dynamic graph."""
        if self._num_timestamps is None:
            self._num_timestamps = self._storage.get_num_timestamps(
                self._start_time, self._end_time, self._node_slice
            )
        return self._num_timestamps

    def _invalidate_cache(self) -> None:
        self._start_time = None
        self._end_time = None
        self._num_nodes = None
        self._num_edges = None
        self._num_timestamps = None
        self._node_slice = None

    def _check_slice_time_args(self, start_time: int, end_time: int) -> None:
        if start_time > end_time:
            raise ValueError(
                f'Bad slice: start_time must be <= end_time but received: start_time ({start_time}) > end_time ({end_time})'
            )

    def _check_temporal_coarsening_args(
        self, time_delta: TimeDeltaDG, agg_func: str
    ) -> None:
        if not len(self):
            raise ValueError('Cannot temporally coarsen an empty dynamic graph')

        # TODO: Validate time_delta and agg_func

    def _check_node_feature_shapes(
        self, events: List[Event], expected_shape: Optional[torch.Size] = None
    ) -> Optional[torch.Size]:
        node_feats_shape = expected_shape

        for event in events:
            if isinstance(event, NodeEvent) and event.features is not None:
                if node_feats_shape is None:
                    node_feats_shape = event.features.shape
                elif node_feats_shape != event.features.shape:
                    raise ValueError(
                        f'Incompatible node features shapes: {node_feats_shape} != {event.features.shape}'
                    )
        return node_feats_shape

    def _check_edge_feature_shapes(
        self, events: List[Event], expected_shape: Optional[torch.Size] = None
    ) -> Optional[torch.Size]:
        edge_feats_shape = expected_shape

        for event in events:
            if isinstance(event, EdgeEvent) and event.features is not None:
                if edge_feats_shape is None:
                    edge_feats_shape = event.features.shape
                elif edge_feats_shape != event.features.shape:
                    raise ValueError(
                        f'Incompatible edge features shapes: {edge_feats_shape} != {event.features.shape}'
                    )
        return edge_feats_shape
