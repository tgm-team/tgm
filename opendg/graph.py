import copy
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from torch import Tensor

from opendg._io import read_csv, read_pandas, write_csv
from opendg._storage import DGStorage, DGStorageBase
from opendg.events import Event
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
        self._storage = self._check_storage_args(events, _storage)

        if time_delta is None:
            self._time_delta = TimeDeltaDG('r')  # Default to ordered granularity
        else:
            self._time_delta = time_delta

        self._cache: Dict[str, Any] = {}

    @classmethod
    def from_csv(
        cls,
        file_path: str,
        src_col: str,
        dst_col: str,
        time_col: str,
        edge_feature_col: Optional[List[str]] = None,
        time_delta: Optional[TimeDeltaDG] = None,
    ) -> 'DGraph':
        r"""Load a Dynamic Graph from a csv_file.

        Args:
            file_path (str): The os.pathlike object to read from.
            src_col (str): Column in the csv corresponding to the source edge id.
            dst_col (str): Column in the csv corresponding to the destination edge id.
            time_col (str): Column in the csv corresponding to the timestamp.
            edge_feature_col (Optional[List[str]]): Column list in the csv corresponding to the edge features.
            time_delta (Optional[TimeDeltaDG]): Describes the time granularity associated with the event stream.
                If None, then the events are assumed 'ordered', with no specific time unit.

        Returns:
           DGraph: The newly constructed dynamic graph.
        """
        events = read_csv(file_path, src_col, dst_col, time_col, edge_feature_col)
        return cls(events, time_delta)

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        src_col: str,
        dst_col: str,
        time_col: str,
        edge_feature_col: Optional[str] = None,
        time_delta: Optional[TimeDeltaDG] = None,
    ) -> 'DGraph':
        r"""Load a Dynamic Graph from a Pandas DataFrame.

        Args:
            df (pd.DataFrame): The dataframe to read from.
            src_col (str): Column in the dataframe corresponding to the source edge id.
            dst_col (str): Column in the dataframe corresponding to the destination edge id.
            time_col (str): Column in the dataframe corresponding to the timestamp.
            edge_feature_col (Optional[str]): Optional column in the dataframe corresponding to the edge features.
            time_delta (Optional[TimeDeltaDG]): Describes the time granularity associated with the event stream.
                If None, then the events are assumed 'ordered', with no specific time unit.

        Returns:
           DGraph: The newly constructed dynamic graph.
        """
        events = read_pandas(df, src_col, dst_col, time_col, edge_feature_col)
        return cls(events, time_delta)

    def to_csv(
        self,
        file_path: str,
        src_col: str,
        dst_col: str,
        time_col: str,
        edge_feature_col: Optional[List[str]] = None,
    ) -> None:
        r"""Write a Dynamic Graph to a csv_file.

        Args:
            file_path (str): The os.pathlike object to write to.
            src_col (str): Column in the dataframe corresponding to the source edge id.
            dst_col (str): Column in the dataframe corresponding to the destination edge id.
            time_col (str): Column in the dataframe corresponding to the timestamp.
            edge_feature_col (Optional[List[str]]): Column list in the csv corresponding to the edge features.
        """
        events = self._storage.to_events(
            start_time=self._cache.get('start_time'),
            end_time=self._cache.get('end_time'),
            node_slice=self._cache.get('node_slice'),
        )
        write_csv(events, file_path, src_col, dst_col, time_col, edge_feature_col)

    def slice_time(self, start_time: int, end_time: int) -> 'DGraph':
        r"""Extract temporal slice of the dynamic graph between start_time and end_time.

        Args:
            start_time (int): The start of the temporal slice.
            end_time (int): The end of the temporal slice (inclusive).

        Returns:
            DGraph view of events between start and end_time.
        """
        self._check_slice_time_args(start_time, end_time)

        dg = copy.copy(self)

        # Cache must be deep copied to avoid aliased properties
        dg._cache = copy.deepcopy(self._cache)

        new_start_time = start_time
        new_end_time = end_time - 1  # Because slicing is end range exclusive

        if self.start_time is not None and self.start_time > new_start_time:
            new_start_time = self.start_time  # Keep our previous start time

        if self.end_time is not None and self.end_time < new_end_time:
            new_end_time = self.end_time  # Keep our previous end time

        # Force cache refresh on the new copy if we actually sliced the graph
        if new_start_time != self.start_time or new_end_time != self.end_time:
            dg._cache.clear()

        new_end_time += 1  # Keep the end-range exclusive value in cache
        dg._cache['start_time'] = new_start_time
        dg._cache['end_time'] = new_end_time

        return dg

    def slice_nodes(self, nodes: List[int]) -> 'DGraph':
        r"""Extract topological slice of the dynamcic graph given the list of nodes.

        Args:
            nodes (List[int]): The list of node ids to slice from.

        Returns:
            DGraph copy of events related to the input nodes.
        """
        dg = copy.copy(self)
        dg._cache = {}

        if self._cache.get('node_slice') is None:
            self._cache['node_slice'] = set(range(self.num_nodes))

        dg._cache['node_slice'] = self._cache['node_slice'] & set(nodes)

        return dg

    def append(self, events: Union[Event, List[Event]]) -> None:
        r"""Append events to the temporal end of the dynamic graph.

        Args:
            events (Union[Event, List[Event]]): The event of list of events to add to the temporal graph.

        """
        if isinstance(events, List) and not len(events):
            return
        self._check_append_args(events)

        # TODO: Materialize / copy on write
        if any([cached_value is not None for cached_value in self._cache.values()]):
            # Need to decide how to handle this, e.g. throw a userwarning that
            # they modified the backed DGStorage of a view, and could lead to undefined behaviour
            raise NotImplementedError('Append to view not implemented')

        self._storage.append(events)

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
        if self._cache.get('start_time') is None:
            self._cache['start_time'] = self._storage.get_start_time(
                self._cache.get('node_slice')
            )
        return self._cache['start_time']

    @property
    def end_time(self) -> Optional[int]:
        r"""The end time of the dynamic graph. None, if the graph is empty."""
        if self._cache.get('end_time') is None:
            self._cache['end_time'] = self._storage.get_end_time(
                self._cache.get('node_slice')
            )
            # We cache the end_time + 1 so that all our time constrained queries
            # use the half-open interval: [start_time, end_time + 1) = [start_time, end_time].
            # If we considered everything end-time inclusive, this would not be needed.
            if self._cache['end_time'] is not None:
                self._cache['end_time'] += 1

        if self._cache['end_time'] is not None:
            # Since our cache stores end_time + 1, we subtract back one to yield the
            # actual end time in our DG.
            return self._cache['end_time'] - 1
        return self._cache['end_time']

    @property
    def time_delta(self) -> TimeDeltaDG:
        r"""The time granularity of the dynamic graph."""
        return self._time_delta

    @property
    def num_nodes(self) -> int:
        r"""The total number of unique nodes encountered over the dynamic graph."""
        if self._cache.get('num_nodes') is None:
            if self._cache.get('node_slice') is None:
                self._cache['node_slice'] = self._storage.get_nodes(
                    self._cache.get('start_time'), self._cache.get('end_time')
                )
            if len(self._cache['node_slice']):
                self._cache['num_nodes'] = max(self._cache['node_slice']) + 1
            else:
                self._cache['num_nodes'] = 0
        return self._cache['num_nodes']

    @property
    def num_edges(self) -> int:
        r"""The total number of unique edges encountered over the dynamic graph."""
        if self._cache.get('num_edges') is None:
            self._cache['num_edges'] = self._storage.get_num_edges(
                self._cache.get('start_time'),
                self._cache.get('end_time'),
                self._cache.get('node_slice'),
            )
        return self._cache['num_edges']

    @property
    def num_timestamps(self) -> int:
        r"""The total number of unique timestamps encountered over the dynamic graph."""
        if self._cache.get('num_timestamps') is None:
            self._cache['num_timestamps'] = self._storage.get_num_timestamps(
                self._cache.get('start_time'),
                self._cache.get('end_time'),
                self._cache.get('node_slice'),
            )
        return self._cache['num_timestamps']

    @property
    def node_feats(self) -> Optional[Tensor]:
        r"""The aggregated node features over the dynamic graph.

        Returns a tensor.sparse_coo_tensor of size T x V x d where

        - T = Number of timestamps
        - V = Number of nodes
        - d = Node feature dimension
        or None if there are no node features on the dynamic graph.

        """
        if self._cache.get('node_feats') is None:
            self._cache['node_feats'] = self._storage.get_node_feats(
                self._cache.get('start_time'),
                self._cache.get('end_time'),
                self._cache.get('node_slice'),
            )
        return self._cache['node_feats']

    @property
    def edge_feats(self) -> Optional[Tensor]:
        r"""The aggregated edge features over the dynamic graph.

        Returns a tensor.sparse_coo_tensor of size T x V x V x d where

        - T = Number of timestamps
        - E = Number of edges
        - d = Edge feature dimension

        or None if there are no edge features on the dynamic graph.
        """
        if self._cache.get('edge_feats') is None:
            self._cache['edge_feats'] = self._storage.get_edge_feats(
                self._cache.get('start_time'),
                self._cache.get('end_time'),
                self._cache.get('node_slice'),
            )
        return self._cache['edge_feats']

    def _check_storage_args(
        self,
        events: Optional[List[Event]] = None,
        _storage: Optional[DGStorageBase] = None,
    ) -> 'DGStorageBase':
        if _storage is not None:
            if events is not None:
                raise ValueError(
                    'Cannot simultaneously initialize a DGraph with _storage and events/time_delta.'
                )
            return _storage
        else:
            events_list = [] if events is None else events
            return DGStorage(events_list)

    def _check_slice_time_args(self, start_time: int, end_time: int) -> None:
        if start_time > end_time:
            raise ValueError(
                f'Bad slice: start_time must be <= end_time but received: start_time ({start_time}) > end_time ({end_time})'
            )

    def _check_append_args(self, events: Union[Event, List[Event]]) -> None:
        if not isinstance(events, List):
            events = [events]

        min_new_event_time = min([event.time for event in events])
        if self.end_time is not None and min_new_event_time < self.end_time:
            raise ValueError(
                'Appending is only supported at the end of a DGraph. '
                f'Tried to append a new event with time: {min_new_event_time} which is strictly less '
                f'than the current end time: {self.end_time}'
            )
