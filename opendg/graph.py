from typing import Any, List, Optional, Union

import pandas as pd

from opendg._io import read_csv, read_pandas, write_csv
from opendg._storage import DGStorage
from opendg.events import Event
from opendg.timedelta import TimeDeltaDG


class DGraph:
    r"""The Dynamic Graph Object.

    Args:
        events (List[event]): The list of temporal events (node/edge events) that define the dynamic graph.
        time_delta (Optional[TimeDeltaDG]): Describes the time granularity associated with the event stream.
            If None, then the events are assumed 'ordered', with no specific time unit.
    """

    def __init__(
        self, events: List[Event], time_delta: Optional[TimeDeltaDG] = None
    ) -> None:
        if time_delta is None:
            time_delta = TimeDeltaDG('r')  # Default to ordered if granularity
        self._storage = DGStorage(events, time_delta)

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

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        src_col: str,
        dst_col: str,
        time_col: Optional[str] = None,
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
        ordered_time_delta = time_delta is None or time_delta.is_ordered
        if time_col is None and not ordered_time_delta:
            raise ValueError(
                'Must specify "time_col" when using non-ordered time delta.'
            )

        events = read_pandas(df, src_col, dst_col, time_col, edge_feature_col)
        return cls(events, time_delta)

    def to_csv(self, file_path: str, *args: Any, **kwargs: Any) -> None:
        r"""Write a Dynamic Graph to a csv_file.

        Args:
            file_path (str): The os.pathlike object to write to.
            args (Any): Optional positional arguments.
            kwargs (Any): Optional keyword arguments.
        """
        events = self._storage.to_events()
        write_csv(events, file_path, *args, **kwargs)

    def slice_time(self, start_time: int, end_time: int) -> None:
        r"""Extract temporal slice of the dynamic graph between start_time and end_time.

        Args:
            start_time (int): The start of the temporal slice.
            end_time (int): The end of the temporal slice (exclusive).
        """
        self._storage.slice_time(start_time, end_time)

    def slice_nodes(self, nodes: List[int]) -> None:
        r"""Extract topological slice of the dynamcic graph given the list of nodes.

        Args:
            nodes (List[int]): The list of node ids to slice from.
        """
        self._storage.slice_nodes(nodes)

    def append(self, events: Union[Event, List[Event]]) -> None:
        r"""Append events to the temporal end of the dynamic graph.

        Args:
            events (Union[Event, List[Event]]): The event of list of events to add to the temporal graph.

        """
        self._storage.append(events)

    def temporal_coarsening(
        self, time_delta: TimeDeltaDG, agg_func: str = 'sum'
    ) -> None:
        r"""Re-index the temporal axis of the dynamic graph.

        Args:
            time_delta (TimeDeltaDG): The time granularity to use.
            agg_func (Union[str, Callable]): The aggregation / reduction function to apply.
        """
        self._storage.temporal_coarsening(time_delta, agg_func)

    def __len__(self) -> int:
        r"""Returns the number of temporal length of the dynamic graph."""
        return self._storage.num_timestamps

    def __str__(self) -> str:
        r"""Returns summary properties of the dynamic graph."""
        return self._storage.__str__()

    @property
    def start_time(self) -> Optional[int]:
        r"""The start time of the dynamic graph. None if the graph is empty."""
        return self._storage.start_time

    @property
    def end_time(self) -> Optional[int]:
        r"""The end time of the dynamic graph. None, if the graph is empty."""
        return self._storage.end_time

    @property
    def time_delta(self) -> Optional[TimeDeltaDG]:
        r"""The time granularity of the dynamic graph."""
        return self._storage.time_delta

    @property
    def num_nodes(self) -> int:
        r"""The total number of unique nodes encountered over the dynamic graph."""
        return self._storage.num_nodes

    @property
    def num_edges(self) -> int:
        r"""The total number of unique edges encountered over the dynamic graph."""
        return self._storage.num_edges

    @property
    def num_timestamps(self) -> int:
        r"""The total number of unique timestamps encountered over the dynamic graph."""
        return self._storage.num_timestamps
