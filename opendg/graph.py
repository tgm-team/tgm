from typing import List, Optional, Union

from opendg._storage import DGStorage
from opendg.events import Event
from opendg.typing import TimeDelta


class DGraph:
    r"""The Dynamic Graph Object."""

    def __init__(self, events: List[Event]) -> None:
        self._storage = DGStorage(events)

    @classmethod
    def from_csv(cls, file_path: str) -> 'DGraph':
        r"""Load a Dynamic Graph from a csv_file.

        Args:
            file_path (str): The os.pathlike object to read from.

        Returns:
           DGraph: The newly constructed dynamic graph.
        """
        raise NotImplementedError()

    def to_csv(self, file_path: str) -> None:
        r"""Write a Dynamic Graph to a csv_file.

        Args:
            file_path (str): The os.pathlike object to write to.
        """
        raise NotImplementedError()

    def slice_time(self, start_time: int, end_time: int) -> None:
        r"""Extract temporal slice of the dynamic graph between start_time and end_time.

        Args:
            start_time (int): The start of the temporal slice.
            end_time (int): The end of the temporal slice (exclusive).
        """
        self._storage.slice_time(start_time, end_time)

    def append(self, events: Union[Event, List[Event]]) -> None:
        r"""Append events to the temporal end of the dynamic graph.

        Args:
            events (Union[Event, List[Event]]): The event of list of events to add to the temporal graph.

        """
        self._storage.append(events)

    def temporal_coarsening(self, time_delta: TimeDelta, agg_func: str = 'sum') -> None:
        r"""Re-index the temporal axis of the dynamic graph.

        Args:
            time_delta (TimeDelta): The time granularity to use.
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
    def time_granularity(self) -> Optional[TimeDelta]:
        r"""The time granularity of the dynamic graph. None, if the graph has less than 2 temporal events."""
        return self._storage.time_granularity

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
