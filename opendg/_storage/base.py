from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

from torch import Tensor

from opendg.typing import Event


class DGStorageBase(ABC):
    r"""Base class for temporal graph storage engine."""

    @classmethod
    @abstractmethod
    def from_events(cls, events: List[Event]) -> 'DGStorageBase':
        r"""Create dynamic graph from a list of events."""

    @abstractmethod
    def to_events(self) -> List[Event]:
        r"""Convert dynamic graph to a list of events."""

    @abstractmethod
    def slice_time(self, start_time: int, end_time: int) -> 'DGStorageBase':
        r"""Extract temporal slice of the dynamic graph between start_time and end_time."""

    @abstractmethod
    def slice_nodes(self, nodes: List[int]) -> 'DGStorageBase':
        r"""Extract topological slice of the dynamcic graph given the list of nodes."""

    @abstractmethod
    def get_nbrs(self, nodes: List[int]) -> Dict[int, List[Tuple[int, int]]]:
        r"""Return a list of neighbour, timestamp pairs for each node in the nodes list."""

    @abstractmethod
    def materialize_node_features(self) -> Tensor:
        r"""Materialiize the dynamic graph node feature data."""

    @abstractmethod
    def materialize_edge_features(self) -> Tensor:
        r"""Materialiize the dynamic graph edge feature data."""

    @abstractmethod
    def update(self, events: Union[Event, List[Event]]) -> 'DGStorageBase':
        r"""Update the dynamic graph with an event of list of events."""

    @abstractmethod
    def temporal_coarsening(
        self, time_delta: int, agg_func: str = 'sum'
    ) -> 'DGStorageBase':
        r"""Re-index the temporal axis of the dynamic graph."""

    def __len__(self) -> int:
        r"""Returns the number of temporal length of the dynamic graph."""
        return self.num_timestamps

    @property
    @abstractmethod
    def start_time(self) -> int:
        r"""The start time of the temporal graph."""

    @property
    @abstractmethod
    def end_time(self) -> int:
        r"""The end time of the temporal graph."""

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        r"""The total number of unique nodes encountered over the temporal graph."""

    @property
    @abstractmethod
    def num_edges(self) -> int:
        r"""The total number of unique edges encountered over the temporal graph."""

    @property
    @abstractmethod
    def num_timestamps(self) -> int:
        r"""The total number of unique timestamps encountered over the temporal graph."""

    def _check_slice_time_args(self, start_time: int, end_time: int):
        if start_time > end_time:
            raise ValueError(
                f'Bad slice: start_time must be <= end_time but received: start_time ({start_time}) > end_time ({end_time})'
            )
