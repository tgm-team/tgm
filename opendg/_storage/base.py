from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

from torch import Tensor

from opendg.typing import Event, TimeDelta


class DGStorageBase(ABC):
    r"""Base class for dynamic graph storage engine."""

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
    def append(self, events: Union[Event, List[Event]]) -> 'DGStorageBase':
        r"""Append events to the temporal end of the dynamic graph."""

    @abstractmethod
    def temporal_coarsening(
        self, time_delta: TimeDelta, agg_func: str = 'sum'
    ) -> 'DGStorageBase':
        r"""Re-index the temporal axis of the dynamic graph."""

    def __len__(self) -> int:
        r"""Returns the number of temporal length of the dynamic graph."""
        return self.num_timestamps

    def __str__(self) -> str:
        r"""Returns summary properties of the dynamic graph."""
        return f'Dynamic Graph Storage Engine ({self.__class__.__name__}), Start Time: {self.start_time}, End Time: {self.end_time}, Nodes: {self.num_nodes}, Edges: {self.num_edges}, Timestamps: {self.num_timestamps}, Time Granularity: {self.time_granularity}'

    @property
    @abstractmethod
    def start_time(self) -> Optional[int]:
        r"""The start time of the dynamic graph. None if the graph is empty."""

    @property
    @abstractmethod
    def end_time(self) -> Optional[int]:
        r"""The end time of the dynamic graph. None, if the graph is empty."""

    @property
    @abstractmethod
    def time_granularity(self) -> Optional[TimeDelta]:
        r"""The time granularity of the dynamic graph. None, if the graph has less than 2 temporal events."""

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        r"""The total number of unique nodes encountered over the dynamic graph."""

    @property
    @abstractmethod
    def num_edges(self) -> int:
        r"""The total number of unique edges encountered over the dynamic graph."""

    @property
    @abstractmethod
    def num_timestamps(self) -> int:
        r"""The total number of unique timestamps encountered over the dynamic graph."""

    @property
    @abstractmethod
    def node_feats(self) -> Optional[Tensor]:
        r"""The aggregated node features over the dynamic graph.

        Retuns a tensor of size T x V x d where

        - T = Number of timestamps
        - V = Number of nodes
        - d = Node feature dimension

        or None if there are no node features on the dynamic graph.
        """

    @property
    @abstractmethod
    def edge_feats(self) -> Optional[Tensor]:
        r"""The aggregated edge features over the dynamic graph.

        Retuns a tensor of size T x E x d where

        - T = Number of timestamps
        - E = Number of edges
        - d = Edge feature dimension

        or None if there are no edge features on the dynamic graph.
        """

    def _check_slice_time_args(self, start_time: int, end_time: int) -> None:
        if start_time > end_time:
            raise ValueError(
                f'Bad slice: start_time must be <= end_time but received: start_time ({start_time}) > end_time ({end_time})'
            )

    def _check_temporal_coarsening_args(
        self, time_delta: TimeDelta, agg_func: str
    ) -> None:
        if not len(self):
            raise ValueError('Cannot temporally coarsen an empty dynamic graph')

        # TODO: Validate time_delta and agg_func
