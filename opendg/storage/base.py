from abc import ABC, abstractmethod
from typing import Dict, List, Union

from opendg.typing import Event, Snapshot


class DGStorageBase(ABC):
    """Base class for all temporal graph data structures."""

    @classmethod
    @abstractmethod
    def from_events(cls, events: List[Event]) -> 'DGStorageBase':
        """Create dynamic graph from a list of events."""

    @classmethod
    @abstractmethod
    def from_snapshots(cls, snapshots: List[Snapshot]) -> 'DGStorageBase':
        """Create dynamic graph from a list of snapshots."""

    @abstractmethod
    def to_events(self) -> List[Event]:
        """Convert dynamic graph to a list of events."""

    @abstractmethod
    def to_snapshots(self) -> List[Snapshot]:
        """Converts dynamic graph to a list of snapshots."""

    @abstractmethod
    def slice_time(self, start_time: int, end_time: int) -> 'DGStorageBase':
        """Extract temporal slice of the dynamic graph between start_time and end_time."""

    @abstractmethod
    def slice_nodes(self, nodes: List[int]) -> 'DGStorageBase':
        """Extract topological slice of the dynamcic graph given the list of nodes."""

    @abstractmethod
    def get_nbrs(self, nodes: List[int]) -> Dict[int, List[int]]:
        """Return the list of neighbours for each node in the nodes list."""

    @abstractmethod
    def update(self, events: Union[Event, List[Event]]) -> 'DGStorageBase':
        """Update the dynamic graph with an event of list of events."""

    @abstractmethod
    def reindex(self, time_delta: int) -> 'DGStorageBase':
        """Re-index the temporal axis of the dynamic graph."""

    @property
    @abstractmethod
    def start_time(self) -> int:
        """The start time of the temporal graph."""

    @property
    @abstractmethod
    def end_time(self) -> int:
        """The end time of the temporal graph."""

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        """The total number of unique nodes encountered over the temporal graph."""

    @property
    @abstractmethod
    def num_edges(self) -> int:
        """The total number of unique edges encountered over the temporal graph."""

    @property
    @abstractmethod
    def num_timestamps(self) -> int:
        """The total number of unique timestamps encountered over the temporal graph."""
