from abc import ABC, abstractmethod
from typing import List, Optional, Set, Union

from torch import Tensor

from opendg.events import Event
from opendg.timedelta import TimeDeltaDG


class DGStorageBase(ABC):
    r"""Base class for dynamic graph storage engine."""

    @abstractmethod
    def __init__(self, events: List[Event]) -> None:
        pass

    @abstractmethod
    def to_events(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        node_slice: Optional[Set[int]] = None,
    ) -> List[Event]:
        pass

    @abstractmethod
    def get_start_time(self, node_slice: Optional[Set[int]] = None) -> Optional[int]:
        pass

    @abstractmethod
    def get_end_time(self, node_slice: Optional[Set[int]] = None) -> Optional[int]:
        pass

    @abstractmethod
    def get_nodes(
        self, start_time: Optional[int] = None, end_time: Optional[int] = None
    ) -> Set[int]:
        pass

    @abstractmethod
    def get_num_edges(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        node_slice: Optional[Set[int]] = None,
    ) -> int:
        pass

    @abstractmethod
    def get_num_timestamps(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        node_slice: Optional[Set[int]] = None,
    ) -> int:
        pass

    @abstractmethod
    def append(self, events: Union[Event, List[Event]]) -> 'DGStorageBase':
        pass

    @abstractmethod
    def temporal_coarsening(
        self, time_delta: TimeDeltaDG, agg_func: str = 'sum'
    ) -> 'DGStorageBase':
        pass

    @abstractmethod
    def get_node_feats(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        node_slice: Optional[Set[int]] = None,
    ) -> Optional[Tensor]:
        pass

    @abstractmethod
    def get_edge_feats(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        node_slice: Optional[Set[int]] = None,
    ) -> Optional[Tensor]:
        pass
