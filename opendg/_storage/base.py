import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from torch import Tensor

from opendg.events import EdgeEvent, Event, NodeEvent


@dataclass(slots=True)
class DGSliceTracker:
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    node_slice: Optional[Set[int]] = None


class DGStorageBase(ABC):
    r"""Base class for dynamic graph storage engine."""

    @abstractmethod
    def __init__(self, events: List[Event]) -> None:
        pass

    @abstractmethod
    def to_events(self, slice: DGSliceTracker) -> List[Event]:
        pass

    @abstractmethod
    def get_start_time(self, slice: DGSliceTracker) -> Optional[int]:
        pass

    @abstractmethod
    def get_end_time(self, slice: DGSliceTracker) -> Optional[int]:
        pass

    @abstractmethod
    def get_nodes(self, slice: DGSliceTracker) -> Set[int]:
        pass

    @abstractmethod
    def get_edges(self, slice: DGSliceTracker) -> Tuple[Tensor, Tensor, Tensor]:
        pass

    @abstractmethod
    def get_num_timestamps(self, slice: DGSliceTracker) -> int:
        pass

    @abstractmethod
    def get_nbrs(
        self,
        seed_nodes: Set[int],
        num_nbrs: List[int],
        slice: DGSliceTracker,
    ) -> Dict[int, List[List[Tuple[int, int]]]]:
        pass

    @abstractmethod
    def get_node_feats(self, slice: DGSliceTracker) -> Optional[Tensor]:
        pass

    @abstractmethod
    def get_edge_feats(self, slice: DGSliceTracker) -> Optional[Tensor]:
        pass

    @abstractmethod
    def get_node_feats_dim(self) -> Optional[int]:
        pass

    @abstractmethod
    def get_edge_feats_dim(self) -> Optional[int]:
        pass

    def _sort_events_list_if_needed(self, events: List[Event]) -> List[Event]:
        if not all(isinstance(event, Event) for event in events):
            raise ValueError('bad type when initializing DGStorage from events list')
        if all(events[i].t <= events[i + 1].t for i in range(len(events) - 1)):
            return events
        warnings.warn('received a non-chronological list of events, sorting by time')
        events.sort(key=lambda x: x.t)
        return events

    def _check_node_feature_dim(self, events: List[Event]) -> Optional[int]:
        shape = None
        for event in events:
            if isinstance(event, NodeEvent) and event.features is not None:
                if shape is None:
                    shape = event.features.shape
                elif shape != event.features.shape:
                    raise ValueError(
                        f'Node feature shapes non-homogenous: {shape} != {event.features.shape}'
                    )
        if shape is not None and len(shape) > 1:
            raise ValueError(f'Only 1-d node features supported but got: ({shape})')
        return shape[0] if shape is not None else shape

    def _check_edge_feature_dim(self, events: List[Event]) -> Optional[int]:
        shape = None
        for event in events:
            if isinstance(event, EdgeEvent) and event.features is not None:
                if shape is None:
                    shape = event.features.shape
                elif shape != event.features.shape:
                    raise ValueError(
                        f'Edge feature shapes non-homogenous: {shape} != {event.features.shape}'
                    )
        if shape is not None and len(shape) > 1:
            raise ValueError(f'Only 1-d node features supported but got: ({shape})')
        return shape[0] if shape is not None else shape
