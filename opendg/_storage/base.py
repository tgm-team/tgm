import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor

from opendg.events import EdgeEvent, Event, NodeEvent


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
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        node_slice: Optional[Set[int]] = None,
    ) -> Set[int]:
        pass

    @abstractmethod
    def get_edges(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        node_slice: Optional[Set[int]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
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
    def get_nbrs(
        self,
        seed_nodes: List[int],
        num_nbrs: List[int],
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        node_slice: Optional[Set[int]] = None,
    ) -> Dict[int, List[List[Tuple[int, int]]]]:
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

    def _sort_events_list_if_needed(self, events: List[Event]) -> List[Event]:
        if not all(isinstance(event, Event) for event in events):
            raise ValueError('bad type when initializing DGStorage from events list')
        if all(events[i].t <= events[i + 1].t for i in range(len(events) - 1)):
            return events
        warnings.warn('received a non-chronological list of events, sorting by time')
        events.sort(key=lambda x: x.t)
        return events

    def _check_node_feature_shapes(self, events: List[Event]) -> Optional[torch.Size]:
        shape = None
        for event in events:
            if isinstance(event, NodeEvent) and event.features is not None:
                if shape is None:
                    shape = event.features.shape
                elif shape != event.features.shape:
                    raise ValueError(
                        f'Node feature shapes non-homogenous: {shape} != {event.features.shape}'
                    )
        return shape

    def _check_edge_feature_shapes(self, events: List[Event]) -> Optional[torch.Size]:
        shape = None
        for event in events:
            if isinstance(event, EdgeEvent) and event.features is not None:
                if shape is None:
                    shape = event.features.shape
                elif shape != event.features.shape:
                    raise ValueError(
                        f'Edge feature shapes non-homogenous: {shape} != {event.features.shape}'
                    )
        return shape
