from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from torch import Tensor

from opendg.events import EdgeEvent, Event, NodeEvent
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
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        node_slice: Optional[Set[int]] = None,
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
    def append(self, events: Union[Event, List[Event]]) -> None:
        pass

    @abstractmethod
    def temporal_coarsening(
        self, time_delta: TimeDeltaDG, agg_func: str = 'sum'
    ) -> 'DGStorageBase':
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
