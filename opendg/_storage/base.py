import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from torch import Tensor

from opendg.events import EdgeEvent, Event, NodeEvent


@dataclass(slots=True)
class DGSliceTracker:
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None
    node_slice: Optional[Set[int]] = None


class DGStorageBase(ABC):
    r"""Base class for dynamic graph storage engine."""

    @abstractmethod
    def __init__(self, events: List[Event]) -> None: ...

    @abstractmethod
    def to_events(self, slice: DGSliceTracker) -> List[Event]: ...

    @abstractmethod
    def get_start_time(self, slice: DGSliceTracker) -> Optional[int]: ...

    @abstractmethod
    def get_end_time(self, slice: DGSliceTracker) -> Optional[int]: ...

    @abstractmethod
    def get_nodes(self, slice: DGSliceTracker) -> Set[int]: ...

    @abstractmethod
    def get_edges(self, slice: DGSliceTracker) -> Tuple[Tensor, Tensor, Tensor]: ...

    @abstractmethod
    def get_num_timestamps(self, slice: DGSliceTracker) -> int: ...

    @abstractmethod
    def get_num_events(self, slice: DGSliceTracker) -> int: ...

    @abstractmethod
    def get_node_feats(self, slice: DGSliceTracker) -> Optional[Tensor]: ...

    @abstractmethod
    def get_edge_feats(self, slice: DGSliceTracker) -> Optional[Tensor]: ...

    @abstractmethod
    def get_node_feats_dim(self) -> Optional[int]: ...

    @abstractmethod
    def get_edge_feats_dim(self) -> Optional[int]: ...

    @abstractmethod
    def get_nbrs(
        self,
        seed_nodes: Tensor,
        num_nbrs: List[int],
        slice: DGSliceTracker,
    ) -> Tuple[List[Tensor], ...]: ...

    def _sort_events_list_if_needed(self, events: List[Event]) -> List[Event]:
        if not all(isinstance(event, Event) for event in events):
            raise ValueError('bad type when initializing DGStorage from events list')
        if all(events[i].t <= events[i + 1].t for i in range(len(events) - 1)):
            return events
        warnings.warn('received a non-chronological list of events, sorting by time')
        events.sort(key=lambda x: x.t)
        return events

    def _check_feature_dims(
        self, events: List[Event]
    ) -> Tuple[Optional[int], Optional[int]]:
        node_shape, edge_shape = None, None
        for event in events:
            feats = event.features
            if isinstance(event, NodeEvent) and feats is not None:
                if (node_shape := node_shape or feats.shape) != feats.shape:
                    raise ValueError(f'Node feat mismatch: {node_shape}!={feats.shape}')
            if isinstance(event, EdgeEvent) and feats is not None:
                if (edge_shape := edge_shape or feats.shape) != feats.shape:
                    raise ValueError(f'Edge feat mismatch: {edge_shape}!={feats.shape}')
        if node_shape is not None and len(node_shape) > 1:
            raise ValueError(f'Only 1-d node features supported, got: ({node_shape})')
        if edge_shape is not None and len(edge_shape) > 1:
            raise ValueError(f'Only 1-d edge features supported, got: ({edge_shape})')
        node_dim = node_shape[0] if node_shape is not None else node_shape
        edge_dim = edge_shape[0] if edge_shape is not None else edge_shape
        return node_dim, edge_dim
