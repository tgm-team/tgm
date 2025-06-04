from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from torch import Tensor

from opendg.data import DGData


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
    def __init__(self, data: DGData) -> None: ...

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
    def get_dynamic_node_feats(self, slice: DGSliceTracker) -> Optional[Tensor]: ...

    @abstractmethod
    def get_edge_feats(self, slice: DGSliceTracker) -> Optional[Tensor]: ...

    @abstractmethod
    def get_dynamic_node_feats_dim(self) -> Optional[int]: ...

    @abstractmethod
    def get_edge_feats_dim(self) -> Optional[int]: ...

    @abstractmethod
    def get_nbrs(
        self,
        seed_nodes: Tensor,
        num_nbrs: List[int],
        slice: DGSliceTracker,
    ) -> Tuple[List[Tensor], ...]: ...
