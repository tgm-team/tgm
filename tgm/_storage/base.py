from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Set, Tuple

from torch import Tensor

from tgm.data import DGData


@dataclass(slots=True)
class DGSliceTracker:
    """Tracks a temporal or event-based slice within a dynamic graph."""

    start_time: Optional[int] = None
    end_time: Optional[int] = None
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None


class DGStorageBase(ABC):
    """Base class for dynamic graph storage engines."""

    @abstractmethod
    def __init__(self, data: DGData) -> None:
        """Initialize the storage engine from DGData."""
        ...

    @abstractmethod
    def get_start_time(self, slice: DGSliceTracker) -> Optional[int]:
        """Return the first timestamp in the slice, or None if empty."""
        ...

    @abstractmethod
    def get_end_time(self, slice: DGSliceTracker) -> Optional[int]:
        """Return the last timestamp in the slice, or None if empty."""
        ...

    @abstractmethod
    def get_nodes(self, slice: DGSliceTracker) -> Set[int]:
        """Return the set of nodes present in the slice."""
        ...

    @abstractmethod
    def get_edges(self, slice: DGSliceTracker) -> Tuple[Tensor, Tensor, Tensor]:
        """Return (src, dst, time) tensors for edges in the slice."""
        ...

    @abstractmethod
    def get_num_timestamps(self, slice: DGSliceTracker) -> int:
        """Return the number of unique timestamps in the slice."""
        ...

    @abstractmethod
    def get_num_events(self, slice: DGSliceTracker) -> int:
        """Return the total number of events in the slice."""
        ...

    @abstractmethod
    def get_dynamic_node_feats(self, slice: DGSliceTracker) -> Optional[Tensor]:
        """Return dynamic node features as a sparse COO tensor within the slice, if any."""
        ...

    @abstractmethod
    def get_edge_feats(self, slice: DGSliceTracker) -> Optional[Tensor]:
        """Return edge features as a sparse COO tensor within the slice, if any."""
        ...

    @abstractmethod
    def get_static_node_feats(self) -> Optional[Tensor]:
        """Return static node features as a sparse COO tensor within the slice, if any."""
        ...

    @abstractmethod
    def get_dynamic_node_feats_dim(self) -> Optional[int]:
        """Return dimension of dynamic node features, if any."""
        ...

    @abstractmethod
    def get_edge_feats_dim(self) -> Optional[int]:
        """Return dimension of edge features, if any."""
        ...

    @abstractmethod
    def get_static_node_feats_dim(self) -> Optional[int]:
        """Return dimension of static node features, if any."""
        ...

    @abstractmethod
    def get_nbrs(
        self,
        seed_nodes: Tensor,
        num_nbrs: int,
        slice: DGSliceTracker,
        directed: bool = False,
    ) -> Tuple[Tensor, ...]:
        """Return neighbors for the given seed nodes within the slice.

        Args:
            seed_nodes: Tensor of node ids to query neighbors for.
            num_nbrs: Number of neighbors to sample per node.
            slice: The temporal/event slice to consider.
            directed (bool): If true, aggregates interactions in src->dst direction only (default=False).

        Returns:
            (nbr_nids, nbr_times, nbr_feats) tensors containing the relevant neighborhood
            information, padded using tgm.constants.PADDED_NODE_ID.
        """
        ...
