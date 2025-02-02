from abc import ABC, abstractmethod
from typing import Any, List, Optional

from opendg.graph import DGraph


class BaseDGSampler(ABC):
    r"""Base class for sampling from a dynamic graph."""

    @abstractmethod
    def sample(
        self,
        seed_nodes: Optional[List[int]] = None,
        from_timestamp: Optional[int] = None,
        until_timestamp: Optional[int] = None,
        *kwargs: Any,
    ) -> DGraph:
        r"""Sample seed nodes from a dynamic graph in the temporal window [from_timestamp, until_timestamp].

        Args:
            seed_nodes (Optional[List[int]]): The indices of the seed nodes to sample from, or the entire graph if None.
            from_timestamp (Optional[int]): The minimum time to sampling from, or the dynamic graph start time if None.
            until_timestamp (Optional[int]): The maximum time to sampling until, or the dynamic graph start time if None.
            kwargs (Any): Optional sampler specific keyword arguments.

        Returns:
            A subgraph view on the DGraph according to the sampling strategy and input arguments.
        """

    @property
    @abstractmethod
    def num_neighbours(self) -> List[int]:
        r"""The number of neighbours to sampler for each hop."""

    @property
    @abstractmethod
    def num_hops(self) -> int:
        r"""The number of hops specified in the sampler."""

    def __len__(self) -> int:
        return self.num_hops
