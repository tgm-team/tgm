from typing import Any, Callable, List, Optional, Union

from opendg.graph import DGraph
from opendg.sampler.base import BaseDGSampler

from .base import BaseDGSampler
from .sampling_func import SamplingFunc, construct_sampling_func

SamplingFuncType = Union[
    str,
    SamplingFunc,
    Callable[[float], float],
]


class DGNeighborSampler(BaseDGSampler):
    r"""In memory neighborhood sampling using a specific sampling_func.

    Args:
        dg (DGraph): The original dynamic graph to sample from.
        num_neighbors (List[int]): The number of neighbors to sample at each hop.
        sampling_func (Union[str, SamplingFunc, Callable[[float], float]]): The temporal sampling function to apply.
    """

    def __init__(
        self,
        dg: DGraph,
        num_neighbors: Union[int, List[int]],
        sampling_func: SamplingFuncType,
        **kwargs: Any,
    ) -> None:
        self._dg = dg

        if isinstance(num_neighbors, int):
            num_neighbors = [num_neighbors]
        if not len(num_neighbors):
            raise ValueError(
                'Must specify the numberf of neighbors to sampler for at least 1 hop.'
            )
        self._num_neighbors = num_neighbors
        self._sampling_func = construct_sampling_func(sampling_func, **kwargs)

    def sample(
        self,
        seed_nodes: Optional[List[int]] = None,
        from_timestamp: Optional[int] = None,
        until_timestamp: Optional[int] = None,
    ) -> DGraph:
        # TODO
        return self._dg

    @property
    def num_neighbors(self) -> List[int]:
        r"""The number of neighbours to sampler for each hop."""
        return self._num_neighbors

    @property
    def num_hops(self) -> int:
        r"""The number of hops specified in the sampler."""
        return len(self.num_neighbors)

    def __len__(self) -> int:
        return self.num_hops
