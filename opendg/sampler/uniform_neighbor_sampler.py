from typing import Any, List, Union

from opendg.graph import DGraph

from .neighbor_sampler import DGNeighborSampler


class DGUniformNeighborSampler(DGNeighborSampler):
    r"""In memory neighborhood sampling using uniform weight with piecewise cutoff sampling probability.

    Args:
        dg (DGraph): The original dynamic graph to sample from.
        num_neighbors (List[int]): The number of neighbors to sample at each hop.
        **kwargs (Any): Additional keyworc arguments for the decayed sampling strategy.
    """

    def __init__(
        self,
        dg: DGraph,
        num_neighbors: Union[int, List[int]],
        **kwargs: Any,
    ) -> None:
        super().__init__(dg, num_neighbors, sampling_func='uniform', **kwargs)
