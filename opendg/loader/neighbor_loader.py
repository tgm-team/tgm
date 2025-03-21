from __future__ import annotations

from typing import List

from opendg.graph import DGraph
from opendg.loader.base import DGBaseLoader


class DGNeighborLoader(DGBaseLoader):
    r"""Load data from DGraph using a memory based sampling function.

    Args:
        dg (DGraph): The dynamic graph to iterate.
        num_nbrs (List[int]): Number of neighbors to sample at each hop (-1 to keep all)
        batch_size (int): The batch size to yield at each iteration.
        batch_unit (str): The unit corresponding to the batch_size.
        drop_last (bool): Set to True to drop the last incomplete batch.
        materialize (bool): If True (default), auto-materialize the DGraph before yielding a batch.

    Raises:
        ValueError: If the num_nbrs list is empty.
        ValueError: If the batch_unit and dg time unit are not both ordered or both not ordered.
        ValueError: If the batch_unit and dg time unit are both ordered but the graph is coarser than the batch.
    """

    def __init__(
        self,
        dg: DGraph,
        num_nbrs: List[int],
        batch_size: int = 1,
        batch_unit: str = 'r',
        drop_last: bool = False,
        materialize: bool = True,
    ) -> None:
        if not len(num_nbrs):
            raise ValueError('num_nbrs must be non-empty')
        if not all([isinstance(x, int) and (x == -1 or x > 0) for x in num_nbrs]):
            raise ValueError('Each value in num_nbrs must be a positive integer or -1')
        self._num_nbrs = num_nbrs
        super().__init__(dg, batch_size, batch_unit, drop_last, materialize)

    @property
    def num_hops(self) -> int:
        return len(self.num_nbrs)

    @property
    def num_nbrs(self) -> List[int]:
        return self._num_nbrs

    def pre_yield(self, batch: DGraph) -> DGraph:
        # Dict[int, List[List[Tuple[int, int]]]]
        nbrs = self._dg._storage.get_nbrs(
            seed_nodes=list(batch.nodes),
            num_nbrs=self.num_nbrs,
            end_time=batch.end_time,
        )
        # TODO: Extract subgraph with nbrs
        return batch
