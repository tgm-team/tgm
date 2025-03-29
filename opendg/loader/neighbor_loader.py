from __future__ import annotations

from typing import Any, List

from opendg._storage import DGSliceTracker
from opendg.graph import DGraph
from opendg.loader.dataloader import DGDataLoader


class DGNeighborLoader(DGDataLoader):
    r"""Load data from DGraph using a memory based sampling function.

    Args:
        dg (DGraph): The dynamic graph to iterate.
        num_nbrs (List[int]): Number of neighbors to sample at each hop (-1 to keep all)
        **kwargs (Any): Additional arguments to the DGDataLoader

    Raises:
        ValueError: If the num_nbrs list is empty.
    """

    def __init__(
        self,
        dg: DGraph,
        num_nbrs: List[int],
        **kwargs: Any,
    ) -> None:
        if not len(num_nbrs):
            raise ValueError('num_nbrs must be non-empty')
        if not all([isinstance(x, int) and (x == -1 or x > 0) for x in num_nbrs]):
            raise ValueError('Each value in num_nbrs must be a positive integer or -1')
        self._num_nbrs = num_nbrs
        super().__init__(dg, **kwargs)

    @property
    def num_hops(self) -> int:
        return len(self.num_nbrs)

    @property
    def num_nbrs(self) -> List[int]:
        return self._num_nbrs

    def pre_yield(self, batch: DGraph) -> DGraph:
        slice = DGSliceTracker(
            end_time=batch.start_time, end_idx=batch._slice.start_idx
        )
        nbrs = self._dg._storage.get_nbrs(
            seed_nodes=batch.nodes, num_nbrs=self.num_nbrs, slice=slice
        )
        temporal_nbrhood = batch.nodes
        for seed_nbrhood in nbrs.values():
            for node, _ in seed_nbrhood[-1]:  # Only care about final hop
                temporal_nbrhood.add(node)  # Don't care about time info either

        if self._iterate_by_time:
            batch = self._dg.slice_time(end_time=batch.end_time)
        else:
            batch = self._dg.slice_events(end_idx=batch._slice.end_idx)
        batch = batch.slice_nodes(list(temporal_nbrhood))
        return batch
