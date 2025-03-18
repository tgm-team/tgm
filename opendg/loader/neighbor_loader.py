from typing import List

from opendg.graph import DGraph
from opendg.loader.base import DGBaseLoader
from opendg.timedelta import TimeDeltaUnit


class DGNeighborLoader(DGBaseLoader):
    r"""Load data from DGraph using a memory based sampling function.

    Args:
        dg (DGraph): The dynamic graph to iterate.
        num_nbrs (List[int]): The number of hops to sample, and the number of neighbors to sample at each hop (-1 to keep all)
        batch_size (int): The batch size to yield at each iteration.
        batch_unit (str): The unit corresponding to the batch_size.
        drop_last (bool): Set to True to drop the last incomplete batch.

    Raises:
        ValueError: If the num_nbrs list is empty.
        ValueError: If any of the values in num_nbrs is not a positive integer or -1.
        ValueError: If the batch_unit is not a valid TimeDeltaUnit type.
        ValueError: If the batch_size is not a positive integer.
        ValueError: If the batch_unit is not TimeDeltaUnit.ORDERED and the graph is TimeDeltaUnit.ORDERED.
        ValueError: If both the batch_unit and the graph are not TimeDeltaUnit.ORDERED, but the graph TimeDelta
                    is coarser than the batch TimeDelta. In this case, there is ambiguity in how to yield
                    events due to loss of information.

    Note:
        Ordered batch_unit ('TimeDeltaUnit.ORDERED) iterates using normal batch size semantics
        in which case each yielded batch has a constant number of events (except the last batch if drop_last=False).
            e.g. batch_size=5, batch_unit=TimeDeltaUnit.ORDERED -> yield 5 events at time

        Unordered batch_unit iterates uses the appropriate temporal unit in which case each yielded
        batch may have different number of events but has the same temporal length (except the last batch if drop_last=False).
            e.g. batch_size=5, batch_unit=TimeDeltaUnit.SECONDS -> yield 5 seconds of data at a time

        When using the ordered batch_unit, the order of yielded _events within the same timestamp
        is non-deterministic.
    """

    def __init__(
        self,
        dg: DGraph,
        num_nbrs: List[int],
        batch_size: int = 1,
        batch_unit: str = TimeDeltaUnit.ORDERED,
        drop_last: bool = False,
    ) -> None:
        if not len(num_nbrs):
            raise ValueError('num_nbrs must be non-empty')

        if not all([isinstance(x, int) and (x == -1 or x > 0) for x in num_nbrs]):
            raise ValueError(
                'Each value in num_nbrs must be a positive integer or -1 to keep all neighbors'
            )
        self._num_nbrs = num_nbrs
        super().__init__(dg, batch_size, batch_unit, drop_last)

    @property
    def num_hops(self) -> int:
        r"""The number of hops that the sampler is using."""
        return len(self.num_nbrs)

    @property
    def num_nbrs(self) -> List[int]:
        r"""The number of hops to sample, and the number of neighbors to sample at each hop."""
        return self._num_nbrs

    def sample(self, batch: 'DGraph') -> 'DGraph':
        r"""DGDataLoader performs no subsampling. Returns the full batch.

        Args:
            batch (DGraph): Incoming batch of data. May not be materialized.

        Returns:
            (DGraph): The input batch of data.
        """
        # TODO: Need a way to easily get nodes in the graph
        batch_nodes = self._dg._storage.get_nodes(batch.start_time, batch.end_time)

        # Dict[int, List[List[Tuple[int, int]]]]
        nbrs = self._dg._storage.get_nbrs(
            seed_nodes=list(batch_nodes),
            num_nbrs=self.num_nbrs,
            end_time=batch.start_time,
        )
        # TODO: Extract subgraph with nbrs
        return batch
