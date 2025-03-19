from abc import ABC, abstractmethod

from opendg.graph import DGraph
from opendg.timedelta import TimeDeltaDG


class DGBaseLoader(ABC):
    r"""Base class iterator over a DGraph.

    Args:
        dg (DGraph): The dynamic graph to iterate.
        batch_size (int): The batch size to yield at each iteration.
        batch_unit (str): The unit corresponding to the batch_size.
        drop_last (bool): Set to True to drop the last incomplete batch.

    Raises:
        ValueError: If the batch_unit is not a valid TimeDeltaDG unit.
        ValueError: If the batch_size is not a positive integer.
        ValueError: If the batch_unit and dg time unit are not both ordered or both not ordered.
        ValueError: If the batch_unit and dg time unit are both ordered but the graph is coarser than the batch.
    """

    def __init__(
        self,
        dg: DGraph,
        batch_size: int = 1,
        batch_unit: str = 'r',
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f'batch_size must be > 0 but got {batch_size}')
        if not len(dg):
            raise ValueError('Cannot iterate an empty DGraph')

        self._dg = dg
        self._batch_size = batch_size
        self._batch_unit = batch_unit
        self._drop_last = drop_last
        dg_is_ordered = self._dg.time_delta.is_ordered
        batch_unit_is_ordered = self._batch_unit == 'r'

        if dg_is_ordered and not batch_unit_is_ordered:
            raise ValueError('Cannot iterate ordered dg using non-ordered batch_unit')
        if not dg_is_ordered and batch_unit_is_ordered:
            raise ValueError('Cannot iterate non-ordered dg using ordered batch_unit')

        if not dg_is_ordered and not batch_unit_is_ordered:
            # Check to ensure the graph time unit is smaller (more granular) than batch time unit.
            # If this is not the case, temporal iteration losses information, so we throw an Exception.
            batch_time_delta = TimeDeltaDG(self._batch_unit, value=self._batch_size)
            if self._dg.time_delta.is_coarser_than(batch_time_delta):
                raise ValueError(
                    f'Tried to construct a data loader on a DGraph with time delta: {self._dg.time_delta} '
                    f'which is strictly coarser than the batch_unit: {batch_unit}, batch_size: {batch_size}. '
                    'Cannot iterate a non-ordered DGraph with a more granular batch_unit due to loss of informmation. '
                    'Either choose a larger batch size, larger batch unit or consider iterate using ordered batching.'
                )

            self._batch_size = int(batch_time_delta.convert(self._dg.time_delta))

        assert self._dg.start_time is not None
        self._idx = self._dg.start_time

    @abstractmethod
    def sample(self, batch: 'DGraph') -> 'DGraph':
        r"""Downsample a given temporal batch, using neighborhood information, for instance.

        Args:
            batch (DGraph): Incoming batch of data. May not be materialized.

        Returns:
            (DGraph): Downsampled batch of data. Must be naterialized.
        """

    def __iter__(self) -> 'DGBaseLoader':
        return self

    def __next__(self) -> 'DGraph':
        if self._done_iteration():
            raise StopIteration

        batch = self._dg.slice_time(self._idx, self._idx + self._batch_size - 1)
        self._idx += self._batch_size
        return self.sample(batch)

    def _done_iteration(self) -> bool:
        if not len(self._dg):
            return True

        if self._drop_last:
            check_idx = self._idx + self._batch_size - 1
        else:
            check_idx = self._idx

        assert self._dg.end_time is not None
        return check_idx >= self._dg.end_time + 1
