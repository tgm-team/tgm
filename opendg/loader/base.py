from __future__ import annotations

from abc import ABC, abstractmethod

from opendg.graph import DGBatch, DGraph
from opendg.timedelta import TimeDeltaDG


class DGBaseLoader(ABC):
    r"""Base class iterator over a DGraph.

    Args:
        dg (DGraph): The dynamic graph to iterate.
        batch_size (int): The batch size to yield at each iteration.
        batch_unit (str): The unit corresponding to the batch_size.
        drop_last (bool): Set to True to drop the last incomplete batch.
        materialize (bool): If True (default), auto-materialize the DGraph before yielding a batch.

    Raises:
        ValueError: If the batch_unit and dg time unit are not both ordered or both not ordered.
        ValueError: If the batch_unit and dg time unit are both ordered but the graph is coarser than the batch.
    """

    def __init__(
        self,
        dg: DGraph,
        batch_size: int = 1,
        batch_unit: str = 'r',
        drop_last: bool = False,
        materialize: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f'batch_size must be > 0 but got {batch_size}')
        if not len(dg):
            raise ValueError('Cannot iterate an empty DGraph')

        dg_ordered = dg.time_delta.is_ordered
        batch_ordered = batch_unit == 'r'

        if dg_ordered and not batch_ordered:
            raise ValueError('Cannot iterate ordered dg using non-ordered batch_unit')
        if not dg_ordered and batch_ordered:
            raise ValueError('Cannot iterate non-ordered dg using ordered batch_unit')
        if not dg_ordered and not batch_ordered:
            # Ensure the graph time unit is more granular than batch time unit.
            batch_time_delta = TimeDeltaDG(batch_unit, value=batch_size)
            if dg.time_delta.is_coarser_than(batch_time_delta):
                raise ValueError(
                    f'Tried to construct a data loader on a DGraph with time delta: {dg.time_delta} '
                    f'which is strictly coarser than the batch_unit: {batch_unit}, batch_size: {batch_size}. '
                    'Either choose a larger batch size, batch unit or consider iterate using ordered batching.'
                )
            batch_size = int(batch_time_delta.convert(dg.time_delta))

        # Warning: Cache miss
        assert dg.start_time is not None
        assert dg.end_time is not None

        self._dg = dg
        self._materialize = materialize
        self._batch_size = batch_size

        self._slice_op = dg.slice_events if batch_ordered else dg.slice_time
        self._idx = 0 if batch_ordered else dg.start_time
        self._stop_idx = 1_000_000 if batch_ordered else dg.end_time  # TODO
        if not drop_last:
            self._stop_idx += batch_size - 1

    @abstractmethod
    def pre_yield(self, batch: DGraph) -> DGraph:
        r"""Perform arbitary processing (e.g. neighborhood sampling) on the batch before yielding."""

    def __iter__(self) -> DGBaseLoader:
        return self

    def __next__(self) -> DGraph | DGBatch:
        slice_end = self._idx + self._batch_size - 1
        if slice_end > self._stop_idx:
            raise StopIteration

        batch = self._slice_op(self._idx, slice_end)
        self._idx += self._batch_size
        batch = self.pre_yield(batch)
        return batch.materialize() if self._materialize else batch
