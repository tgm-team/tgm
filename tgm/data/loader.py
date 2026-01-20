from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator, List, Literal

import torch

from tgm.core import DGBatch, DGraph, TimeDeltaDG
from tgm.exceptions import (
    EmptyBatchError,
    EventOrderedConversionError,
    InvalidDiscretizationError,
)
from tgm.hooks import HookManager
from tgm.util.logging import _get_logger

logger = _get_logger(__name__)


class _SkippableDataLoaderMixin(ABC):
    """Mixin to optionally skip or raise on empty batches.

    This mixin adds the ability to either skip or raise an error when an
    empty batch is encountered during iteration over a dataset.

    Args:
        on_empty (Literal['skip', 'raise', None], optional): Action to take
            on empty batches. 'skip' to silently skip, 'raise' to raise an error,
            None for no action. Defaults to None.

    Raises:
        ValueError: If `on_empty` is not one of 'skip', 'raise', or None.
    """

    def __init__(
        self,
        *args: Any,
        on_empty: Literal['skip', 'raise', None] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        valid_on_empty = ['skip', 'raise', None]
        if on_empty not in valid_on_empty:
            raise ValueError(
                f'Invalid on_empty={on_empty}, expected one of: {valid_on_empty}'
            )
        self._on_empty = on_empty

    @abstractmethod
    def _is_batch_empty(self, batch: Any) -> bool: ...

    def __iter__(self) -> Iterator[Any]:
        for batch in super().__iter__():  # type: ignore
            if self._is_batch_empty(batch):
                if self._on_empty == 'raise':
                    raise EmptyBatchError('Empty batch encountered')
                elif self._on_empty == 'skip':
                    logger.debug('Skipping empty batch')
                    continue
            yield batch


class DGDataLoader(_SkippableDataLoaderMixin, torch.utils.data.DataLoader):  # type: ignore
    """Iterate and materialize batches from a `DGraph`.

    This DataLoader supports both event-ordered and time-ordered temporal graphs.
    Optional hooks can be applied to each batch, and empty batches can be skipped
    or raise an exception depending on configuration.

    Args:
        dg (DGraph): The dynamic graph to iterate.
        batch_size (int, optional): The batch size to yield at each iteration.
        batch_unit (str, optional): The unit corresponding to the batch_size
            ('r' for event-ordered batches, or a time unit for time-ordered). Defaults to 'r'.
        on_empty (Literal['skip', 'raise', None], optional): Behavior for empty batches.
            'skip' to ignore, 'raise' to throw an error. Defaults to 'skip'.
        hook_manager (HookManager | None, optional): Optional hooks to apply
            transformations to each batch before returning. Defaults to None.
        **kwargs (Any): Additional arguments passed to `torch.utils.data.DataLoader`.

    Raises:
        ValueError: If `batch_size` <= 0.
        EventOrderedConversionError: If iterating an event-ordered DGraph using a time-ordered batch_unit.
        InvalidDiscretizationError: If a time-ordered DGraph has a time unit coarser than the batch_unit.
        EmptyBatchError: If an empty batch is encountered with on_empty='raise'.

    Note:
        - Event-ordered batching ('r') iterates sequentially over event indices.
          TIme-ordered batching iterates over temporal slices according to `batch_unit`.
        - For time-ordered batching, `batch_unit` must not be coarser than the DGraph
          time delta. Otherwise, a ValueError is raised.
        - The effective batch size may be adjusted internally when using time-ordered
          batching to match the graph's time granularity.
        - The length returned by `len(DGDataLoader)` may be inaccurate for time-ordered
          batches with `on_empty='skip'`, since skipped batches are still counted.
        - Slices and batch materialization return new `DGBatch` objects; underlying
          graph storage is not copied but views are used for efficiency.
    """

    def __init__(
        self,
        dg: DGraph,
        batch_size: int = 1,
        batch_unit: str = 'r',
        on_empty: Literal['skip', 'raise', None] = 'skip',
        hook_manager: HookManager | None = None,
        **kwargs: Any,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f'batch_size must be > 0 but got {batch_size}')

        batch_time_delta = TimeDeltaDG(batch_unit)
        logger.info(
            'Initializing DGDataLoader: batch_size=%d, batch_unit=%s',
            batch_size,
            batch_unit,
        )

        if dg.time_delta.is_event_ordered and batch_time_delta.is_time_ordered:
            raise EventOrderedConversionError(
                'Cannot iterate event-ordered dg using time-ordered batch_unit'
            )
        if dg.time_delta.is_time_ordered and batch_time_delta.is_time_ordered:
            # Ensure the graph time unit is more granular than batch time unit.
            batch_time_delta = TimeDeltaDG(batch_unit, value=batch_size)
            if dg.time_delta.is_coarser_than(batch_time_delta):
                raise InvalidDiscretizationError(
                    f'Tried to construct a data loader on a DGraph with time delta: {dg.time_delta} '
                    f'which is strictly coarser than the batch_unit: {batch_unit}, batch_size: {batch_size}. '
                    'Either choose a larger batch size, batch unit or consider iterate using event-ordered batching.'
                )
            batch_size = int(batch_time_delta.convert(dg.time_delta))

        # Warning: Cache miss
        assert dg.start_time is not None and dg.end_time is not None

        self._dg = dg
        self._batch_size = batch_size
        self._hook_manager = hook_manager

        if batch_time_delta.is_event_ordered:
            self._slice_op = dg.slice_events
            start_idx, stop_idx = 0, dg.num_events
        else:
            self._slice_op = dg.slice_time  # type: ignore
            start_idx, stop_idx = dg.start_time, dg.end_time + 1

        if kwargs.get('drop_last', False):
            slice_start = range(start_idx, stop_idx - batch_size, batch_size)
        else:
            slice_start = range(start_idx, stop_idx, batch_size)

        super().__init__(
            slice_start, 1, shuffle=False, collate_fn=self, on_empty=on_empty, **kwargs
        )

    def __call__(self, slice_start: List[int]) -> DGBatch:
        slice_end = slice_start[0] + self._batch_size
        dg = self._slice_op(slice_start[0], slice_end)
        batch = dg.materialize()
        if self._hook_manager is not None:
            logger.debug(
                'Applying hooks to batch %s [%d:%d)',
                self._slice_op.__name__,
                slice_start[0],
                slice_end,
            )
            batch = self._hook_manager.execute_active_hooks(dg, batch)
        return batch

    @property
    def dgraph(self) -> DGraph:
        return self._dg

    def _is_batch_empty(self, batch: DGBatch) -> bool:
        num_edge_events = batch.edge_src.numel()
        num_node_events = (
            batch.node_x_nids.numel() if batch.node_x_nids is not None else 0
        )
        num_node_labels = (
            batch.node_y_nids.numel() if batch.node_y_nids is not None else 0
        )
        return num_edge_events + num_node_events + num_node_labels == 0
