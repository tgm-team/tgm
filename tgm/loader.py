from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator, List, Literal

import torch

from tgm import DGBatch, DGraph
from tgm.hooks import HookManager
from tgm.timedelta import TimeDeltaDG


class _SkippableDataLoaderMixin(ABC):
    r"""Mixin to optionally skip or raise on empty batches."""

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
    def is_batch_empty(self, batch: Any) -> bool: ...

    def __iter__(self) -> Iterator[Any]:
        for batch in super().__iter__():  # type: ignore
            if self.is_batch_empty(batch):
                if self._on_empty == 'raise':
                    raise ValueError('Empty batch encountered')
                elif self._on_empty == 'skip':
                    continue
            yield batch


class DGDataLoader(_SkippableDataLoaderMixin, torch.utils.data.DataLoader):  # type: ignore
    r"""Iterate and materialize from a DGraph.

    Args:
        dg (DGraph): The dynamic graph to iterate.
        batch_size (int): The batch size to yield at each iteration.
        batch_unit (str): The unit corresponding to the batch_size.
        on_empty (Literal['skip', 'raise', None]): Action for empty batches.
        hook_manager (HookManager | None): Arbitrary transform behaviour to execute before materializing a batch.
        **kwargs (Any): Additional arguments to torch.utils.data.DataLoader.

    Raises:
        ValueError: If the batch_unit and dg time unit are not both ordered or both not ordered.
        ValueError: If the batch_unit and dg time unit are both ordered but the graph is coarser than the batch.
        ValueError: If an empty batch was encountered an on_empty='raise'.

    Note:
        The length returned by `len(DGDataLoader)` may be inaccurate when using a non-ordered
        `batch_unit` with `on_empty='skip'`. The reported length counts all batches, including
        those that would be skipped due to being empty.
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

        dg_ordered = dg.time_delta.is_ordered
        batch_ordered = batch_unit == 'r'

        if dg_ordered and not batch_ordered:
            raise ValueError('Cannot iterate ordered dg using non-ordered batch_unit')
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
        assert dg.start_time is not None and dg.end_time is not None

        self._dg = dg
        self._batch_size = batch_size
        self._hook_manager = (
            HookManager(dg.device) if hook_manager is None else hook_manager
        )
        self._slice_op = dg.slice_events if batch_ordered else dg.slice_time

        start_idx = 0 if batch_ordered else dg.start_time
        stop_idx = dg.num_events if batch_ordered else dg.end_time + 1

        if kwargs.get('drop_last', False):
            slice_start = range(start_idx, stop_idx - batch_size, batch_size)
        else:
            slice_start = range(start_idx, stop_idx, batch_size)

        super().__init__(
            slice_start, 1, shuffle=False, collate_fn=self, on_empty=on_empty, **kwargs
        )

    def __call__(self, slice_start: List[int]) -> DGBatch:
        slice_end = slice_start[0] + self._batch_size
        batch = self._slice_op(slice_start[0], slice_end)
        return self._hook_manager.execute_active_hooks(batch)

    def is_batch_empty(self, batch: DGBatch) -> bool:
        return batch.src.numel() == 0
