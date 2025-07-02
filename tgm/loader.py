from __future__ import annotations

from typing import Any, List

import torch

from tgm import DGBatch, DGraph
from tgm.hooks import DGHook, HookManager
from tgm.timedelta import TimeDeltaDG


class DGDataLoader(torch.utils.data.DataLoader):
    r"""Iterate and materialize from a DGraph.

    Args:
        dg (DGraph): The dynamic graph to iterate.
        batch_size (int): The batch size to yield at each iteration.
        batch_unit (str): The unit corresponding to the batch_size.
        hook (HookManager | Hook | List[Hook] | None): Arbitrary transform behaviour to execute before materializing a batch.
        **kwargs (Any): Additional arguments to torch.utils.data.DataLoader.

    Raises:
        ValueError: If the batch_unit and dg time unit are not both ordered or both not ordered.
        ValueError: If the batch_unit and dg time unit are both ordered but the graph is coarser than the batch.
    """

    def __init__(
        self,
        dg: DGraph,
        batch_size: int = 1,
        batch_unit: str = 'r',
        hook: HookManager | DGHook | List[DGHook] | None = None,
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
        assert dg.start_time is not None
        assert dg.end_time is not None

        self._dg = dg
        self._batch_size = batch_size
        self._hook = HookManager.from_any(dg, hook)
        self._slice_op = dg.slice_events if batch_ordered else dg.slice_time

        start_idx = 0 if batch_ordered else dg.start_time
        stop_idx = dg.num_events if batch_ordered else dg.end_time + 1

        if kwargs.get('drop_last', False):
            slice_start = range(start_idx, stop_idx - batch_size, batch_size)
        else:
            slice_start = range(start_idx, stop_idx, batch_size)
        super().__init__(slice_start, 1, shuffle=False, collate_fn=self, **kwargs)  # type: ignore

    def __call__(self, slice_start: List[int]) -> DGBatch:
        slice_end = slice_start[0] + self._batch_size
        batch = self._slice_op(slice_start[0], slice_end)
        return self._hook(batch)
