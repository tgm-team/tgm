from typing import Any, List

import torch

from opendg.graph import DGraph


class TemporalLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges DGraph events to a mini-batch.

    Args:
        dg (DGraph): The dynamic graph from which to load the data.
        batch_size (int, optional): How many samples per batch to load (default = 1).
        **kwargs (optional): Additional arguments to torch.utils.data.DataLoader.
    """

    def __init__(self, dg: DGraph, batch_size: int = 1, **kwargs: Any):
        self._dg = dg
        self._batch_size = batch_size

        # TODO: Initialize temporal batches properly
        ts_batch = range(0, len(dg), batch_size)
        super().__init__(ts_batch, 1, shuffle=False, collate_fn=self, **kwargs)

    def __call__(self, ts_batch: List[int]) -> DGraph:
        # TODO: Slice time
        return self._slice_time[ts_batch[0] : ts_batch[0] + self._batch_size]
