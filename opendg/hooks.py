from __future__ import annotations

from typing import List, Protocol

import torch

from opendg._storage import DGSliceTracker
from opendg.graph import DGBatch, DGraph


class DGHook(Protocol):
    r"""The behaviours to be executed on a DGraph before materializing."""

    def __call__(self, dg: DGraph) -> DGBatch: ...


class NegativeEdgeSamplerHook:
    r"""Sample negative edges for dynamic link prediction.

    Args:
        low (int): The minimum node id to sample
        high (int) : The maximum node id to sample
        neg_sampling_ratio (float): The ratio of sampled negative destination nodes
            to the number of positive destination nodes (default = 1.0).
    """

    def __init__(self, low: int, high: int, neg_sampling_ratio: float = 1.0) -> None:
        self.low = low
        self.high = high
        self.neg_sampling_ratio = neg_sampling_ratio

    # TODO: Historical vs. random
    def __call__(self, dg: DGraph) -> DGBatch:
        batch = dg.materialize()
        size = (round(self.neg_sampling_ratio * batch.dst.size(0)),)
        batch.neg = torch.randint(self.low, self.high, size)  # type: ignore
        return batch


class NeighborSamplerHook:
    r"""Load data from DGraph using a memory based sampling function.

    Args:
        num_nbrs (List[int]): Number of neighbors to sample at each hop (-1 to keep all)
        **kwargs (Any): Additional arguments to the DGDataLoader

    Raises:
        ValueError: If the num_nbrs list is empty.
    """

    def __init__(self, num_nbrs: List[int]) -> None:
        if not len(num_nbrs):
            raise ValueError('num_nbrs must be non-empty')
        if not all([isinstance(x, int) and (x == -1 or x > 0) for x in num_nbrs]):
            raise ValueError('Each value in num_nbrs must be a positive integer or -1')
        self._num_nbrs = num_nbrs

    @property
    def num_nbrs(self) -> List[int]:
        return self._num_nbrs

    def __call__(self, dg: DGraph) -> DGBatch:
        slice = DGSliceTracker(end_time=dg.start_time, end_idx=dg._slice.start_idx)
        nbrs = dg._storage.get_nbrs(
            seed_nodes=dg.nodes, num_nbrs=self.num_nbrs, slice=slice
        )
        temporal_nbrhood = dg.nodes
        for seed_nbrhood in nbrs.values():
            for node, _ in seed_nbrhood[-1]:  # Only care about final hop
                temporal_nbrhood.add(node)  # Don't care about time info either

        # TODO: Verify we don't need the original graph!!!!
        # batch = self._dg.slice_events(end_idx=batch._slice.end_idx)
        # batch = batch.slice_nodes(list(temporal_nbrhood))
        # if self._iterate_by_time: # TODO: We need to store info about whether we are iterating by time or events
        # batch = self._dg.slice_time(end_time=batch.end_time)
        dg._slice = DGSliceTracker(
            end_idx=dg._slice.start_idx, node_slice=temporal_nbrhood
        )
        return dg.materialize()
