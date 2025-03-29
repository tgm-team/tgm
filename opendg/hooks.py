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
        if not 0 < neg_sampling_ratio <= 1:
            raise ValueError('neg_sampling_ratio must be in (0, 1]')
        if not low < high:
            raise ValueError(f'low ({low}) must be strictly less than high ({high})')
        self.low = low
        self.high = high
        self.neg_sampling_ratio = neg_sampling_ratio

    # TODO: Historical vs. random
    def __call__(self, dg: DGraph) -> DGBatch:
        batch = dg.materialize()
        size = (round(self.neg_sampling_ratio * batch.dst.size(0)),)
        batch.neg = torch.randint(self.low, self.high, size)  # type: ignore
        return batch


class LastNeighborHook:
    r"""Keep track of last neighbor interactions.

    Args:
        num_nodes (int): Total number of nodes to track memory for
        size (int): The number of nbrs to track for each node in the memory.

    Raises:
        ValueError: If the num_nodes or size are non-positive.
    """

    def __init__(self, num_nodes: int, size: int) -> None:
        if num_nodes <= 0:
            raise ValueError('Number of total nodes must be strictly positive')
        if size <= 0:
            raise ValueError('Number of nbrs to track must be strictly positive')

        self._nbrs = torch.empty((num_nodes, size), dtype=torch.long)
        self._size = size

        self._cur_e_id = 0
        self._e_id = torch.empty((num_nodes, size), dtype=torch.long)
        self._assoc = torch.empty(num_nodes, dtype=torch.long)
        self.reset()

    def __call__(self, dg: DGraph) -> DGBatch:
        # TODO: What about negatives (compose hooks)
        batch = dg.materialize(materialize_features=False)
        batch_size = len(batch.src)
        batch_nbrs = torch.cat([batch.src, batch.dst], dim=0)
        batch_nodes = torch.cat([batch.dst, batch.src], dim=0)
        unique_batch_nodes = batch_nodes.unique()
        unique_batch_nodes_repeat = unique_batch_nodes.view(-1, 1).repeat(1, self._size)

        mask = (e_id := self._e_id[unique_batch_nodes]) > 0
        nbrs = self._nbrs[unique_batch_nodes][mask]
        n_id = torch.cat([unique_batch_nodes, nbrs]).unique()
        self._assoc[n_id] = torch.arange(n_id.size(0))

        batch.nbrs = self._assoc[nbrs]  # type: ignore
        batch.nodes = self._assoc[unique_batch_nodes_repeat[mask]]  # type: ignore
        batch.e_id = e_id[mask]  # type: ignore

        # Convert interaction ids to point to "dense" locations of shape [num_nodes, size]
        e_id = torch.arange(self._cur_e_id, self._cur_e_id + batch_size).repeat(2)
        self._cur_e_id += batch_size
        batch_nodes, perm = batch_nodes.sort()
        batch_nbrs, e_id = batch_nbrs[perm], e_id[perm]

        # Duplicate work?
        n_id = batch_nodes.unique()
        self._assoc[n_id] = torch.arange(n_id.numel())

        dense_id = torch.arange(batch_nodes.size(0)) % self._size
        dense_id += self._assoc[batch_nodes].mul_(self._size)

        dense_e_id = e_id.new_full((n_id.numel() * self._size,), -1)
        dense_e_id[dense_id] = e_id
        dense_e_id = dense_e_id.view(-1, self._size)

        dense_nbrs = e_id.new_empty(n_id.numel() * self._size)
        dense_nbrs[dense_id] = batch_nbrs
        dense_nbrs = dense_nbrs.view(-1, self._size)

        e_id = torch.cat([self._e_id[n_id, : self._size], dense_e_id], dim=-1)
        nbrs = torch.cat([self._nbrs[n_id, : self._size], dense_nbrs], dim=-1)
        self._e_id[n_id], perm = e_id.topk(self._size, dim=-1)
        self._nbrs[n_id] = torch.gather(nbrs, 1, perm)
        return batch

    def reset(self) -> None:
        self._cur_e_id = 0
        self._e_id.fill_(-1)


class NeighborSamplerHook:
    r"""Load data from DGraph using a memory based sampling function.

    Args:
        num_nbrs (List[int]): Number of nbrs to sample at each hop (-1 to keep all)

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
