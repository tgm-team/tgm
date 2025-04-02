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


class NeighborSamplerHook:
    r"""Load data from DGraph using a memory based sampling function.

    Args:
        num_nbrs (List[int]): Number of neighbors to sample at each hop (-1 to keep all)

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
        batch = dg.materialize()
        nbrs = dg._storage.get_nbrs(
            seed_nodes=dg.nodes,
            num_nbrs=self.num_nbrs,
            slice=DGSliceTracker(end_idx=dg._slice.end_idx),
        )

        # TODO: This is terrible, should change get_nbrs to return the right format
        hop = 0
        seed_nodes = torch.cat([batch.src, batch.dst, batch.dst])
        nbr_nids = torch.empty(len(seed_nodes), self.num_nbrs[hop], dtype=torch.long)
        nbr_times = torch.empty(len(seed_nodes), self.num_nbrs[hop])
        nbr_feats = torch.zeros(len(seed_nodes), self.num_nbrs[hop], dg.edge_feats_dim)  # type: ignore
        nbr_mask = torch.zeros(len(seed_nodes), self.num_nbrs[hop])
        for batch_idx, node in enumerate(seed_nodes):
            for nbr_idx, (nbr_id, nbr_time) in enumerate(nbrs[node.item()][hop]):
                nbr_nids[batch_idx, nbr_idx] = nbr_id
                nbr_times[batch_idx, nbr_idx] = nbr_time
                nbr_mask[batch_idx, nbr_idx] = 1
                # nbr_feats[batch_idx, nbr_idx] = nbr_feat

        batch.nids = [seed_nodes]  # type: ignore
        batch.nbr_nids = [nbr_nids]  # type: ignore
        batch.nbr_times = [nbr_times]  # type: ignore
        batch.nbr_feats = [nbr_feats]  # type: ignore
        batch.nbr_mask = [nbr_mask]  # type: ignore
        return batch
