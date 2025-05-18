from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Protocol, Union

import numpy as np
import torch

from opendg._storage import DGSliceTracker
from opendg.graph import DGBatch, DGraph


class DGHook(Protocol):
    r"""The behaviours to be executed on a DGraph before materializing."""

    def __call__(self, dg: Union[DGraph, DGBatch]) -> DGBatch: ...

    def is_neighbor_sampler(self) -> bool: ...

    def is_negative_sampler(self) -> bool: ...


class HookManager:
    r"""A manager to handle multiple hooks.

    Args:
        hooks (List[DGHook]): A list of hooks to be
    """

    def __init__(self, hooks: List[DGHook]) -> None:
        self.hooks = hooks
        # * negative sampler hooks must appear before neighbor sampler hooks in order for batch.neg to be set
        self.hooks.sort(
            key=lambda x: x.is_negative_sampler(), reverse=True
        )  # start with negative sampler hooks

    def __call__(self, dg: DGraph) -> DGBatch:
        r"""Apply all hooks to the DGraph and return the materialized batch."""
        batch = dg.materialize()
        batch.num_nodes = dg.num_nodes
        batch.edge_feats_dim = dg.edge_feats_dim
        for hook in self.hooks:
            batch = hook(batch)
        return batch

    def is_neighbor_sampler(self) -> bool:
        return False

    def is_negative_sampler(self) -> bool:
        return False


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
    def __call__(self, dg: Union[DGraph, DGBatch]) -> DGBatch:
        if isinstance(dg, DGraph):
            batch = dg.materialize()
        else:
            batch = dg
        size = (round(self.neg_sampling_ratio * batch.dst.size(0)),)
        batch.neg = torch.randint(self.low, self.high, size)  # type: ignore
        return batch

    def is_negative_sampler(self) -> bool:
        return True

    def is_neighbor_sampler(self) -> bool:
        return False


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
        batch = dg.materialize(materialize_features=False)

        # TODO: Compose hooks
        #! due to the use of dg._storage.get_nbrs, can't become a composite hook at this stage
        self.neg_sampling_ratio = 1.0
        self.low = 0
        self.high = dg.num_nodes
        size = (round(self.neg_sampling_ratio * batch.dst.size(0)),)
        batch.neg = torch.randint(self.low, self.high, size, dtype=torch.long)  # type: ignore

        batch.nids, batch.nbr_nids, batch.nbr_times, batch.nbr_feats, batch.nbr_mask = (  # type: ignore
            dg._storage.get_nbrs(
                seed_nodes=torch.cat([batch.src, batch.dst, batch.neg]),  # type: ignore
                num_nbrs=self.num_nbrs,
                slice=DGSliceTracker(end_idx=dg._slice.end_idx),
            )
        )
        return batch

    def is_negative_sampler(self) -> bool:
        return False

    def is_neighbor_sampler(self) -> bool:
        return True


class TGBNeighborSamplerHook:
    r"""Load data from DGraph using pre-generated TGB negative samples.
    Make sure to perform `dataset.load_val_ns()` or `dataset.load_test_ns()` before using this hook.

    Args:
        neg_sampler (object): The negative sampler object to use for sampling.

    Raises:
        ValueError: If neg_sampler is not provided.
    """

    def __init__(self, neg_sampler: object, split_mode: str) -> None:
        if neg_sampler is None:
            raise ValueError('neg_sampler must be provided')
        if split_mode not in ['val', 'test']:
            raise ValueError('split_mode must be one of val, test')
        if neg_sampler.eval_set[split_mode] is None:
            raise ValueError(
                f'please run load_{split_mode}_ns() before using this hook'
            )

        self.neg_sampler = neg_sampler
        self.split_mode = split_mode

    def __call__(self, dg: Union[DGraph, DGBatch]) -> DGBatch:
        if isinstance(dg, DGraph):
            batch = dg.materialize()
        else:
            batch = dg
        neg_batch_list = self.neg_sampler.query_batch(
            batch.src, batch.dst, batch.time, split_mode=self.split_mode
        )
        batch.neg_batch_list = neg_batch_list
        queries = []
        for idx, neg_batch in enumerate(neg_batch_list):
            queries.append(neg_batch)
        unique_neg = np.unique(np.concatenate(queries))
        batch.neg = torch.tensor(unique_neg, dtype=torch.long)  # type: ignore
        return batch

    def is_negative_sampler(self) -> bool:
        return True

    def is_neighbor_sampler(self) -> bool:
        return False


class RecencyNeighborHook:
    r"""Load neighbors from DGraph using a recency sampling. Each node maintains a fixed number of recent neighbors.

    Args:
        num_nodes (int): Total number of nodes to track.
        num_nbrs (List[int]): Number of neighbors to sample at each hop (max neighbors to keep).
        gen_neg (bool): Whether to generate negative samples (default = True).

    Raises:
        ValueError: If the num_nbrs list is empty.
    """

    def __init__(
        self,
        num_nodes: int,
        num_nbrs: List[int],
        gen_neg: bool = True,
    ) -> None:
        if not len(num_nbrs):
            raise ValueError('num_nbrs must be non-empty')
        if not all([isinstance(x, int) and (x > 0) for x in num_nbrs]):
            raise ValueError('Each value in num_nbrs must be a positive integer')
        if len(num_nbrs) > 1:
            raise ValueError('RecencyNeighborSamplerHook only supports 1 hop for now')
        self._num_nbrs = num_nbrs
        self._nbrs: Dict[int, List[Deque[Any]]] = {}
        self._gen_neg = gen_neg
        #! [Andy] maybe should be called total_nodes instead of num_nodes
        for node in range(num_nodes):
            self._nbrs[node] = [deque(maxlen=num_nbrs[0]) for _ in range(3)]

    @property
    def num_nbrs(self) -> List[int]:
        return self._num_nbrs

    def is_neighbor_sampler(self) -> bool:
        return True

    def is_negative_sampler(self) -> bool:
        return False

    def __call__(self, dg: Union[DGraph, DGBatch]) -> DGBatch:
        if isinstance(dg, DGraph):
            batch = dg.materialize()
            edge_feats_dim = dg.edge_feats_dim
        else:
            batch = dg
            edge_feats_dim = batch.edge_feats_dim
        if self._gen_neg and (not hasattr(batch, 'neg')):
            self.neg_sampling_ratio = 1.0
            self.low = 0
            if isinstance(dg, DGraph):
                num_nodes = dg.num_nodes
            else:
                num_nodes = batch.num_nodes
            self.high = num_nodes
            size = (round(self.neg_sampling_ratio * batch.dst.size(0)),)
            batch.neg = torch.randint(self.low, self.high, size, dtype=torch.long)  # type: ignore

        if hasattr(batch, 'neg'):
            seed_nodes = torch.cat([batch.src, batch.dst, batch.neg])  # type: ignore
        else:
            seed_nodes = torch.cat([batch.src, batch.dst])
        unique, inverse_indices = seed_nodes.unique(return_inverse=True)

        batch_size = len(seed_nodes)
        nbr_nids = torch.empty(batch_size, self._num_nbrs[0], dtype=torch.long)
        nbr_times = torch.empty(batch_size, self._num_nbrs[0], dtype=torch.long)
        nbr_feats = torch.zeros(batch_size, self._num_nbrs[0], edge_feats_dim)  # type: ignore
        nbr_mask = torch.zeros(batch_size, self._num_nbrs[0], dtype=torch.long)
        for i, node in enumerate(unique.tolist()):
            if nn := len(self._nbrs[node][0]):
                mask = inverse_indices == i
                nbr_nids[mask, :nn] = torch.LongTensor(self._nbrs[node][0])
                nbr_times[mask, :nn] = torch.LongTensor(self._nbrs[node][1])
                nbr_feats[mask, :nn] = torch.stack(list(self._nbrs[node][2]))  # TODO:
                nbr_mask[mask, :nn] = nn >= self._num_nbrs[0]

        batch.nids = [seed_nodes]  # type: ignore
        batch.nbr_nids = [nbr_nids]  # type: ignore
        batch.nbr_times = [nbr_times]  # type: ignore
        batch.nbr_feats = [nbr_feats]  # type: ignore
        batch.nbr_mask = [nbr_mask]  # type: ignore

        self._update(batch)
        return batch

    def _update(self, batch: DGBatch) -> None:
        for i in range(batch.src.size(0)):
            src_nbr = int(batch.src[i].item())
            dst_nbr = int(batch.dst[i].item())
            time = batch.time[i].item()

            self._nbrs[src_nbr][0].append(dst_nbr)
            self._nbrs[src_nbr][1].append(time)
            self._nbrs[dst_nbr][0].append(src_nbr)
            self._nbrs[dst_nbr][1].append(time)
            if batch.edge_feats is not None:
                self._nbrs[src_nbr][2].append(batch.edge_feats[i])
                self._nbrs[dst_nbr][2].append(batch.edge_feats[i])
