from __future__ import annotations

from queue import Queue
from typing import Any, Dict, List, Protocol

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


class RecencyNeighborSamplerHook:
    r"""Load neighbors from DGraph using a recency sampling strategy where each node maintains a fixed number of most recent neighbors.

    Args:
        num_nbrs (List[int]): Number of neighbors to sample at each hop (max neighbors to keep).

    Raises:
        ValueError: If the num_nbrs list is empty.
    """

    def __init__(self, num_nbrs: List[int]) -> None:
        if not len(num_nbrs):
            raise ValueError('num_nbrs must be non-empty')
        if not all([isinstance(x, int) and (x > 0) for x in num_nbrs]):
            raise ValueError('Each value in num_nbrs must be a positive integer')
        if len(num_nbrs) > 1:
            raise ValueError('RecencyNeighborSamplerHook only supports 1 hop for now')
        self._num_nbrs = num_nbrs
        self._nbr_dict: Dict[int, List[Queue[Any]]] = {}

    @property
    def num_nbrs(self) -> List[int]:
        return self._num_nbrs

    def __call__(self, dg: DGraph) -> DGBatch:
        batch = dg.materialize()
        src = batch.src
        dst = batch.dst
        nids = torch.unique(torch.cat((src, dst)))
        out_nbrs: Dict[int, List[torch.Tensor]] = {}

        # retrieve recent neighbors for each node
        for node in nids.tolist():
            if node not in self._nbr_dict:
                if batch.edge_feats is not None:
                    self._nbr_dict[node] = [
                        Queue(maxsize=self._num_nbrs[0]),
                        Queue(maxsize=self._num_nbrs[0]),
                        Queue(maxsize=self._num_nbrs[0]),
                    ]
                else:
                    self._nbr_dict[node] = [
                        Queue(maxsize=self._num_nbrs[0]),
                        Queue(maxsize=self._num_nbrs[0]),
                    ]

            out_nbrs[node] = []  # (dst,time, edge_feats)
            out_nbrs[node].append(
                torch.tensor(list(self._nbr_dict[node][0].queue))
            )  # dst
            out_nbrs[node].append(
                torch.tensor(list(self._nbr_dict[node][1].queue))
            )  # time
            if batch.edge_feats is not None:
                if self._nbr_dict[node][2].empty():
                    out_nbrs[node].append(torch.tensor([]))
                else:
                    out_nbrs[node].append(
                        torch.stack(list(self._nbr_dict[node][2].queue))
                    )  # stack edge_feats [num_edge, num_feats]

        # add new neighbors to the recency dict
        #! do we need it to be undirected? don't think so, thus only adding src->dst
        for i in range(src.size(0)):
            src_nbr = int(src[i].item())
            if self._nbr_dict[src_nbr][0].full():
                # pop the oldest neighbor
                for k in range(len(self._nbr_dict[src_nbr])):
                    self._nbr_dict[src_nbr][k].get()

            self._nbr_dict[src_nbr][0].put(dst[i].item())
            self._nbr_dict[src_nbr][1].put(batch.time[i].item())
            if batch.edge_feats is not None:
                self._nbr_dict[src_nbr][2].put(batch.edge_feats[i])

        batch.nbrs = out_nbrs  # type: ignore
        return batch
