import random
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor

from tgm.constants import PADDED_NODE_ID
from tgm.util.logging import _get_logger

from ..base import DGSliceTracker, DGStorageBase

logger = _get_logger(__name__)


class DGStorageArrayBackend(DGStorageBase):
    r"""Array backed implementation of temporal graph storage engine."""

    def __init__(self, data: 'DGData') -> None:  # type: ignore
        self._data = data

        # Binary search caches for finding timestamps in event array
        self._lb_cache: Dict[Optional[int], int] = {}
        self._ub_cache: Dict[Optional[int], int] = {}

    def get_start_time(self, slice: DGSliceTracker) -> Optional[int]:
        lb_idx, ub_idx = self._binary_search(slice)
        if lb_idx >= ub_idx:
            logger.debug('No events in slice: %s', slice)
            return None
        return int(self._data.time[lb_idx].item())

    def get_end_time(self, slice: DGSliceTracker) -> Optional[int]:
        lb_idx, ub_idx = self._binary_search(slice)
        if lb_idx >= ub_idx:
            logger.debug('No events in slice: %s', slice)
            return None
        return int(self._data.time[ub_idx - 1].item())

    def get_nodes(self, slice: DGSliceTracker) -> Set[int]:
        all_nodes: Set[int] = set()
        lb_idx, ub_idx = self._binary_search(slice)

        edge_mask = (self._data.edge_mask >= lb_idx) & (self._data.edge_mask < ub_idx)
        edge_event_nodes = self._data.edge_index[edge_mask].unique().tolist()
        all_nodes.update(edge_event_nodes)

        if self._data.node_x_mask is not None:
            node_x_mask = (self._data.node_x_mask >= lb_idx) & (
                self._data.node_x_mask < ub_idx
            )
            node_x_nids = self._data.node_x_nids[node_x_mask].unique().tolist()
            all_nodes.update(node_x_nids)
        if not all_nodes:
            logger.debug('No events in slice: %s', slice)
        return all_nodes

    def get_edges(self, slice: DGSliceTracker) -> Tuple[Tensor, Tensor, Tensor]:
        lb_idx, ub_idx = self._binary_search(slice)
        edge_mask = (self._data.edge_mask >= lb_idx) & (self._data.edge_mask < ub_idx)
        edges = self._data.edge_index[edge_mask]
        src, dst = edges[:, 0], edges[:, 1]
        time = self._data.time[self._data.edge_mask[edge_mask]]

        src, dst, time = src.contiguous(), dst.contiguous(), time.contiguous()

        if not edges.numel():
            logger.debug('No edge events in slice: %s', slice)
        return src, dst, time

    def get_node_events(self, slice: DGSliceTracker) -> Tuple[Tensor, Tensor]:
        if self._data.node_x_mask is None:
            return torch.empty(0, dtype=torch.int), torch.empty(0, dtype=torch.long)

        lb_idx, ub_idx = self._binary_search(slice)
        node_x_mask = (self._data.node_x_mask >= lb_idx) & (
            self._data.node_x_mask < ub_idx
        )
        node_x_nids = self._data.node_x_nids[node_x_mask]
        time = self._data.time[self._data.node_x_mask[node_x_mask]]

        if not node_x_nids.numel():
            logger.debug('No node events in slice: %s', slice)
        return node_x_nids, time

    def get_node_labels(self, slice: DGSliceTracker) -> Tuple[Tensor, Tensor]:
        if self._data.node_y_mask is None:
            return (torch.empty(0, dtype=torch.int), torch.empty(0, dtype=torch.long))

        lb_idx, ub_idx = self._binary_search(slice)
        node_y_mask = (self._data.node_y_mask >= lb_idx) & (
            self._data.node_y_mask < ub_idx
        )
        node_y_nids = self._data.node_y_nids[node_y_mask]
        time = self._data.time[self._data.node_y_mask[node_y_mask]]

        if not node_y_nids.numel():
            logger.debug('No node labels in slice: %s', slice)
        return node_y_nids, time

    def get_num_timestamps(self, slice: DGSliceTracker) -> int:
        lb_idx, ub_idx = self._binary_search(slice)
        return len(self._data.time[lb_idx:ub_idx].unique())

    def get_num_events(self, slice: DGSliceTracker) -> int:
        lb_idx, ub_idx = self._binary_search(slice)
        return ub_idx - lb_idx

    def get_nbrs(
        self,
        seed_nodes: Tensor,
        num_nbrs: int,
        slice: DGSliceTracker,
        directed: bool,
    ) -> Tuple[Tensor, ...]:
        # TODO: Take in a sample_func to enable more than uniform sampling
        device = seed_nodes.device
        unique_nodes, inverse_indices = seed_nodes.unique(return_inverse=True)
        seed_nodes_set = set(unique_nodes.tolist())

        lb_idx, ub_idx = self._binary_search(slice)
        edge_mask = (self._data.edge_mask >= lb_idx) & (self._data.edge_mask < ub_idx)
        edges = self._data.edge_index[edge_mask]
        event_ids = self._data.edge_mask[edge_mask]

        src_list = edges[:, 0].tolist()
        dst_list = edges[:, 1].tolist()
        eid_list = event_ids.tolist()

        # This is a loop over the entire graph up to (not including) the current batch end time
        # which results in quadratic cost for a single epoch. Consider raises a warning to let the
        # user know that this is not the right backend for this.
        nbrs: Dict[int, List[Tuple[int, int]]] = {node: [] for node in seed_nodes_set}
        for s, d, i in zip(src_list, dst_list, eid_list):
            if s in nbrs:
                nbrs[s].append((i, d))
            if not directed and d in nbrs:
                nbrs[d].append((i, s))

        B = len(seed_nodes)
        nbr_nids = torch.full(
            (B, num_nbrs), PADDED_NODE_ID, dtype=torch.int32, device=device
        )
        nbr_times = torch.zeros(B, num_nbrs, dtype=torch.int64, device=device)
        nbr_feats = torch.zeros(B, num_nbrs, self.get_edge_x_dim() or 0, device=device)

        for i, node in enumerate(unique_nodes.tolist()):
            node_nbrs = nbrs[node]
            if not len(node_nbrs):
                continue

            # Subsample if we have more neighbours than was queried
            if len(node_nbrs) > num_nbrs:
                node_nbrs = random.sample(node_nbrs, k=num_nbrs)

            nbr_ids, times, feats = [], [], []
            for eid, nbr_id in node_nbrs:
                nbr_ids.append(nbr_id)
                times.append(self._data.time[eid])
                if self._data.edge_x is not None:
                    feats.append(self._data.edge_x[eid])

            nn = len(nbr_ids)
            mask = inverse_indices == i
            nbr_nids[mask, :nn] = torch.tensor(
                nbr_ids, dtype=torch.int32, device=device
            )
            nbr_times[mask, :nn] = torch.tensor(times, dtype=torch.int64, device=device)
            if self._data.edge_x is not None:
                nbr_feats[mask, :nn] = torch.stack(feats).to(device)

        return nbr_nids, nbr_times, nbr_feats

    def get_static_node_x(self) -> Optional[Tensor]:
        return self._data.static_node_x

    def get_node_type(self) -> Optional[Tensor]:
        return self._data.node_type

    def get_node_x(self, slice: DGSliceTracker) -> Optional[Tensor]:
        if self._data.node_x is None:
            return None
        assert self._data.node_x_mask is not None  # for mypy
        assert self._data.node_x_nids is not None  # for mypy

        lb_idx, ub_idx = self._binary_search(slice)
        node_x_mask = (self._data.node_x_mask >= lb_idx) & (
            self._data.node_x_mask < ub_idx
        )
        if node_x_mask.sum() == 0:
            logger.debug(f'No dynamic node features in slice {slice}')
            return None

        time = self._data.time[self._data.node_x_mask[node_x_mask]]
        nodes = self._data.node_x_nids[node_x_mask]
        indices = torch.stack([time, nodes], dim=0)
        values = self._data.node_x[node_x_mask]

        max_node_id = nodes.max()
        edge_mask = (self._data.edge_mask >= lb_idx) & (self._data.edge_mask < ub_idx)
        if edge_mask.sum() != 0 and len(self._data.edge_index[edge_mask]):
            max_node_id = max(max_node_id, self._data.edge_index[edge_mask].max())

        # Note: even though the node label node ids are asserted to be in range
        # of the graph, this need not hold locally, (i.e., within a graph slice).
        # Therefore, we have to run the logic below to get the max node id within
        # the query slice, to return the correctly sized sparse tensor.
        if self._data.node_y_mask is not None:
            node_y_mask = (self._data.node_y_mask >= lb_idx) & (
                self._data.node_y_mask < ub_idx
            )
            if node_y_mask.sum() != 0 and len(self._data.node_y_nids[node_y_mask]):
                max_node_id = max(
                    max_node_id, self._data.node_y_nids[node_y_mask].max()
                )

        max_time = slice.end_time or self._data.time[ub_idx - 1]
        node_x_dim = self.get_node_x_dim()
        shape = (max_time + 1, max_node_id + 1, node_x_dim)
        return torch.sparse_coo_tensor(indices, values, shape)  # type: ignore

    def get_node_y(self, slice: DGSliceTracker) -> Optional[Tensor]:
        if self._data.node_y is None:
            return None
        assert self._data.node_y_mask is not None  # for mypy
        assert self._data.node_y_nids is not None  # for mypy

        lb_idx, ub_idx = self._binary_search(slice)
        node_y_mask = (self._data.node_y_mask >= lb_idx) & (
            self._data.node_y_mask < ub_idx
        )
        if node_y_mask.sum() == 0:
            logger.debug(f'No dynamic node labels in slice {slice}')
            return None

        time = self._data.time[self._data.node_y_mask[node_y_mask]]
        nodes = self._data.node_y_nids[node_y_mask]
        indices = torch.stack([time, nodes], dim=0)
        values = self._data.node_y[node_y_mask]

        max_node_id = nodes.max()
        edge_mask = (self._data.edge_mask >= lb_idx) & (self._data.edge_mask < ub_idx)
        if edge_mask.sum() != 0 and len(self._data.edge_index[edge_mask]):
            max_node_id = max(max_node_id, self._data.edge_index[edge_mask].max())

        if self._data.node_x_mask is not None:
            node_x_mask = (self._data.node_x_mask >= lb_idx) & (
                self._data.node_x_mask < ub_idx
            )
            if node_x_mask.sum() != 0 and len(self._data.node_x_nids[node_x_mask]):
                max_node_id = max(
                    max_node_id, self._data.node_x_nids[node_x_mask].max()
                )

        max_time = slice.end_time or self._data.time[ub_idx - 1]
        node_y_dim = self.get_node_y_dim()
        shape = (max_time + 1, max_node_id + 1, node_y_dim)
        return torch.sparse_coo_tensor(indices, values, shape)  # type: ignore

    def get_edge_x(self, slice: DGSliceTracker) -> Optional[Tensor]:
        if self._data.edge_x is None:
            return None

        lb_idx, ub_idx = self._binary_search(slice)
        edge_mask = (self._data.edge_mask >= lb_idx) & (self._data.edge_mask < ub_idx)
        if edge_mask.sum() == 0:
            return None

        return self._data.edge_x[edge_mask]

    def get_edge_type(self, slice: DGSliceTracker) -> Optional[Tensor]:
        if self._data.edge_type is None:
            return None

        lb_idx, ub_idx = self._binary_search(slice)
        edge_mask = (self._data.edge_mask >= lb_idx) & (self._data.edge_mask < ub_idx)
        if edge_mask.sum() == 0:
            return None

        return self._data.edge_type[edge_mask]

    def get_static_node_x_dim(self) -> Optional[int]:
        if self._data.static_node_x is None:
            return None
        return self._data.static_node_x.shape[1]

    def get_node_x_dim(self) -> Optional[int]:
        if self._data.node_x is None:
            return None
        return self._data.node_x.shape[1]

    def get_node_y_dim(self) -> Optional[int]:
        if self._data.node_y is None:
            return None
        return self._data.node_y.shape[1]

    def get_edge_x_dim(self) -> Optional[int]:
        if self._data.edge_x is None:
            return None
        return self._data.edge_x.shape[1]

    def _binary_search(self, slice: DGSliceTracker) -> Tuple[int, int]:
        ts = self._data.time
        if slice.start_time not in self._lb_cache:
            t = ts[0] if slice.start_time is None else slice.start_time
            self._lb_cache[slice.start_time] = int(torch.searchsorted(ts, t))
            logger.debug(f'Cache miss: start_time={slice.start_time}')
        if slice.end_time not in self._ub_cache:
            t = ts[-1] if slice.end_time is None else slice.end_time
            self._ub_cache[slice.end_time] = int(
                torch.searchsorted(ts, t, side='right')
            )
            logger.debug(f'Cache miss: end_time={slice.start_time}')

        lb = self._lb_cache[slice.start_time]
        ub = self._ub_cache[slice.end_time]

        # Additional clamping on possible index constraints
        clamp = lambda x, lo, hi: max(lo, min(hi, x))
        lb = clamp(lb, slice.start_idx or 0, slice.end_idx or len(ts))
        ub = clamp(ub, slice.start_idx or 0, slice.end_idx or len(ts))
        return lb, ub
