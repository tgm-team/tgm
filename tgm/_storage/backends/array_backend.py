import random
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor

from tgm.data import DGData

from ..base import DGSliceTracker, DGStorageBase


class DGStorageArrayBackend(DGStorageBase):
    r"""Array backed implementation of temporal graph storage engine."""

    def __init__(self, data: DGData) -> None:
        self._data = data

        # Pre-compute edge event indices
        self._edge_idx_map = {
            idx.item(): i for i, idx in enumerate(self._data.edge_event_idx)
        }
        self._node_idx_map = (
            {idx.item(): i for i, idx in enumerate(self._data.node_event_idx)}
            if self._data.node_event_idx is not None
            else None
        )

        # Binary search caches for finding timestamps in event array
        self._lb_cache: Dict[Optional[int], int] = {}
        self._ub_cache: Dict[Optional[int], int] = {}

    def get_start_time(self, slice: DGSliceTracker) -> Optional[int]:
        lb_idx, ub_idx = self._binary_search(slice)
        if lb_idx >= ub_idx:
            return None
        return int(self._data.timestamps[lb_idx].item())

    def get_end_time(self, slice: DGSliceTracker) -> Optional[int]:
        lb_idx, ub_idx = self._binary_search(slice)
        if lb_idx >= ub_idx:
            return None
        return int(self._data.timestamps[ub_idx - 1].item())

    def get_nodes(self, slice: DGSliceTracker) -> Set[int]:
        all_nodes: Set[int] = set()
        lb_idx, ub_idx = self._binary_search(slice)

        edge_mask = (self._data.edge_event_idx >= lb_idx) & (
            self._data.edge_event_idx < ub_idx
        )
        edge_event_nodes = self._data.edge_index[edge_mask].unique().tolist()
        all_nodes.update(edge_event_nodes)

        if self._data.node_event_idx is not None:
            node_mask = (self._data.node_event_idx >= lb_idx) & (
                self._data.node_event_idx < ub_idx
            )
            node_event_nodes = self._data.node_ids[node_mask].unique().tolist()  # type: ignore
            all_nodes.update(node_event_nodes)
        return all_nodes

    def get_edges(self, slice: DGSliceTracker) -> Tuple[Tensor, Tensor, Tensor]:
        lb_idx, ub_idx = self._binary_search(slice)
        edge_mask = (self._data.edge_event_idx >= lb_idx) & (
            self._data.edge_event_idx < ub_idx
        )
        edges = self._data.edge_index[edge_mask]
        src, dst = edges[:, 0], edges[:, 1]
        time = self._data.timestamps[self._data.edge_event_idx[edge_mask]]
        return src, dst, time

    def get_num_timestamps(self, slice: DGSliceTracker) -> int:
        lb_idx, ub_idx = self._binary_search(slice)
        return len(self._data.timestamps[lb_idx:ub_idx].unique())

    def get_num_events(self, slice: DGSliceTracker) -> int:
        lb_idx, ub_idx = self._binary_search(slice)
        return ub_idx - lb_idx

    def get_nbrs(
        self,
        seed_nodes: Tensor,
        num_nbrs: List[int],
        slice: DGSliceTracker,
    ) -> Tuple[List[Tensor], ...]:
        # TODO: Take in a sample_func to enable more than uniform sampling
        if len(num_nbrs) > 1:
            raise NotImplementedError(f'Multi-hop not implemented')
        n_nbrs = num_nbrs[0]
        unique, inverse_indices = seed_nodes.unique(return_inverse=True)
        seed_nodes_set = set(unique.tolist())

        lb_idx, ub_idx = self._binary_search(slice)
        edge_mask = (self._data.edge_event_idx >= lb_idx) & (
            self._data.edge_event_idx < ub_idx
        )
        edges = self._data.edge_index[edge_mask]
        event_ids = self._data.edge_event_idx[edge_mask]

        nbrs: Dict[int, Set[Tuple[int, int]]] = {node: set() for node in seed_nodes_set}
        for edge, i in zip(edges, event_ids):
            src, dst = edge
            # Use 0/1 flag to denote dst/src neighbor, respectively
            if src in seed_nodes_set:
                nbrs[src.item()].add((i, 1))
            if dst in seed_nodes_set:
                nbrs[dst.item()].add((i, 0))

        # TODO: Node feats
        batch_size = len(seed_nodes)
        nbr_nids = torch.empty(batch_size, n_nbrs, dtype=torch.long)
        nbr_times = torch.empty(batch_size, n_nbrs, dtype=torch.long)
        nbr_feats = torch.zeros(batch_size, n_nbrs, self.get_edge_feats_dim())  # type: ignore
        nbr_mask = torch.zeros(batch_size, n_nbrs, dtype=torch.long)
        for i, nbrs_set in enumerate(nbrs.values()):
            node_nbrs = list(nbrs_set)
            if not len(node_nbrs):
                continue
            if n_nbrs != -1 and len(node_nbrs) > n_nbrs:
                node_nbrs = random.sample(node_nbrs, k=n_nbrs)

            nbr_nids_, nbr_times_, nbr_feats_ = [], [], []
            for event_idx, edge_idx in node_nbrs:
                nbr_nids_.append(self._data.edge_index[event_idx, edge_idx].item())
                nbr_times_.append(self._data.timestamps[event_idx])
                if self._data.edge_feats is not None:
                    nbr_feats_.append(self._data.edge_feats[event_idx])

            nn = len(node_nbrs)
            mask = inverse_indices == i
            nbr_nids[mask, :nn] = torch.LongTensor(nbr_nids_)
            nbr_times[mask, :nn] = torch.LongTensor(nbr_times_)
            nbr_feats[mask, :nn] = torch.stack(nbr_feats_)
            nbr_mask[mask, :nn] = 1
        return [seed_nodes], [nbr_nids], [nbr_times], [nbr_feats], [nbr_mask]

    def get_static_node_feats(self) -> Optional[Tensor]:
        return self._data.static_node_feats

    def get_dynamic_node_feats(self, slice: DGSliceTracker) -> Optional[Tensor]:
        if self._data.dynamic_node_feats is None:
            return None
        assert self._data.node_event_idx is not None  # for mypy
        assert self._data.node_ids is not None  # for mypy

        lb_idx, ub_idx = self._binary_search(slice)
        node_mask = (self._data.node_event_idx >= lb_idx) & (
            self._data.node_event_idx < ub_idx
        )
        if node_mask.sum() == 0:
            return None

        time = self._data.timestamps[self._data.node_event_idx[node_mask]]
        nodes = self._data.node_ids[node_mask]
        indices = torch.stack([time, nodes], dim=0)
        values = self._data.dynamic_node_feats[node_mask]

        max_node_id = nodes.max()
        edge_mask = (self._data.edge_event_idx >= lb_idx) & (
            self._data.edge_event_idx < ub_idx
        )
        if edge_mask.sum() != 0 and len(self._data.edge_index[edge_mask]):
            max_node_id = max(max_node_id, self._data.edge_index[edge_mask].max())  # type: ignore

        max_time = slice.end_time or self._data.timestamps[ub_idx - 1]
        node_feats_dim = self.get_dynamic_node_feats_dim()
        shape = (max_time + 1, max_node_id + 1, node_feats_dim)
        return torch.sparse_coo_tensor(indices, values, shape)  # type: ignore

    def get_edge_feats(self, slice: DGSliceTracker) -> Optional[Tensor]:
        if self._data.edge_feats is None:
            return None

        lb_idx, ub_idx = self._binary_search(slice)
        edge_mask = (self._data.edge_event_idx >= lb_idx) & (
            self._data.edge_event_idx < ub_idx
        )
        if edge_mask.sum() == 0:
            return None

        time = self._data.timestamps[self._data.edge_event_idx[edge_mask]]
        edges = self._data.edge_index[edge_mask]
        indices = torch.stack([time, edges[:, 0], edges[:, 1]], dim=0)
        values = self._data.edge_feats[edge_mask]

        max_node_id = edges.max()
        if self._data.node_event_idx is not None:
            node_mask = (self._data.node_event_idx >= lb_idx) & (
                self._data.node_event_idx < ub_idx
            )
            if len(self._data.node_ids[node_mask]):  # type: ignore
                max_node_id = max(max_node_id, self._data.node_ids[node_mask].max())  # type: ignore

        max_time = slice.end_time or self._data.timestamps[ub_idx - 1]
        edge_feats_dim = self.get_edge_feats_dim()
        shape = (max_time + 1, max_node_id + 1, max_node_id + 1, edge_feats_dim)
        return torch.sparse_coo_tensor(indices, values, shape)  # type: ignore

    def get_static_node_feats_dim(self) -> Optional[int]:
        if self._data.static_node_feats is None:
            return None
        return self._data.static_node_feats.shape[1]

    def get_dynamic_node_feats_dim(self) -> Optional[int]:
        if self._data.dynamic_node_feats is None:
            return None
        return self._data.dynamic_node_feats.shape[1]

    def get_edge_feats_dim(self) -> Optional[int]:
        if self._data.edge_feats is None:
            return None
        return self._data.edge_feats.shape[1]

    def _binary_search(self, slice: DGSliceTracker) -> Tuple[int, int]:
        ts = self._data.timestamps
        if slice.start_time not in self._lb_cache:
            t = ts[0] if slice.start_time is None else slice.start_time
            self._lb_cache[slice.start_time] = int(torch.searchsorted(ts, t))
        if slice.end_time not in self._ub_cache:
            t = ts[-1] if slice.end_time is None else slice.end_time
            self._ub_cache[slice.end_time] = int(
                torch.searchsorted(ts, t, side='right')
            )
        lb = self._lb_cache[slice.start_time]
        ub = self._ub_cache[slice.end_time]

        # Additional clamping on possible index constraints
        clamp = lambda x, lo, hi: lo if x < lo else hi if x > hi else x
        lb = clamp(lb, slice.start_idx or 0, slice.end_idx or len(ts))
        ub = clamp(ub, slice.start_idx or 0, slice.end_idx or len(ts))
        return lb, ub
