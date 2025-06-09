import random
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor

from opendg.data import DGData

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
        for i in range(*self._binary_search(slice)):
            nodes = self._nodes_in_event(i)
            if slice.node_slice is None or any(x in slice.node_slice for x in nodes):
                return int(self._data.timestamps[i].item())
        return None

    def get_end_time(self, slice: DGSliceTracker) -> Optional[int]:
        lb_idx, ub_idx = self._binary_search(slice)
        for i in range(ub_idx - 1, lb_idx - 1, -1):
            nodes = self._nodes_in_event(i)
            if slice.node_slice is None or any(x in slice.node_slice for x in nodes):
                return int(self._data.timestamps[i].item())
        return None

    def get_nodes(self, slice: DGSliceTracker) -> Set[int]:
        all_nodes: Set[int] = set()
        for i in range(*self._binary_search(slice)):
            nodes = self._nodes_in_event(i)
            if slice.node_slice is None or any(x in slice.node_slice for x in nodes):
                all_nodes.update(nodes)
        return all_nodes

    def get_edges(self, slice: DGSliceTracker) -> Tuple[Tensor, Tensor, Tensor]:
        src, dst, time = [], [], []
        for i in range(*self._binary_search(slice)):
            if i not in self._edge_idx_map:
                continue
            nodes = self._nodes_in_event(i)
            if slice.node_slice is None or any(x in slice.node_slice for x in nodes):
                src.append(nodes[0])
                dst.append(nodes[1])
                time.append(self._data.timestamps[i].item())

        src_tensor = torch.LongTensor(src)
        dst_tensor = torch.LongTensor(dst)
        time_tensor = torch.LongTensor(time)
        return src_tensor, dst_tensor, time_tensor

    def get_num_timestamps(self, slice: DGSliceTracker) -> int:
        timestamps = set()
        for i in range(*self._binary_search(slice)):
            nodes = self._nodes_in_event(i)
            if slice.node_slice is None or any(x in slice.node_slice for x in nodes):
                timestamps.add(self._data.timestamps[i].item())
        return len(timestamps)

    def get_num_events(self, slice: DGSliceTracker) -> int:
        num_events = 0
        for i in range(*self._binary_search(slice)):
            nodes = self._nodes_in_event(i)
            if slice.node_slice is None or any(x in slice.node_slice for x in nodes):
                num_events += 1
        return num_events

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

        nbrs: Dict[int, Set[Tuple[int, int]]] = {node: set() for node in seed_nodes_set}
        for i in range(*self._binary_search(slice)):
            if i not in self._edge_idx_map:
                continue
            nodes = self._nodes_in_event(i)
            if slice.node_slice is None or all(x in slice.node_slice for x in nodes):
                # Use 0/1 flag to denote dst/src neighbor, respectively
                src, dst = nodes
                if src in seed_nodes_set:
                    nbrs[src].add((i, 1))
                if dst in seed_nodes_set:
                    nbrs[dst].add((i, 0))

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

        max_time, max_node_id = -1, -1  # Assuming these are both non-negative
        indices, values = [], []
        for i in range(*self._binary_search(slice)):
            nodes = self._nodes_in_event(i)
            if slice.node_slice is None or any(x in slice.node_slice for x in nodes):
                time = int(self._data.timestamps[i].item())
                max_time = max(max_time, time)
                max_node_id = max(max_node_id, *nodes)
                if i not in self._edge_idx_map:
                    indices.append([time, nodes[0]])
                    values.append(self._data.dynamic_node_feats[self._node_idx_map[i]])  # type: ignore

        if not len(values):
            return None

        # If the end_time is given, then it determines the dimension of the temporal axis
        # even if there are no events at the end time (could be the case after calling slice_time)
        max_time = slice.end_time or max_time
        node_feats_dim = self.get_dynamic_node_feats_dim()
        assert node_feats_dim is not None

        # https://pytorch.org/docs/stable/sparse.html#construction
        values_tensor = torch.stack(values)
        indices_tensor = torch.tensor(indices).t()
        shape = (max_time + 1, max_node_id + 1, node_feats_dim)
        return torch.sparse_coo_tensor(indices_tensor, values_tensor, shape)

    def get_edge_feats(self, slice: DGSliceTracker) -> Optional[Tensor]:
        if self._data.edge_feats is None:
            return None

        max_time, max_node_id = -1, -1  # Assuming these are both non-negative
        indices, values = [], []
        for i in range(*self._binary_search(slice)):
            nodes = self._nodes_in_event(i)
            if slice.node_slice is None or any(x in slice.node_slice for x in nodes):
                time = int(self._data.timestamps[i].item())
                max_time = max(max_time, time)
                max_node_id = max(max_node_id, *nodes)
                if i in self._edge_idx_map:
                    indices.append([time, nodes[0], nodes[1]])
                    values.append(self._data.edge_feats[self._edge_idx_map[i]])

        if not len(values):
            return None

        # If the end_time is given, then it determines the dimension of the temporal axis
        # even if there are no events at the end time (could be the case after calling slice_time)
        max_time = slice.end_time or max_time
        edge_feats_dim = self.get_edge_feats_dim()
        assert edge_feats_dim is not None

        # https://pytorch.org/docs/stable/sparse.html#construction
        values_tensor = torch.stack(values)
        indices_tensor = torch.tensor(indices).t()
        shape = (max_time + 1, max_node_id + 1, max_node_id + 1, edge_feats_dim)
        return torch.sparse_coo_tensor(indices_tensor, values_tensor, shape)

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

    def _nodes_in_event(self, i: int) -> Tuple[int, ...]:
        if i in self._edge_idx_map:
            return tuple(self._data.edge_index[self._edge_idx_map[i]].tolist())
        else:
            return (int(self._data.node_ids[self._node_idx_map[i]].item()),)  # type: ignore

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
