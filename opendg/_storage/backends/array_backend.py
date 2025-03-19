import random
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor

from opendg.events import EdgeEvent, Event, NodeEvent

from ..base import DGStorageBase


class DGStorageArrayBackend(DGStorageBase):
    r"""Array backed implementation of temporal graph storage engine."""

    def __init__(self, events: List[Event]) -> None:
        self._node_feats_shape = self._check_node_feature_shapes(events)
        self._edge_feats_shape = self._check_edge_feature_shapes(events)
        # TODO: Maintain sorted list invariant and create temporal index
        # to avoid brute force linear search when start/end times are given
        self._events = events[:]  # Make a copy

    def to_events(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        node_slice: Optional[Set[int]] = None,
    ) -> List[Event]:
        events: List[Event] = []
        for event in self._events:
            if self._valid_slice(event, start_time, end_time, node_slice):
                events.append(event)
        return events

    def get_start_time(self, node_slice: Optional[Set[int]] = None) -> Optional[int]:
        start_time = None
        for event in self._events:
            if self._valid_slice(event, node_slice=node_slice):
                if start_time is None or event.t < start_time:
                    start_time = event.t
        return start_time

    def get_end_time(self, node_slice: Optional[Set[int]] = None) -> Optional[int]:
        end_time = None
        for event in self._events:
            if self._valid_slice(event, node_slice=node_slice):
                if end_time is None or event.t > end_time:
                    end_time = event.t
        return end_time

    def get_nodes(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        node_slice: Optional[Set[int]] = None,
    ) -> Set[int]:
        nodes = set()
        for event in self._events:
            if self._valid_slice(
                event, start_time=start_time, end_time=end_time, node_slice=node_slice
            ):
                if isinstance(event, NodeEvent):
                    nodes.add(event.src)
                elif isinstance(event, EdgeEvent):
                    nodes.add(event.src)
                    nodes.add(event.dst)
        return nodes

    def get_num_edges(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        node_slice: Optional[Set[int]] = None,
    ) -> int:
        edges = set()
        for event in self._events:
            if isinstance(event, EdgeEvent) and self._valid_slice(
                event,
                start_time=start_time,
                end_time=end_time,
                node_slice=node_slice,
            ):
                edges.add((event.t, event.edge))
        return len(edges)

    def get_num_timestamps(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        node_slice: Optional[Set[int]] = None,
    ) -> int:
        timestamps = set()
        for event in self._events:
            if self._valid_slice(
                event,
                start_time=start_time,
                end_time=end_time,
                node_slice=node_slice,
            ):
                timestamps.add(event.t)
        return len(timestamps)

    def get_nbrs(
        self,
        seed_nodes: List[int],
        num_nbrs: List[int],
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        node_slice: Optional[Set[int]] = None,
    ) -> Dict[int, List[List[Tuple[int, int]]]]:
        if len(num_nbrs) > 1:
            raise NotImplementedError(
                f'Multi-hop not impemented for {self.__class__.__name__}'
            )

        seed_nodes_set = set(seed_nodes)

        # TODO: Multi-hop
        hop = 0
        nbrs: Dict[int, List[Set[Tuple[int, int]]]] = {
            node: [set()] for node in seed_nodes
        }
        for event in self._events:
            if isinstance(event, EdgeEvent) and self._valid_slice(
                event, start_time, end_time, node_slice
            ):
                src, dst, t = event.src, event.dst, event.t
                if src in seed_nodes_set:
                    nbrs[src][hop].add((dst, t))
                if dst in seed_nodes_set:
                    nbrs[dst][hop].add((src, t))

        # TODO: Take in a sample_func to enable more than uniform sampling
        sampled_nbrs: Dict[int, List[List[Tuple[int, int]]]] = {}
        for node, nbrs_list in nbrs.items():
            node_nbrs = list(nbrs_list[hop])
            if num_nbrs[hop] != -1 and len(node_nbrs) > num_nbrs[hop]:
                node_nbrs = random.sample(node_nbrs, k=num_nbrs[hop])
            sampled_nbrs[node] = [node_nbrs]
        return sampled_nbrs

    def get_node_feats(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        node_slice: Optional[Set[int]] = None,
    ) -> Optional[Tensor]:
        # Assuming these are both non-negative
        max_time, max_node_id = -1, -1

        indices, values = [], []
        for event in self._events:
            if self._valid_slice(
                event, start_time=start_time, end_time=end_time, node_slice=node_slice
            ):
                if isinstance(event, NodeEvent):
                    max_time = max(max_time, event.t)
                    max_node_id = max(max_node_id, event.src)

                    if event.msg is not None:
                        indices.append([event.t, event.src])
                        values.append(event.msg)

                elif isinstance(event, EdgeEvent):
                    max_time = max(max_time, event.t)
                    max_node_id = max(max_node_id, event.src, event.dst)

        if not len(values):
            return None

        # If the end_time is given, then it determines the dimension of the temporal axis
        # This is true even if there are no events at the end time, which could occur after
        # calling slice_time on a graph.
        max_time = end_time if end_time is not None else max_time

        values_tensor = torch.stack(values)
        indices_tensor = torch.tensor(
            indices
        ).t()  # https://pytorch.org/docs/stable/sparse.html#construction

        assert self._node_feats_shape is not None
        shape = (max_time + 1, max_node_id + 1, *self._node_feats_shape)

        return torch.sparse_coo_tensor(indices_tensor, values_tensor, shape)

    def get_edge_feats(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        node_slice: Optional[Set[int]] = None,
    ) -> Optional[Tensor]:
        # Assuming these are both non-negative
        max_time, max_node_id = -1, -1

        indices, values = [], []
        for event in self._events:
            if self._valid_slice(
                event, start_time=start_time, end_time=end_time, node_slice=node_slice
            ):
                if isinstance(event, NodeEvent):
                    max_time = max(max_time, event.t)
                    max_node_id = max(max_node_id, event.src)
                elif isinstance(event, EdgeEvent):
                    max_time = max(max_time, event.t)
                    max_node_id = max(max_node_id, event.src, event.dst)

                    if event.msg is not None:
                        indices.append([event.t, event.src, event.dst])
                        values.append(event.msg)

        if not len(values):
            return None

        values_tensor = torch.stack(values)
        indices_tensor = torch.tensor(
            indices
        ).t()  # https://pytorch.org/docs/stable/sparse.html#construction

        assert self._edge_feats_shape is not None

        # If the end_time is given, then it determines the dimension of the temporal axis
        # This is true even if there are no events at the end time, which could occur after
        # calling slice_time on a graph.
        max_time = end_time if end_time is not None else max_time

        shape = (
            max_time + 1,
            max_node_id + 1,
            max_node_id + 1,
            *self._edge_feats_shape,
        )

        return torch.sparse_coo_tensor(indices_tensor, values_tensor, shape)

    def _valid_slice(
        self,
        event: Event,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        node_slice: Optional[Set[int]] = None,
    ) -> bool:
        lb_time = float('-inf') if start_time is None else start_time
        ub_time = float('inf') if end_time is None else end_time

        time_valid = lb_time <= event.t <= ub_time
        node_valid = (
            node_slice is None
            or (isinstance(event, NodeEvent) and event.src in node_slice)
            or (
                isinstance(event, EdgeEvent)
                and len(set(event.edge).intersection(node_slice)) > 0
            )
        )
        # TODO: This can be optimized by returning these seperately, and hence early
        # returning out of the event loop if we already know the timestamp is not valid
        return time_valid and node_valid
