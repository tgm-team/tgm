from abc import ABC, abstractmethod
from typing import List, Optional, Union

import torch
from torch import Tensor

from opendg.events import EdgeEvent, Event, NodeEvent
from opendg.timedelta import TimeDeltaTG


class DGStorageBase(ABC):
    r"""Base class for dynamic graph storage engine."""

    @abstractmethod
    def __init__(self, events: List[Event], time_delta: TimeDeltaTG) -> None:
        r"""Initialize a dynamic graph from a list of events and a time delta."""

    @abstractmethod
    def to_events(self) -> List[Event]:
        r"""Convert dynamic graph to a list of events."""

    @abstractmethod
    def slice_time(self, start_time: int, end_time: int) -> 'DGStorageBase':
        r"""Extract temporal slice of the dynamic graph between start_time and end_time."""

    @abstractmethod
    def slice_nodes(self, nodes: List[int]) -> 'DGStorageBase':
        r"""Extract topological slice of the dynamcic graph given the list of nodes."""

    @abstractmethod
    def append(self, events: Union[Event, List[Event]]) -> 'DGStorageBase':
        r"""Append events to the temporal end of the dynamic graph."""

    @abstractmethod
    def temporal_coarsening(
        self, time_delta: TimeDeltaTG, agg_func: str = 'sum'
    ) -> 'DGStorageBase':
        r"""Re-index the temporal axis of the dynamic graph."""

    def __len__(self) -> int:
        r"""Returns the number of temporal length of the dynamic graph."""
        return self.num_timestamps

    def __str__(self) -> str:
        r"""Returns summary properties of the dynamic graph."""
        return f'Dynamic Graph Storage Engine ({self.__class__.__name__}), Start Time: {self.start_time}, End Time: {self.end_time}, Nodes: {self.num_nodes}, Edges: {self.num_edges}, Timestamps: {self.num_timestamps}, Time Delta: {self.time_delta}'

    @property
    @abstractmethod
    def start_time(self) -> Optional[int]:
        r"""The start time of the dynamic graph. None if the graph is empty."""

    @property
    @abstractmethod
    def end_time(self) -> Optional[int]:
        r"""The end time of the dynamic graph. None, if the graph is empty."""

    @property
    @abstractmethod
    def time_delta(self) -> TimeDeltaTG:
        r"""The time granularity of the dynamic graph."""

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        r"""The total number of unique nodes encountered over the dynamic graph."""

    @property
    @abstractmethod
    def num_edges(self) -> int:
        r"""The total number of unique edges encountered over the dynamic graph."""

    @property
    @abstractmethod
    def num_timestamps(self) -> int:
        r"""The total number of unique timestamps encountered over the dynamic graph."""

    @property
    @abstractmethod
    def node_feats(self) -> Optional[Tensor]:
        r"""The aggregated node features over the dynamic graph.

        Returns a tensor.sparse_coo_tensor of size T x V x d where

        - T = Number of timestamps
        - V = Number of nodes
        - d = Node feature dimension
        or None if there are no node features on the dynamic graph.

        """

    @property
    @abstractmethod
    def edge_feats(self) -> Optional[Tensor]:
        r"""The aggregated edge features over the dynamic graph.

        Returns a tensor.sparse_coo_tensor of size T x V x V x  d where

        - T = Number of timestamps
        - E = Number of edges
        - d = Edge feature dimension

        or None if there are no edge features on the dynamic graph.
        """

    def _check_slice_time_args(self, start_time: int, end_time: int) -> None:
        if start_time > end_time:
            raise ValueError(
                f'Bad slice: start_time must be <= end_time but received: start_time ({start_time}) > end_time ({end_time})'
            )

    def _check_temporal_coarsening_args(
        self, time_delta: TimeDeltaTG, agg_func: str
    ) -> None:
        if not len(self):
            raise ValueError('Cannot temporally coarsen an empty dynamic graph')

        # TODO: Validate time_delta and agg_func

    def _check_node_feature_shapes(
        self, events: List[Event], expected_shape: Optional[torch.Size] = None
    ) -> Optional[torch.Size]:
        node_feats_shape = expected_shape

        for event in events:
            if isinstance(event, NodeEvent) and event.features is not None:
                if node_feats_shape is None:
                    node_feats_shape = event.features.shape
                elif node_feats_shape != event.features.shape:
                    raise ValueError(
                        f'Incompatible node features shapes: {node_feats_shape} != {event.features.shape}'
                    )
        return node_feats_shape

    def _check_edge_feature_shapes(
        self, events: List[Event], expected_shape: Optional[torch.Size] = None
    ) -> Optional[torch.Size]:
        edge_feats_shape = expected_shape

        for event in events:
            if isinstance(event, EdgeEvent) and event.features is not None:
                if edge_feats_shape is None:
                    edge_feats_shape = event.features.shape
                elif edge_feats_shape != event.features.shape:
                    raise ValueError(
                        f'Incompatible edge features shapes: {edge_feats_shape} != {event.features.shape}'
                    )
        return edge_feats_shape
