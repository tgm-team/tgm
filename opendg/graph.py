from __future__ import annotations

from dataclasses import dataclass

import pathlib
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from torch import Tensor

from opendg._io import read_events
from opendg._storage import DGStorage
from opendg.events import Event
from opendg.timedelta import TimeDeltaDG


class DGraph:
    r"""Dynamic Graph Object provides a 'view' over a DGStorage backend."""

    def __init__(
        self,
        data: DGStorage | List[Event] | str | pathlib.Path | pd.DataFrame,
        time_delta: TimeDeltaDG = TimeDeltaDG(unit='r'),
        **kwargs: Any,
    ) -> None:
        if not isinstance(time_delta, TimeDeltaDG):
            raise ValueError(f'bad time_delta type: {type(time_delta)}')
        self.time_delta = time_delta

        if isinstance(data, DGStorage):
            self._storage = data
        else:
            events = data if isinstance(data, list) else read_events(data, **kwargs)
            self._storage = DGStorage(events)

        self._cache = DGSliceCache()

    def to_events(self) -> List[Event]:
        r"""Materialize the events in the DGraph."""
        return self._storage.to_events(
            start_time=self._cache.start_time,
            end_time=self._cache.end_time,
            node_slice=self._cache.node_slice,
        )

    def slice_time(
        self, start_time: Optional[int] = None, end_time: Optional[int] = None
    ) -> DGraph:
        r"""Create and return a new view by slicing temporally (end_time inclusive)."""
        if start_time is not None and end_time is not None and start_time > end_time:
            raise ValueError(
                f'start_time ({start_time}) must be <= end_time ({end_time})'
            )

        dg = DGraph(data=self._storage, time_delta=self.time_delta)

        new_start_time = start_time if start_time is not None else float('-inf')
        new_end_time = end_time if end_time is not None else float('inf')
        if self.start_time is not None and self.start_time > new_start_time:
            new_start_time = self.start_time
        if self.end_time is not None and self.end_time < new_end_time:
            new_end_time = self.end_time

        # Force cache refresh on the new copy if we actually sliced the graph
        if new_start_time != self.start_time or new_end_time != self.end_time:
            dg._cache.node_slice = self._cache.node_slice
        dg._cache.start_time = new_start_time
        dg._cache.end_time = new_end_time
        return dg

    def slice_nodes(self, nodes: List[int]) -> DGraph:
        r"""Create and return a new view by slicing nodes to include."""
        dg = DGraph(data=self._storage, time_delta=self.time_delta)

        if self._cache.node_slice is None:
            self._cache.node_slice = set(range(self.num_nodes))

        # Take intersection of nodes
        dg._cache.node_slice = self._cache.node_slice & set(nodes)

        # Update start time
        start_time_with_node_slice = self._storage.get_start_time(dg._cache.node_slice)
        if self.start_time is None:
            dg._cache.start_time = start_time_with_node_slice
        else:
            dg._cache.start_time = (
                max(start_time_with_node_slice, self._cache.start_time)
                if start_time_with_node_slice is not None
                else self.start_time
            )

        # Update end time
        end_time_with_node_slice = self._storage.get_end_time(dg._cache.node_slice)
        if self.end_time is None:
            dg._cache.end_time = end_time_with_node_slice
        else:
            dg._cache.end_time = (
                min(end_time_with_node_slice, self._cache.end_time)
                if end_time_with_node_slice is not None
                else self._cache.end_time
            )
        return dg

    def __len__(self) -> int:
        r"""The number of timestamps in the dynamic graph."""
        return self.num_timestamps

    def __str__(self) -> str:
        return f'DGraph(storage={self._storage.__class__.__name__}, time_delta={self.time_delta})'

    @property
    def start_time(self) -> Optional[int]:
        r"""The start time of the dynamic graph. None if the graph is empty."""
        if self._cache.start_time is None:
            self._cache.start_time = self._storage.get_start_time(
                self._cache.node_slice
            )
        return self._cache.start_time

    @property
    def end_time(self) -> Optional[int]:
        r"""The end time of the dynamic graph. None, if the graph is empty."""
        if self._cache.end_time is None:
            self._cache.end_time = self._storage.get_end_time(self._cache.node_slice)
        return self._cache.end_time

    @property
    def num_nodes(self) -> int:
        r"""The total number of unique nodes encountered over the dynamic graph."""
        if self._cache.num_nodes is None:
            self._cache.node_slice = self._storage.get_nodes(
                self._cache.start_time,
                self._cache.end_time,
                self._cache.node_slice,
            )
            if len(self._cache.node_slice):
                self._cache.num_nodes = max(self._cache.node_slice) + 1
            else:
                self._cache.num_nodes = 0
        return self._cache.num_nodes

    @property
    def num_edges(self) -> int:
        r"""The total number of unique edges encountered over the dynamic graph."""
        if self._cache.num_edges is None:
            self._cache.num_edges = self._storage.get_num_edges(
                self._cache.start_time,
                self._cache.end_time,
                self._cache.node_slice,
            )
        return self._cache.num_edges

    @property
    def num_timestamps(self) -> int:
        r"""The total number of unique timestamps encountered over the dynamic graph."""
        if self._cache.num_timestamps is None:
            self._cache.num_timestamps = self._storage.get_num_timestamps(
                self._cache.start_time,
                self._cache.end_time,
                self._cache.node_slice,
            )
        return self._cache.num_timestamps

    @property
    def node_feats(self) -> Optional[Tensor]:
        r"""The aggregated node features over the dynamic graph.

        Returns a Tensor.sparse_coo_tensor of size T x V x d_node or None if
        there are no node features on the dynamic graph.
        """
        if self._cache.node_feats is None:
            self._cache.node_feats = self._storage.get_node_feats(
                self._cache.start_time,
                self._cache.end_time,
                self._cache.node_slice,
            )
        return self._cache.node_feats

    @property
    def edge_feats(self) -> Optional[Tensor]:
        r"""The aggregated edge features over the dynamic graph.

        Returns a Tensor.sparse_coo_tensor of size T x V x V x d_edge or None if
        there are no edge features on the dynamic graph.
        """
        if self._cache.edge_feats is None:
            self._cache.edge_feats = self._storage.get_edge_feats(
                self._cache.start_time,
                self._cache.end_time,
                self._cache.node_slice,
            )
        return self._cache.edge_feats


@dataclass(slots=True)
class DGSliceCache:
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    node_slice: Optional[Set[int]] = None
    num_timestamps: Optional[int] = None
    num_nodes: Optional[int] = None
    num_edges: Optional[int] = None
    node_feats: Optional[Tensor] = None
    edge_feats: Optional[Tensor] = None
