from __future__ import annotations

import pathlib
from dataclasses import dataclass
from functools import cached_property
from typing import Any, List, Optional, Set

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

        self._slice = DGSliceTracker()

    def to_events(self) -> List[Event]:
        r"""Materialize the events in the DGraph."""
        return self._storage.to_events(
            start_time=self._slice.start_time,
            end_time=self._slice.end_time,
            node_slice=self._slice.node_slice,
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
            dg._slice.node_slice = self._slice.node_slice
        dg._slice.start_time = new_start_time
        dg._slice.end_time = new_end_time
        return dg

    def slice_nodes(self, nodes: List[int]) -> DGraph:
        r"""Create and return a new view by slicing nodes to include."""
        dg = DGraph(data=self._storage, time_delta=self.time_delta)

        if self._slice.node_slice is None:
            self._slice.node_slice = set(range(self.num_nodes))

        # Take intersection of nodes
        dg._slice.node_slice = self._slice.node_slice & set(nodes)

        # Update start time
        start_time_with_node_slice = self._storage.get_start_time(dg._slice.node_slice)
        if self.start_time is None:
            dg._slice.start_time = start_time_with_node_slice
        else:
            dg._slice.start_time = (
                max(start_time_with_node_slice, self._slice.start_time)
                if start_time_with_node_slice is not None
                else self.start_time
            )

        # Update end time
        end_time_with_node_slice = self._storage.get_end_time(dg._slice.node_slice)
        if self.end_time is None:
            dg._slice.end_time = end_time_with_node_slice
        else:
            dg._slice.end_time = (
                min(end_time_with_node_slice, self._slice.end_time)
                if end_time_with_node_slice is not None
                else self._slice.end_time
            )
        return dg

    def __len__(self) -> int:
        r"""The number of timestamps in the dynamic graph."""
        return self.num_timestamps

    def __str__(self) -> str:
        return f'DGraph(storage={self._storage.__class__.__name__}, time_delta={self.time_delta})'

    @cached_property
    def start_time(self) -> Optional[int]:
        r"""The start time of the dynamic graph. None if the graph is empty."""
        if self._slice.start_time is None:
            self._slice.start_time = self._storage.get_start_time(
                self._slice.node_slice
            )
        return self._slice.start_time

    @cached_property
    def end_time(self) -> Optional[int]:
        r"""The end time of the dynamic graph. None, if the graph is empty."""
        if self._slice.end_time is None:
            self._slice.end_time = self._storage.get_end_time(self._slice.node_slice)
        return self._slice.end_time

    @cached_property
    def num_nodes(self) -> int:
        r"""The total number of unique nodes encountered over the dynamic graph."""
        self._slice.node_slice = self._storage.get_nodes(
            self._slice.start_time,
            self._slice.end_time,
            self._slice.node_slice,
        )
        return max(self._slice.node_slice) + 1 if len(self._slice.node_slice) else 0

    @cached_property
    def num_edges(self) -> int:
        r"""The total number of unique edges encountered over the dynamic graph."""
        return self._storage.get_num_edges(
            self._slice.start_time,
            self._slice.end_time,
            self._slice.node_slice,
        )

    @cached_property
    def num_timestamps(self) -> int:
        r"""The total number of unique timestamps encountered over the dynamic graph."""
        return self._storage.get_num_timestamps(
            self._slice.start_time,
            self._slice.end_time,
            self._slice.node_slice,
        )

    @cached_property
    def node_feats(self) -> Optional[Tensor]:
        r"""The aggregated node features over the dynamic graph.

        Returns a Tensor.sparse_coo_tensor of size T x V x d_node or None if
        there are no node features on the dynamic graph.
        """
        return self._storage.get_node_feats(
            self._slice.start_time,
            self._slice.end_time,
            self._slice.node_slice,
        )

    @cached_property
    def edge_feats(self) -> Optional[Tensor]:
        r"""The aggregated edge features over the dynamic graph.

        Returns a Tensor.sparse_coo_tensor of size T x V x V x d_edge or None if
        there are no edge features on the dynamic graph.
        """
        return self._storage.get_edge_feats(
            self._slice.start_time,
            self._slice.end_time,
            self._slice.node_slice,
        )


@dataclass(slots=True)
class DGSliceTracker:
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    node_slice: Optional[Set[int]] = None
