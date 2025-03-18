import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

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
        if isinstance(data, DGStorage):
            self._storage = data
        else:
            events = data if isinstance(data, list) else read_events(data, **kwargs)
            self._storage = DGStorage(events)

        if not isinstance(time_delta, TimeDeltaDG):
            raise ValueError(f'bad time_delta type: {type(time_delta)}')
        self.time_delta = time_delta

        self._cache: Dict[str, Any] = {}

    def to_events(self) -> List[Event]:
        r"""Materialize the events in the DGraph.

        Returns:
            List of events in the current DGraph. Events are ordered by time, but event ordering
            within a single timestamp is not deterministic.
        """
        return self._storage.to_events(
            start_time=self._cache.get('start_time'),
            end_time=self._cache.get('end_time'),
            node_slice=self._cache.get('node_slice'),
        )

    def slice_time(
        self, start_time: Optional[int] = None, end_time: Optional[int] = None
    ) -> 'DGraph':
        r"""Extract temporal slice of the dynamic graph between start_time and end_time.

        If not specified, the start_time (resp. end_time) defaults to `self.start_time` (resp. `self.end_time`).
        The end_time parameter is exclusive, meaning the returned view contains all events up to, but not including
        those at time end_time. The returned view may or may not have events at the new temporal boundaries.

        Args:
            start_time (Optional[int]): The start of the temporal slice. If None, slices the graph with no lower bound on time.
            end_time (Optional[int]): The end of the temporal slice (inclusive). If None, slices the graph with no upper bound on time.

        Returns:
            DGraph view of events between start and end_time.
        """
        new_start_time, new_end_time = self._check_slice_time_args(start_time, end_time)
        new_end_time -= 1  # Because slicing is end range exclusive

        dg = DGraph(data=self._storage, time_delta=self.time_delta)
        dg._cache = dict(self._cache)  # Deep copy cache to avoid dict alias

        if self.start_time is not None and self.start_time > new_start_time:
            new_start_time = self.start_time

        if self.end_time is not None and self.end_time < new_end_time:
            new_end_time = self.end_time

        # Force cache refresh on the new copy if we actually sliced the graph
        if new_start_time != self.start_time or new_end_time != self.end_time:
            dg._cache.clear()
            dg._cache['node_slice'] = self._cache.get('node_slice')

        dg._cache['start_time'] = new_start_time
        dg._cache['end_time'] = new_end_time + 1  # End-range exclusive value in cache

        return dg

    def slice_nodes(self, nodes: List[int]) -> 'DGraph':
        r"""Extract topological slice of the dynamcic graph given the list of nodes.

        The returned view contains events which interact with the list of input nodes. This means
        the edges with at least one endpoint node in the input list are included in the returned view.
        The start_time and end_time of the returned view correspond to the minimum (resp. maximum) timestamp
        in the newly sliced graph (or None if the newly created view is empty).

        Args:
            nodes (List[int]): The list of node ids to slice from.

        Returns:
            DGraph copy of events related to the input nodes.
        """
        dg = DGraph(data=self._storage, time_delta=self.time_delta)

        if self._cache.get('node_slice') is None:
            self._cache['node_slice'] = set(range(self.num_nodes))

        # Take intersection of nodes
        dg._cache['node_slice'] = self._cache['node_slice'] & set(nodes)

        # Update start time
        start_time_with_node_slice = self._storage.get_start_time(
            dg._cache.get('node_slice')
        )
        if self.start_time is None:
            dg._cache['start_time'] = start_time_with_node_slice
        else:
            dg._cache['start_time'] = (
                max(start_time_with_node_slice, self._cache['start_time'])
                if start_time_with_node_slice is not None
                else self.start_time
            )

        # Update end time
        end_time_with_node_slice = self._storage.get_end_time(
            dg._cache.get('node_slice')
        )
        if self.end_time is None:
            dg._cache['end_time'] = end_time_with_node_slice
        else:
            dg._cache['end_time'] = (
                min(end_time_with_node_slice, self._cache['end_time'])
                if end_time_with_node_slice is not None
                else self._cache['end_time']
            )

        # Cache end-exclusive result
        if dg._cache['end_time'] is not None:
            dg._cache['end_time'] += 1

        return dg

    def __len__(self) -> int:
        r"""Returns the number of temporal length of the dynamic graph."""
        return self.num_timestamps

    def __str__(self) -> str:
        r"""Returns summary properties of the dynamic graph."""
        return f'Dynamic Graph Storage Engine ({self._storage.__class__.__name__}), Start Time: {self.start_time}, End Time: {self.end_time}, Nodes: {self.num_nodes}, Edges: {self.num_edges}, Timestamps: {self.num_timestamps}, Time Delta: {self.time_delta}'

    @property
    def start_time(self) -> Optional[int]:
        r"""The start time of the dynamic graph. None if the graph is empty."""
        if self._cache.get('start_time') is None:
            self._cache['start_time'] = self._storage.get_start_time(
                self._cache.get('node_slice')
            )
        return self._cache['start_time']

    @property
    def end_time(self) -> Optional[int]:
        r"""The end time of the dynamic graph. None, if the graph is empty."""
        if self._cache.get('end_time') is None:
            self._cache['end_time'] = self._storage.get_end_time(
                self._cache.get('node_slice')
            )
            # We cache the end_time + 1 so that all our time constrained queries
            # use the half-open interval: [start_time, end_time + 1) = [start_time, end_time].
            # If we considered everything end-time inclusive, this would not be needed.
            if self._cache['end_time'] is not None:
                self._cache['end_time'] += 1

        if self._cache['end_time'] is not None:
            # Since our cache stores end_time + 1, we subtract back one to yield the
            # actual end time in our DG.
            return self._cache['end_time'] - 1
        return self._cache['end_time']

    @property
    def num_nodes(self) -> int:
        r"""The total number of unique nodes encountered over the dynamic graph."""
        if self._cache.get('num_nodes') is None:
            self._cache['node_slice'] = self._storage.get_nodes(
                self._cache.get('start_time'),
                self._cache.get('end_time'),
                self._cache.get('node_slice'),
            )
            if len(self._cache['node_slice']):
                self._cache['num_nodes'] = max(self._cache['node_slice']) + 1
            else:
                self._cache['num_nodes'] = 0
        return self._cache['num_nodes']

    @property
    def num_edges(self) -> int:
        r"""The total number of unique edges encountered over the dynamic graph."""
        if self._cache.get('num_edges') is None:
            self._cache['num_edges'] = self._storage.get_num_edges(
                self._cache.get('start_time'),
                self._cache.get('end_time'),
                self._cache.get('node_slice'),
            )
        return self._cache['num_edges']

    @property
    def num_timestamps(self) -> int:
        r"""The total number of unique timestamps encountered over the dynamic graph."""
        if self._cache.get('num_timestamps') is None:
            self._cache['num_timestamps'] = self._storage.get_num_timestamps(
                self._cache.get('start_time'),
                self._cache.get('end_time'),
                self._cache.get('node_slice'),
            )
        return self._cache['num_timestamps']

    @property
    def node_feats(self) -> Optional[Tensor]:
        r"""The aggregated node features over the dynamic graph.

        Returns a tensor.sparse_coo_tensor of size T x V x d where

        - T = Number of timestamps
        - V = Number of nodes
        - d = Node feature dimension
        or None if there are no node features on the dynamic graph.

        """
        if self._cache.get('node_feats') is None:
            self._cache['node_feats'] = self._storage.get_node_feats(
                self._cache.get('start_time'),
                self._cache.get('end_time'),
                self._cache.get('node_slice'),
            )
        return self._cache['node_feats']

    @property
    def edge_feats(self) -> Optional[Tensor]:
        r"""The aggregated edge features over the dynamic graph.

        Returns a tensor.sparse_coo_tensor of size T x V x V x d where

        - T = Number of timestamps
        - V = Number of nodes
        - d = Edge feature dimension

        or None if there are no edge features on the dynamic graph.
        """
        if self._cache.get('edge_feats') is None:
            self._cache['edge_feats'] = self._storage.get_edge_feats(
                self._cache.get('start_time'),
                self._cache.get('end_time'),
                self._cache.get('node_slice'),
            )
        return self._cache['edge_feats']

    def _check_slice_time_args(
        self, start_time: Optional[int], end_time: Optional[int]
    ) -> Tuple[Union[int, float], Union[int, float]]:
        new_start_time = start_time if start_time is not None else float('-inf')
        new_end_time = end_time if end_time is not None else float('inf')
        if new_start_time > new_end_time:
            raise ValueError(
                f'Bad slice: start_time must be <= end_time but received: start_time ({new_start_time}) > end_time ({new_end_time})'
            )
        return new_start_time, new_end_time
