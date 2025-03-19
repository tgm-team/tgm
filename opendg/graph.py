from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional, Tuple

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

        self._cache: Dict[str, Any] = {}

    def to_events(self) -> List[Event]:
        r"""Materialize the events in the DGraph."""
        return self._storage.to_events(
            start_time=self._cache.get('start_time'),
            end_time=self._cache.get('end_time'),
            node_slice=self._cache.get('node_slice'),
        )

    def slice(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        nodes: Optional[List[int]] = None,
    ) -> DGraph:
        r"""Create and return a new view by slicing the current DGraph.

        Args:
            start_time (Optional[int]): The start of the temporal slice.
            end_time (Optional[int]): The end of the temporal slice (inclusive).
            nodes: (Optional[List[int]]): The list of nodes to include in the slice.

        Returns:
            DGraph view of events satisfying the input constraints.
        """
        if start_time is not None and end_time is not None and start_time > end_time:
            raise ValueError(
                f'start_time ({start_time}) must be <= end_time ({end_time})'
            )

        def _max(seq: Tuple[Any, Any]) -> Optional[int]:
            return max(filter(lambda x: x is not None, seq), default=None)

        def _min(seq: Tuple[Any, Any]) -> Optional[int]:
            return max(filter(lambda x: x is not None, seq), default=None)

        dg = DGraph(self._storage, self.time_delta)
        dg._cache['start_time'] = _max((self._cache.get('start_time'), start_time))
        dg._cache['end_time'] = _min((self._cache.get('end_time'), end_time))
        dg._cache['node_slice'] = self._cache.get('node_slice')

        # TODO: Confirm that node slices affect start/end_time
        if nodes is not None:
            if dg._cache['node_slice'] is None:
                dg._cache['node_slice'] = set(nodes)
            else:
                dg._cache['node_slice'] &= set(nodes)
        return dg

    def __len__(self) -> int:
        r"""The number of timestamps in the dynamic graph."""
        return self.num_timestamps

    def __str__(self) -> str:
        return f'DGraph(storage={self._storage.__class__.__name__}, time_delta={self.time_delta})'

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

        Returns a Tensor.sparse_coo_tensor of size T x V x d_node or None if
        there are no node features on the dynamic graph.
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

        Returns a Tensor.sparse_coo_tensor of size T x V x V x d_edge or None if
        there are no edge features on the dynamic graph.
        """
        if self._cache.get('edge_feats') is None:
            self._cache['edge_feats'] = self._storage.get_edge_feats(
                self._cache.get('start_time'),
                self._cache.get('end_time'),
                self._cache.get('node_slice'),
            )
        return self._cache['edge_feats']
