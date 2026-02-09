from __future__ import annotations

import logging
from dataclasses import replace
from functools import cached_property
from typing import Any, Optional, Tuple

import torch
from torch import Tensor

from tgm.util.logging import _get_logger, _logged_cached_property, log_latency

from ._storage import DGSliceTracker, DGStorage
from .batch import DGBatch
from .timedelta import TimeDeltaDG

logger = _get_logger(__name__)


class DGraph:
    """Dynamic Graph object providing a view over a DGStorage backend.

    This class allows efficient slicing, batching, and materialization of
    temporal/dynamic graph data. It exposes properties for node and edge counts,
    features, timestamps, and supports device placement for tensors.

    Args:
        data (DGData): The source DGData object to construct the dynamic graph view.
        device (str | torch.device, optional): The device to place tensors on. Defaults to 'cpu'.

    Raises:
        TypeError: If `data` is not a DGData instance.

    Note:
        - Slicing operations (`slice_events` or `slice_time`) return a new `DGraph`
          view sharing the underlying storage; they do not copy data unless
          explicitly materialized via `materialize()`.
        - Cached properties (e.g., `num_nodes`, `edges`, `static_node_x`) are
          computed on first access and then stored. Modifying the underlying
          storage does not automatically invalidate these cached values.
        - `materialize()` returns dense tensors for src, dst, time, and optionally
          dynamic node and edge features. This operation may be memory-intensive
          for large graphs.
        - `slice_events` uses **event indices** (position in chronological order),
          while `slice_time` uses **timestamp values**. For `slice_time`, the
          end_time is exclusive but internally adjusted to include events at
          end_time - 1.
        - `num_nodes` counts the maximum node ID in the slice + 1; this may differ
          from the number of nodes in the underlying DGData if slicing excludes
          some nodes.
        - `static_node_x` shape is `(num_nodes_global, d_node_static)` and
          is **independent of slices**, whereas `node_x` reflects the
          current slice.
        - Operations such as `.to(device)` or slicing create **views**; data is not
          copied unless explicitly materialized.
    """

    def __init__(self, data: 'DGData', device: str | torch.device = 'cpu') -> None:  # type: ignore
        from tgm.data import DGData  # Avoid circular dependency

        if not isinstance(data, DGData):
            raise TypeError(f'DGraph must be initialized with DGData, got {type(data)}')

        self._time_delta = data.time_delta
        self._storage = DGStorage(data)
        self._device = torch.device(device)
        self._slice = DGSliceTracker()

        logger.debug(
            'Created DGraph with device=%s, time_delta=%s, device, data.time_delta'
        )

    @log_latency(level=logging.DEBUG)
    def materialize(self, materialize_features: bool = True) -> DGBatch:
        """Materialize the current DGraph slice into a dense `DGBatch`.

        Args:
            materialize_features (bool, optional): If True, includes dynamic node
                features, node IDs/times, and edge features. Defaults to True.

        Returns:
            DGBatch: A batch containing src, dst, timestamps, and optionally
                features from the current slice.
        """
        batch = DGBatch(self.edge_src, self.edge_dst, self.edge_time)
        if materialize_features and self.node_x is not None:
            batch.node_x_time, batch.node_x_nids = self.node_x._indices()
            batch.node_x_nids = batch.node_x_nids.to(torch.int32)  # type: ignore
            batch.node_x = self.node_x._values()

        if materialize_features and self.edge_x is not None:
            batch.edge_x = self.edge_x

        if materialize_features and self.node_y is not None:
            batch.node_y_time, batch.node_y_nids = self.node_y._indices()
            batch.node_y_nids = batch.node_y_nids.to(torch.int32)  # type: ignore
            batch.node_y = self.node_y._values()

        if self.edge_type is not None:
            batch.edge_type = self.edge_type

        logger.debug(
            'Materialized DGraph slice: %d edge events, %d node events, %d node labels',
            batch.edge_src.numel(),
            0 if batch.node_x_nids is None else batch.node_x_nids.numel(),
            0 if batch.node_y_nids is None else batch.node_y_nids.numel(),
        )
        return batch

    def slice_events(
        self, start_idx: Optional[int] = None, end_idx: Optional[int] = None
    ) -> DGraph:
        """Return a new DGraph view sliced by event indices (end_idx exclusive).

        Args:
            start_idx (int, optional): Starting event index.
            end_idx (int, optional): Ending event index (exclusive).

        Raises:
            ValueError: If start_idx > end_idx.
        """
        if start_idx is not None and end_idx is not None and start_idx > end_idx:
            raise ValueError(f'start_idx ({start_idx}) must be <= end_idx ({end_idx})')

        slice = replace(self._slice)
        slice.start_idx = self._maybe_max(start_idx, slice.start_idx)
        slice.end_idx = self._maybe_min(end_idx, slice.end_idx)
        return DGraph._from_storage(self._storage, self.time_delta, self.device, slice)

    def slice_time(
        self, start_time: Optional[int] = None, end_time: Optional[int] = None
    ) -> DGraph:
        """Return a new DGraph view sliced by timestamps (end_time exclusive).

        Args:
            start_time (int, optional): Starting timestamp.
            end_time (int, optional): Ending timestamp (exclusive).

        Raises:
            ValueError: If start_time > end_time.
        """
        if start_time is not None and end_time is not None and start_time > end_time:
            raise ValueError(
                f'start_time ({start_time}) must be <= end_time ({end_time})'
            )
        if end_time is not None:
            end_time -= 1

        slice = replace(self._slice)
        slice.start_time = self._maybe_max(start_time, slice.start_time)
        slice.end_time = self._maybe_min(end_time, slice.end_time)
        return DGraph._from_storage(self._storage, self.time_delta, self.device, slice)

    def __len__(self) -> int:
        """The number of timestamps in the dynamic graph."""
        return self.num_timestamps

    def __str__(self) -> str:
        return f'DGraph(storage={self._storage.__class__.__name__}, time_delta={self.time_delta}, device={self.device})'

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def time_delta(self) -> TimeDeltaDG:
        return self._time_delta  # type: ignore

    def to(self, device: str | torch.device) -> DGraph:
        """Return a copy of the DGraph view on a different device.

        Args:
            device (str | torch.device): The target device.

        Returns:
            DGraph: A new view on the specified device.
        """
        logger.debug('Moving DGraph to device %s', device)
        device = torch.device(device)
        slice = replace(self._slice)
        return DGraph._from_storage(self._storage, self.time_delta, device, slice)

    @cached_property
    def start_time(self) -> Optional[int]:
        """The start time of the dynamic graph. None if the graph is empty."""
        if self._slice.start_time is None:
            self._slice.start_time = self._storage.get_start_time(self._slice)
        return self._slice.start_time

    @cached_property
    def end_time(self) -> Optional[int]:
        """The end time of the dynamic graph. None, if the graph is empty."""
        if self._slice.end_time is None:
            self._slice.end_time = self._storage.get_end_time(self._slice)
        return self._slice.end_time

    @_logged_cached_property
    def num_nodes(self) -> int:
        """The total number of unique nodes encountered over the dynamic graph."""
        nodes = self._storage.get_nodes(self._slice)
        return max(nodes) + 1 if len(nodes) else 0

    @_logged_cached_property
    def num_node_events(self) -> int:
        """The total number of node events in the dynamic graph."""
        return len(self.node_x_time)

    @_logged_cached_property
    def num_node_labels(self) -> int:
        """The total number of node labels in the dynamic graph."""
        return len(self.node_y_time)

    @_logged_cached_property
    def num_edge_events(self) -> int:
        """The total number of edge events in the dynamic graph."""
        return len(self.edge_time)

    @_logged_cached_property
    def num_timestamps(self) -> int:
        """The total number of unique timestamps encountered over the dynamic graph."""
        return self._storage.get_num_timestamps(self._slice)

    @_logged_cached_property
    def num_events(self) -> int:
        """The total number of events encountered over the dynamic graph."""
        return self._storage.get_num_events(self._slice)

    @_logged_cached_property
    def _edges_cpu(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self._storage.get_edges(self._slice)

    @property
    def edge_src(self) -> Tensor:
        """The edge src tensor over the dynamic graph."""
        src, _, _ = self._edges_cpu
        return src.to(self.device)

    @property
    def edge_dst(self) -> Tensor:
        """The edge dst tensor over the dynamic graph."""
        _, dst, _ = self._edges_cpu
        return dst.to(self.device)

    @property
    def edge_time(self) -> Tensor:
        """The timestamps associated with the edge events over the dynamic graph."""
        _, _, time = self._edges_cpu
        return time.to(self.device)

    @_logged_cached_property
    def _edge_x_cpu(self) -> Optional[Tensor]:
        return self._storage.get_edge_x(self._slice)

    @property
    def edge_x(self) -> Optional[Tensor]:
        """The aggregated edge features over the dynamic graph.

        If edge features exist, returns a tensor of shape (T x V x V x d_edge).
        """
        feats = self._edge_x_cpu
        if feats is not None:
            feats = feats.to(self.device)
        return feats

    @_logged_cached_property
    def _edge_type_cpu(self) -> Optional[Tensor]:
        return self._storage.get_edge_type(self._slice)

    @property
    def edge_type(self) -> Optional[Tensor]:
        """The aggregated edge type over the dynamic graph.

        If edge type exist, returns a tensor of shape (T x V x V).
        """
        edge_type = self._edge_type_cpu
        if edge_type is not None:
            edge_type = edge_type.to(self.device)
        return edge_type

    @_logged_cached_property
    def _node_events_cpu(self) -> Tuple[Tensor, Tensor]:
        return self._storage.get_node_events(self._slice)

    @property
    def node_x_nids(self) -> Tensor:
        """The node ids for associated with the node events over the dynamic graph."""
        node_ids, _ = self._node_events_cpu
        return node_ids.to(self.device)

    @property
    def node_x_time(self) -> Tensor:
        """The timestamps associated with the node events over the dynamic graph."""
        _, node_time = self._node_events_cpu
        return node_time.to(self.device)

    @_logged_cached_property
    def _node_x_cpu(self) -> Optional[Tensor]:
        return self._storage.get_node_x(self._slice)

    @property
    def node_x(self) -> Optional[Tensor]:
        """The aggregated dynamic node features over the dynamic graph.

        If dynamic node features exist, returns a Tensor.sparse_coo_tensor(T x V x d_node_dynamic).
        """
        feats = self._node_x_cpu
        if feats is not None:
            feats = feats.to(self.device)
        return feats

    @cached_property
    def _node_type_cpu(self) -> Optional[Tensor]:
        return self._storage.get_node_type()

    @property
    def node_type(self) -> Optional[Tensor]:
        """If node types exist, returns a dense Tensor(num_nodes_global).

        Note:
            - num_nodes_global is the global number of nodes from the underlying DGData and it will be independent of the slice.
        """
        node_type = self._node_type_cpu
        if node_type is not None:
            node_type = node_type.to(self.device)
        return node_type

    @_logged_cached_property
    def _node_labels_cpu(self) -> Tuple[Tensor, Tensor]:
        return self._storage.get_node_labels(self._slice)

    @property
    def node_y_nids(self) -> Tensor:
        """The node ids for associated with the node labels over the dynamic graph."""
        node_ids, _ = self._node_labels_cpu
        return node_ids.to(self.device)

    @property
    def node_y_time(self) -> Tensor:
        """The timestamps associated with the node labels over the dynamic graph."""
        _, node_time = self._node_labels_cpu
        return node_time.to(self.device)

    @_logged_cached_property
    def _node_y_cpu(self) -> Optional[Tensor]:
        return self._storage.get_node_y(self._slice)

    @property
    def node_y(self) -> Optional[Tensor]:
        """The aggregated dynamic node labels over the dynamic graph.

        If dynamic node labels exist, returns a Tensor.sparse_coo_tensor(T x V x d_node_label).
        """
        feats = self._node_y_cpu
        if feats is not None:
            feats = feats.to(self.device)
        return feats

    @cached_property
    def _static_node_x_cpu(self) -> Optional[Tensor]:
        return self._storage.get_static_node_x()

    @property
    def static_node_x(self) -> Optional[Tensor]:
        """If static node features exist, returns a dense Tensor(num_nodes_global x d_node_static).

        Note:
            - num_nodes_global is the global number of nodes from the underlying DGData and it will be independent of the slice.
        """
        feats = self._static_node_x_cpu
        if feats is not None:
            feats = feats.to(self.device)
        return feats

    @cached_property
    def static_node_x_dim(self) -> Optional[int]:
        """Static Node feature dimension or None if not Node features on the Graph."""
        return self._storage.get_static_node_x_dim()

    @cached_property
    def node_x_dim(self) -> Optional[int]:
        """Dynamic Node feature dimension or None if not Node features on the Graph."""
        return self._storage.get_node_x_dim()

    @cached_property
    def node_y_dim(self) -> Optional[int]:
        """Dynamic Node label feature dimension or None if no Node labels on the Graph."""
        return self._storage.get_node_y_dim()

    @cached_property
    def edge_x_dim(self) -> Optional[int]:
        """Edge feature dimension or None if not Node features on the Graph."""
        return self._storage.get_edge_x_dim()

    @staticmethod
    def _maybe_max(a: Any, b: Any) -> Optional[int]:
        if a is not None and b is not None:
            return max(a, b)
        return a if b is None else b if a is None else None

    @staticmethod
    def _maybe_min(a: Any, b: Any) -> Optional[int]:
        if a is not None and b is not None:
            return min(a, b)
        return a if b is None else b if a is None else None

    @classmethod
    def _from_storage(
        cls,
        storage: DGStorage,
        time_delta: TimeDeltaDG,
        device: torch.device,
        slice: DGSliceTracker,
    ) -> DGraph:
        logger.debug('Creating a DGraph view with slice: %s', slice)
        obj = cls.__new__(cls)
        obj._storage = storage
        obj._time_delta = time_delta
        obj._device = device
        obj._slice = slice
        return obj
