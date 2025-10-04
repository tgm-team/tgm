from __future__ import annotations

import logging
from dataclasses import replace
from functools import cached_property
from typing import Any, Optional, Set, Tuple

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
        - Cached properties (e.g., `num_nodes`, `edges`, `static_node_feats`) are
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
        - `static_node_feats` shape is `(num_nodes_global, d_node_static)` and
          is **independent of slices**, whereas `dynamic_node_feats` reflects the
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
        batch = DGBatch(*self.edges)
        if materialize_features and self.dynamic_node_feats is not None:
            batch.node_times, batch.node_ids = self.dynamic_node_feats._indices()
            batch.node_ids = batch.node_ids.to(torch.int32)  # type: ignore
            batch.dynamic_node_feats = self.dynamic_node_feats._values()
        if materialize_features and self.edge_feats is not None:
            batch.edge_feats = self.edge_feats

        logger.debug(
            'Materialized DGraph slice: %d edge events, %d node events',
            batch.src.numel(),
            0 if batch.node_ids is None else batch.node_ids.numel(),
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
    def num_edges(self) -> int:
        """The total number of unique edges encountered over the dynamic graph."""
        src, *_ = self.edges
        return len(src)

    @_logged_cached_property
    def num_timestamps(self) -> int:
        """The total number of unique timestamps encountered over the dynamic graph."""
        return self._storage.get_num_timestamps(self._slice)

    @_logged_cached_property
    def num_events(self) -> int:
        """The total number of events encountered over the dynamic graph."""
        return self._storage.get_num_events(self._slice)

    @_logged_cached_property
    def nodes(self) -> Set[int]:
        """The set of node ids over the dynamic graph."""
        return self._storage.get_nodes(self._slice)

    @_logged_cached_property
    def _edges_cpu(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self._storage.get_edges(self._slice)

    @property
    def edges(self) -> Tuple[Tensor, Tensor, Tensor]:
        """The src, dst, time tensors over the dynamic graph."""
        src, dst, time = self._edges_cpu
        return src.to(self.device), dst.to(self.device), time.to(self.device)

    @cached_property
    def _static_node_feats_cpu(self) -> Optional[Tensor]:
        return self._storage.get_static_node_feats()

    @property
    def static_node_feats(self) -> Optional[Tensor]:
        """If static node features exist, returns a dense Tensor(num_nodes_global x d_node_static).

        Note:
            - num_nodes_global is the global number of nodes from the underlying DGData and it will be independent of the slice.
        """
        feats = self._static_node_feats_cpu
        if feats is not None:
            feats = feats.to(self.device)
        return feats

    @_logged_cached_property
    def _dynamic_node_feats_cpu(self) -> Optional[Tensor]:
        return self._storage.get_dynamic_node_feats(self._slice)

    @property
    def dynamic_node_feats(self) -> Optional[Tensor]:
        """The aggregated dynamic node features over the dynamic graph.

        If dynamic node features exist, returns a Tensor.sparse_coo_tensor(T x V x d_node_dynamic).
        """
        feats = self._dynamic_node_feats_cpu
        if feats is not None:
            feats = feats.to(self.device)
        return feats

    @_logged_cached_property
    def _edge_feats_cpu(self) -> Optional[Tensor]:
        return self._storage.get_edge_feats(self._slice)

    @property
    def edge_feats(self) -> Optional[Tensor]:
        """The aggregated edge features over the dynamic graph.

        If edge features exist, returns a tensor of shape (T x V x V x d_edge).
        """
        feats = self._edge_feats_cpu
        if feats is not None:
            feats = feats.to(self.device)
        return feats

    @cached_property
    def static_node_feats_dim(self) -> Optional[int]:
        """Static Node feature dimension or None if not Node features on the Graph."""
        return self._storage.get_static_node_feats_dim()

    @cached_property
    def dynamic_node_feats_dim(self) -> Optional[int]:
        """Dynamic Node feature dimension or None if not Node features on the Graph."""
        return self._storage.get_dynamic_node_feats_dim()

    @cached_property
    def edge_feats_dim(self) -> Optional[int]:
        """Edge feature dimension or None if not Node features on the Graph."""
        return self._storage.get_edge_feats_dim()

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
