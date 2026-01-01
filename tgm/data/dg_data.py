from __future__ import annotations

import copy
import csv
import logging
import pathlib
import warnings
from dataclasses import dataclass, fields, replace
from typing import Any, List, Tuple

import numpy as np
import torch
from torch import Tensor

from tgm.constants import PADDED_NODE_ID
from tgm.core import TGB_SEQ_TIME_DELTAS, TGB_TIME_DELTAS, TimeDeltaDG
from tgm.data.split import SplitStrategy, TemporalRatioSplit, TGBSplit
from tgm.exceptions import (
    EmptyGraphError,
    EventOrderedConversionError,
    InvalidDiscretizationError,
    InvalidNodeIDError,
)
from tgm.util.logging import _get_logger, log_latency

logger = _get_logger(__name__)


@dataclass
class DGData:
    """Container for dynamic graph data to be ingested by `DGStorage`.

    Stores edge and node events, their timestamps, features, and optional split strategy.
    Provides methods to split, discretize, and clone the data.

    Attributes:
        time_delta (TimeDeltaDG | str): Time granularity of the graph.
        timestamps (Tensor): 1D tensor of all event timestamps [num_edge_events + num_node_events].
        edge_event_idx (Tensor): Indices of edge events within `timestamps`.
        edge_index (Tensor): Edge connections [num_edge_events, 2].
        edge_feats (Tensor | None): Optional edge features [num_edge_events, D_edge].
        node_event_idx (Tensor | None): Indices of node events within `timestamps`.
        node_ids (Tensor | None): Node IDs corresponding to node events [num_node_events].
        dynamic_node_feats (Tensor | None): Node features over time [num_node_events, D_node_dynamic].
        static_node_feats (Tensor | None): Node features invariant over time [num_nodes, D_node_static].
        edge_type (Tensor | None) : Type of relation of each edge event in edge_index [num_edge_events].
        node_type (Tensor | None) : Type of each node [num_nodes].

    Raises:
        InvalidNodeIDError: If an edge or node ID match `PADDED_NODE_ID`.
        ValueError: If any data attributes have non-well defined tensor shapes.
        EmptyGraphError: If attempting to initialize an empty graph.

    Notes:
        - Timestamps must be non-negative and sorted; DGData will sort automatically if necessary.
        - Cloning creates a deep copy of tensors to prevent in-place modifications.
        - Edge type is only applicable for Heterogeneous & Knowledge graph.
        - Node type is only applicable for Knowledge graph.
    """

    time_delta: TimeDeltaDG | str
    timestamps: Tensor  # [num_events]

    edge_event_idx: Tensor  # [num_edge_events]
    edge_index: Tensor  # [num_edge_events, 2]
    edge_feats: Tensor | None = None  # [num_edge_events, D_edge]

    node_event_idx: Tensor | None = None  # [num_node_events]
    node_ids: Tensor | None = None  # [num_node_events]
    dynamic_node_feats: Tensor | None = None  # [num_node_events, D_node_dynamic]

    static_node_feats: Tensor | None = None  # [num_nodes, D_node_static]
    edge_type: Tensor | None = None  # [num_edge_events]
    node_type: Tensor | None = None  # [num_nodes]

    _split_strategy: SplitStrategy | None = None

    def __post_init__(self) -> None:
        if isinstance(self.time_delta, str):
            self.time_delta = TimeDeltaDG(self.time_delta)

        def _assert_is_tensor(x: Any, name: str) -> None:
            if not isinstance(x, Tensor):
                raise TypeError(f'{name} must be a Tensor, got: {type(x)}')
            if torch.isnan(x).any():
                raise ValueError(f'{name} contains NaN values')

        def _assert_tensor_is_integral(x: Tensor, name: str) -> None:
            int_types = [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]
            if x.dtype not in int_types:
                raise TypeError(f'{name} must have integer dtype but got: {x.dtype}')

        def _maybe_cast_float_tensor(x: Tensor, name: str) -> Tensor:
            if x.dtype == torch.float64:
                logger.warning(
                    'Downcasting %s from torch.float64 to torch.float32', name
                )
                warnings.warn(
                    f'Downcasting {name} from torch.float64 to torch.float32',
                    UserWarning,
                )
                return x.to(torch.float32)
            return x.to(torch.float32) if x.dtype != torch.float32 else x

        def _maybe_cast_integral_tensor(x: Tensor, name: str) -> Tensor:
            if x.dtype == torch.int64:
                logger.warning('Downcasting %s from torch.int64 to torch.int32', name)
                warnings.warn(
                    f'Downcasting {name} from torch.int64 to torch.int32', UserWarning
                )
                return x.to(torch.int32)
            return x.to(torch.int32) if x.dtype != torch.int32 else x

        max_int32_capacity = torch.iinfo(torch.int32).max

        # Validate timestamps
        _assert_is_tensor(self.timestamps, 'timestamps')
        _assert_tensor_is_integral(self.timestamps, 'timestamps')
        if not torch.all(self.timestamps >= 0):
            raise ValueError('timestamps must all be non-negative')
        if not torch.all(self.timestamps < max_int32_capacity):
            raise ValueError(
                f'timestamps exceed the int32 limit ({max_int32_capacity}). '
                'TGM does not yet support graphs this large.'
            )
        if self.timestamps.dtype != torch.int64:  # This can only be an upcast
            logger.debug('Upcasting global timestamps to torch.int64')
            self.timestamps = self.timestamps.to(torch.int64)

        # Ensure our data does not overflow int32 capacity
        if len(self.timestamps) > max_int32_capacity:
            raise ValueError(
                f'Number of events ({len(self.timestamps)}) exceeds the int32 limit '
                f'({max_int32_capacity}). TGM does not yet support graphs this large.'
            )

        # Validate edge index
        _assert_is_tensor(self.edge_index, 'edge_index')
        _assert_tensor_is_integral(self.edge_index, 'edge_index')
        if self.edge_index.ndim != 2 or self.edge_index.shape[1] != 2:
            raise ValueError(
                f'edge_index must have shape [num_edges, 2], got: {self.edge_index.shape}',
            )
        if torch.any(self.edge_index == PADDED_NODE_ID):
            raise InvalidNodeIDError(
                f'Edge events contains node ids matching PADDED_NODE_ID: {PADDED_NODE_ID}, which is used to mark invalid neighbors. Try remapping all node ids to positive integers.'
            )
        if not torch.all(self.edge_index < max_int32_capacity):
            raise InvalidNodeIDError(
                f'Edge events contains node ids that exceed the int32 limit ({max_int32_capacity}). '
                'TGM does not yet support graphs this large.'
            )
        self.edge_index = _maybe_cast_integral_tensor(self.edge_index, 'edge_index')

        num_edges = self.edge_index.shape[0]
        if num_edges == 0:
            raise EmptyGraphError('TGM does not support graphs without edge events')

        # Validate edge event idx
        _assert_is_tensor(self.edge_event_idx, 'edge_event_idx')
        _assert_tensor_is_integral(self.edge_event_idx, 'edge_event_idx')
        # Safe downcast since we ensured len(timestamps) fits in int32
        self.edge_event_idx = self.edge_event_idx.int()

        # Validate edge features
        if self.edge_feats is not None:
            _assert_is_tensor(self.edge_feats, 'edge_feats')
            if self.edge_feats.ndim != 2 or self.edge_feats.shape[0] != num_edges:
                raise ValueError(
                    'edge_feats must have shape [num_edges, D_edge], '
                    f'got {num_edges} edges and shape {self.edge_feats.shape}'
                )
            self.edge_feats = _maybe_cast_float_tensor(self.edge_feats, 'edge_feats')

        # Validate node event idx
        num_node_events = 0
        if self.node_event_idx is not None:
            _assert_is_tensor(self.node_event_idx, 'node_event_idx')
            _assert_tensor_is_integral(self.node_event_idx, 'node_event_idx')

            # Safe downcast since we ensured len(timestamps) fits in int32
            self.node_event_idx = self.node_event_idx.int()

            num_node_events = self.node_event_idx.shape[0]
            if num_node_events == 0:
                raise ValueError(
                    'node_event_idx is an empty tensor, please double-check your inputs'
                )

            # Validate node ids
            _assert_is_tensor(self.node_ids, 'node_ids')
            _assert_tensor_is_integral(self.node_ids, 'node_ids')  # type: ignore
            if self.node_ids.ndim != 1 or self.node_ids.shape[0] != num_node_events:  # type: ignore
                raise ValueError(
                    'node_ids must have shape [num_node_events], ',
                    f'got {num_node_events} node events and shape {self.node_ids.shape}',  # type: ignore
                )
            if torch.any(self.node_ids == PADDED_NODE_ID):  # type: ignore
                raise InvalidNodeIDError(
                    f'Node events contains node ids matching PADDED_NODE_ID: {PADDED_NODE_ID}, which is used to mark invalid neighbors. Try remapping all node ids to positive integers.'
                )
            if not torch.all(self.node_ids < max_int32_capacity):  # type: ignore
                raise InvalidNodeIDError(
                    f'Node events contains node ids that exceed the int32 limit ({max_int32_capacity}). '
                    'TGM does not yet support graphs this large.'
                )
            self.node_ids = _maybe_cast_integral_tensor(self.node_ids, 'node_ids')  # type: ignore

            # Validate dynamic node features (could be None)
            if self.dynamic_node_feats is not None:
                _assert_is_tensor(self.dynamic_node_feats, 'dynamic_node_feats')
                if (
                    self.dynamic_node_feats.ndim != 2
                    or self.dynamic_node_feats.shape[0] != num_node_events
                ):
                    raise ValueError(
                        'dynamic_node_feats must have shape [num_node_events, D_node_dynamic], '
                        f'got {num_node_events} node events and shape {self.dynamic_node_feats.shape}'
                    )
                self.dynamic_node_feats = _maybe_cast_float_tensor(
                    self.dynamic_node_feats, 'dynamic_node_feats'
                )

        # Validate static node features
        num_nodes = torch.max(self.edge_index).item() + 1  # 0-indexed
        if self.node_ids is not None:
            num_nodes = max(num_nodes, torch.max(self.node_ids).item() + 1)  # 0-indexed

        if self.static_node_feats is not None:
            _assert_is_tensor(self.static_node_feats, 'static_node_feats')
            if self.static_node_feats.ndim != 2:
                raise ValueError(
                    f'static_node_feats must be a 2D tensor of shape [N, D_node_static], '
                    f'but got ndim={self.static_node_feats.ndim} with shape {self.static_node_feats.shape}'
                )

            if self.static_node_feats.shape[0] < num_nodes:
                raise ValueError(
                    f'static_node_feats has shape {self.static_node_feats.shape}, '
                    f'but the data requires features for at least {num_nodes} nodes. '
                    f'The first dimension ({self.static_node_feats.shape[0]}) must be >= num_nodes ({num_nodes}).'
                )
            self.static_node_feats = _maybe_cast_float_tensor(
                self.static_node_feats, 'static_node_feats'
            )

        # Validate edge type for Knowledge and Heterogeneous
        if self.edge_type is not None:
            _assert_is_tensor(self.edge_type, 'edge_type')
            _assert_tensor_is_integral(self.edge_type, 'edge_type')
            if self.edge_type.ndim != 1 or self.edge_type.shape[0] != num_edges:
                raise ValueError(
                    'edge_type must have shape [num_edges], '
                    f'got {num_edges} edges and shape {self.edge_type.shape}'
                )
            _maybe_cast_integral_tensor(self.edge_type, 'edge_type')

        # Validate node type for Heterogeneous
        if self.node_type is not None:
            _assert_is_tensor(self.node_type, 'node_type')
            _assert_tensor_is_integral(self.node_type, 'node_type')
            if self.node_type.ndim != 1 or self.node_type.shape[0] < num_nodes:
                raise ValueError(
                    'node_type must have shape [num_nodes], '
                    f'got {num_nodes} nodes and shape {self.node_type.shape}'
                )
            _maybe_cast_integral_tensor(self.node_type, 'node_type')

        # Ensure timestamps match number of global events
        if (
            self.timestamps.ndim != 1
            or self.timestamps.shape[0] != num_edges + num_node_events
        ):
            raise ValueError(
                'timestamps must have shape [num_edges + num_node_events], '
                f'got {num_edges} edges, {num_node_events} node_events, shape: {self.timestamps.shape}. '
                'Please double-check the edge and node timestamps you provided. If this is not resolved '
                'raise an issue and provide instructions on how to reproduce your the error'
            )

        # Sort if necessary
        if not torch.all(torch.diff(self.timestamps) >= 0):
            logger.warning(
                'Timestamps in DGData are not globally sorted. Reordering all events '
                '(edge_index, edge_feats, node_ids, etc.) to match sorted time order'
            )

            # Sort timestamps
            sort_idx = torch.argsort(self.timestamps).int()
            inverse_sort_idx = torch.empty_like(sort_idx)
            inverse_sort_idx[sort_idx] = torch.arange(len(sort_idx), dtype=torch.int32)
            self.timestamps = self.timestamps[sort_idx]

            # Update global event indices
            self.edge_event_idx = inverse_sort_idx[self.edge_event_idx]
            if self.node_event_idx is not None:
                self.node_event_idx = inverse_sort_idx[self.node_event_idx]

            # Reorder edge-specific data
            edge_order = torch.argsort(self.edge_event_idx)
            self.edge_index = self.edge_index[edge_order]
            if self.edge_feats is not None:
                self.edge_feats = self.edge_feats[edge_order]

            if self.edge_type is not None:
                self.edge_type = self.edge_type[edge_order]

            # Reorder node-specific data
            if self.node_event_idx is not None:
                node_order = torch.argsort(self.node_event_idx)
                self.node_ids = self.node_ids[node_order]  # type: ignore
                if self.dynamic_node_feats is not None:
                    self.dynamic_node_feats = self.dynamic_node_feats[node_order]

    def split(self, strategy: SplitStrategy | None = None) -> Tuple[DGData, ...]:
        """Split the dataset according to a strategy.

        Args:
            strategy (SplitStrategy | None): Optional strategy to override the
                default. If None, uses `_split_strategy` or defaults to `TemporalRatioSplit`.

        Returns:
            Tuple[DGData, ...]: Split datasets (train/val/test).

        Raises:
            ValueError: If attempting to override the split strategy for TGB datasets.

        Notes:
            - Splits preserve the underlying storage; only indices are filtered.
        """
        strategy = strategy or self._split_strategy or TemporalRatioSplit()

        if (
            isinstance(self._split_strategy, TGBSplit)
            and strategy is not self._split_strategy
        ):
            raise ValueError('Cannot override split strategy for TGB datasets')

        return strategy.apply(self)

    @log_latency(level=logging.DEBUG)
    def discretize(
        self, time_delta: TimeDeltaDG | str | None, reduce_op: str = 'first'
    ) -> DGData:
        """Return a copy of the dataset discretized to a coarser time granularity.

        Args:
            time_delta (TimeDeltaDG | str | None): Target time granularity.
            reduce_op (str): Aggregation method for multiple events per bucket. Default 'first'.

        Returns:
            DGData: New dataset with discretized timestamps and features.

        Raises:
            EventOrderedConversionError: If discretization is incompatible with event-ordered granularity
            InvalidDiscretizationError: If the target granularity is finer than the current granularity.
        """
        if isinstance(time_delta, str):
            time_delta = TimeDeltaDG(time_delta)
        logger.debug(
            'Discretizing from %s to %s, reduce_op: %s',
            self.time_delta,
            time_delta,
            reduce_op,
        )

        if time_delta is None or self.time_delta == time_delta:
            return self.clone()  # Deepcopy
        if self.time_delta.is_event_ordered or time_delta.is_event_ordered:  # type: ignore
            raise EventOrderedConversionError(
                'Cannot discretize a graph with event-ordered time granularity'
            )
        if self.time_delta.is_coarser_than(time_delta):  # type: ignore
            raise InvalidDiscretizationError(
                f'Cannot discretize to {time_delta} which is strictly'
                f'finer than the self.time_delta: {self.time_delta}'
            )

        valid_ops = ['first']
        if reduce_op not in valid_ops:
            raise ValueError(
                f'Unknown reduce_op: {reduce_op}, expected one of: {valid_ops}'
            )

        # Note: If we simply return a new storage this will 2x our peak memory since the GC
        # won't be able to clean up the current graph storage while `self` is alive.
        time_factor = self.time_delta.convert(time_delta)  # type: ignore
        # Doing time conversion in 64-bit to avoid numerical issues with float cast
        buckets = (self.timestamps.to(torch.float64) * time_factor).floor().int()

        def _get_keep_indices(event_idx: Tensor, ids: Tensor) -> Tensor:
            event_buckets = buckets[event_idx]

            if ids.ndim == 1:
                id_key = ids
            else:
                # Radix-style encoding of edge [src, dst].
                # Collision-free assuming ids >= 0 and no overflow
                src, dst = ids[:, 0], ids[:, 1]
                base = ids.max().item() + 1
                id_key = src * base + dst

            # Radix-style encoding of [bucket, flat_id]
            base = id_key.max().item() + 1
            final_key = event_buckets * base + id_key

            # Stable sort to get adjacent duplicates while preserving original event order
            sort_key, sort_idx = torch.sort(final_key, stable=True)

            # Mark first occurrence in each [bucket, flat_id] group
            is_first = torch.ones_like(sort_key, dtype=torch.bool)
            is_first[1:] = sort_key[1:] != sort_key[:-1]

            # Extract the first [bucket, flat_id] based on our is_first mask
            # Re-sort the final indices so that they index our global timeline chronologically
            keep = sort_idx[is_first]
            keep = keep.sort().values
            return keep

        # Edge events
        edge_mask = _get_keep_indices(self.edge_event_idx, self.edge_index)
        edge_timestamps = buckets[self.edge_event_idx][edge_mask]
        edge_index = self.edge_index[edge_mask]
        edge_feats = None
        if self.edge_feats is not None:
            edge_feats = self.edge_feats[edge_mask]

        edge_type = None
        if self.edge_type is not None:
            edge_type = self.edge_type[edge_mask]

        # Node events
        node_timestamps, node_ids, dynamic_node_feats = None, None, None
        if self.node_event_idx is not None:
            node_mask = _get_keep_indices(
                self.node_event_idx,
                self.node_ids,  # type: ignore
            )
            node_timestamps = buckets[self.node_event_idx][node_mask]
            node_ids = self.node_ids[node_mask]  # type: ignore
            dynamic_node_feats = None
            if self.dynamic_node_feats is not None:
                dynamic_node_feats = self.dynamic_node_feats[node_mask]

        # Need a deep copy
        static_node_feats = None
        if self.static_node_feats is not None:
            logger.debug('Deep copying static_node_features for coarser DGData')
            static_node_feats = self.static_node_feats.clone()

        node_type = None
        if self.node_type is not None:  # Need a deep
            logger.debug('Deep copying node_type for coarser DGData')
            node_type = self.node_type.clone()

        return DGData.from_raw(
            time_delta=time_delta,  # new discretized time_delta
            edge_timestamps=edge_timestamps,
            edge_index=edge_index,
            edge_feats=edge_feats,
            node_timestamps=node_timestamps,
            node_ids=node_ids,
            dynamic_node_feats=dynamic_node_feats,
            static_node_feats=static_node_feats,
            edge_type=edge_type,
            node_type=node_type,
        )

    def clone(self) -> DGData:
        """Deep copy all tensor and non-tensor fields to create a new DGData object."""
        cloned_fields = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                cloned_fields[f.name] = val.clone()
            else:
                cloned_fields[f.name] = copy.deepcopy(val)

        return replace(self, **cloned_fields)  # type: ignore

    @property
    def num_nodes(self) -> int:
        """Global number of nodes in the dataset.

        Note: Assumes node ids span a contiguous range, returning max(node_id) + 1.
        """
        max_id = int(self.edge_index.max())

        if self.node_ids is not None:
            max_id = max(max_id, int(self.node_ids.max()))

        return max_id + 1

    @classmethod
    def from_raw(
        cls,
        edge_timestamps: Tensor,
        edge_index: Tensor,
        edge_feats: Tensor | None = None,
        node_timestamps: Tensor | None = None,
        node_ids: Tensor | None = None,
        dynamic_node_feats: Tensor | None = None,
        static_node_feats: Tensor | None = None,
        time_delta: TimeDeltaDG | str = 'r',
        edge_type: Tensor | None = None,
        node_type: Tensor | None = None,
    ) -> DGData:
        """Construct a DGData from raw tensors for edges, nodes, and features.

        Automatically combines edge and node timestamps, computes event indices,
        and validates tensor shapes.

        Raises:
            InvalidNodeIDError: If an edge or node ID match `PADDED_NODE_ID`.
            ValueError: If any data attributes have non-well defined tensor shapes.
            EmptyGraphError: If attempting to initialize an empty graph.
        """
        _log_tensor_args(
            edge_timestamps=edge_timestamps,
            edge_index=edge_index,
            edge_feats=edge_feats,
            node_timestamps=node_timestamps,
            node_ids=node_ids,
            dynamic_node_feats=dynamic_node_feats,
            static_node_feats=static_node_feats,
            time_delta=time_delta,
            edge_type=edge_type,
            node_type=node_type,
        )
        # Build unified event timeline
        timestamps = edge_timestamps
        event_types = torch.zeros_like(edge_timestamps)
        if node_timestamps is not None:
            timestamps = torch.cat([timestamps, node_timestamps])
            event_types = torch.cat([event_types, torch.ones_like(node_timestamps)])

        # Compute event masks
        edge_event_idx = (event_types == 0).nonzero(as_tuple=True)[0]
        node_event_idx = (
            (event_types == 1).nonzero(as_tuple=True)[0]
            if node_timestamps is not None
            else None
        )

        return cls(
            time_delta=time_delta,
            timestamps=timestamps,
            edge_event_idx=edge_event_idx,
            edge_index=edge_index,
            edge_feats=edge_feats,
            node_event_idx=node_event_idx,
            node_ids=node_ids,
            dynamic_node_feats=dynamic_node_feats,
            static_node_feats=static_node_feats,
            edge_type=edge_type,
            node_type=node_type,
        )

    @classmethod
    def from_csv(
        cls,
        edge_file_path: str | pathlib.Path,
        edge_src_col: str,
        edge_dst_col: str,
        edge_time_col: str,
        edge_feats_col: List[str] | None = None,
        node_file_path: str | pathlib.Path | None = None,
        node_id_col: str | None = None,
        node_time_col: str | None = None,
        dynamic_node_feats_col: List[str] | None = None,
        static_node_feats_file_path: str | pathlib.Path | None = None,
        static_node_feats_col: List[str] | None = None,
        time_delta: TimeDeltaDG | str = 'r',
        edge_type_col: str | None = None,
        node_type_col: str | None = None,
    ) -> DGData:
        """Construct a DGData from CSV files containing edge and optional node events.

        Args:
            edge_file_path: Path to CSV file containing edges.
            edge_src_col: Column name for src nodes.
            edge_dst_col: Column name for dst nodes.
            edge_time_col: Column name for edge times.
            edge_feats_col: Optional edge feature columns.
            node_file_path: Optional CSV file for dynamic node features.
            node_id_col: Column name for dynamic node event ids. Required if node_file_path is specified.
            node_time_col: Column name for dynamic node event times. Required if node_file_path is specified.
            dynamic_node_feats_col: Optional dynamic node feature columns.
            static_node_feats_file_path: Optional CSV file for static node features.
            static_node_feats_col: Required if static_node_feats_file_path is specified.
            time_delta: Time granularity.
            edge_type_col: Column name for edge types.
            node_type_col: Column name for node types.

        Raises:
            InvalidNodeIDError: If an edge or node ID match `PADDED_NODE_ID`.
            ValueError: If any data attributes have non-well defined tensor shapes.
            EmptyGraphError: If attempting to initialize an empty graph.
        """

        def _read_csv(fp: str | pathlib.Path) -> List[dict]:
            # Assumes the whole things fits in memory
            fp = str(fp) if isinstance(fp, pathlib.Path) else fp
            with open(fp, newline='') as f:
                return list(csv.DictReader(f))

        # Read in edge data
        logger.debug('Reading edge_file_path: %s', edge_file_path)
        edge_reader = _read_csv(edge_file_path)
        num_edges = len(edge_reader)

        edge_index = torch.empty((num_edges, 2), dtype=torch.int32)
        timestamps = torch.empty(num_edges, dtype=torch.int64)
        edge_feats = None
        if edge_feats_col is not None:
            edge_feats = torch.empty((num_edges, len(edge_feats_col)))
        edge_type = None
        if edge_type_col is not None:
            edge_type = torch.empty(num_edges, dtype=torch.int32)

        for i, row in enumerate(edge_reader):
            edge_index[i, 0] = int(row[edge_src_col])
            edge_index[i, 1] = int(row[edge_dst_col])
            timestamps[i] = int(row[edge_time_col])
            if edge_feats_col is not None:
                for j, col in enumerate(edge_feats_col):
                    edge_feats[i, j] = float(row[col])  # type: ignore

            if edge_type_col is not None:
                edge_type[i] = int(row[edge_type_col])  # type: ignore

        # Read in dynamic node data
        node_timestamps, node_ids, dynamic_node_feats = None, None, None
        if node_file_path is not None:
            if node_id_col is None or node_time_col is None:
                raise ValueError(
                    'specified node_file_path without specifying node_id_col and node_time_col'
                )
            logger.debug('Reading node_file_path: %s', node_file_path)
            node_reader = _read_csv(node_file_path)
            num_node_events = len(node_reader)

            node_timestamps = torch.empty(num_node_events, dtype=torch.int64)
            node_ids = torch.empty(num_node_events, dtype=torch.int32)
            if dynamic_node_feats_col is not None:
                dynamic_node_feats = torch.empty(
                    (num_node_events, len(dynamic_node_feats_col))
                )

            for i, row in enumerate(node_reader):
                node_timestamps[i] = int(row[node_time_col])
                node_ids[i] = int(row[node_id_col])
                if dynamic_node_feats_col is not None:
                    for j, col in enumerate(dynamic_node_feats_col):
                        dynamic_node_feats[i, j] = float(row[col])  # type: ignore

        # Read in static node data
        static_node_feats = None
        node_type = None
        if static_node_feats_file_path is not None:
            if static_node_feats_col is None and node_type_col is None:
                raise ValueError(
                    'specified static_node_feats_file_path without specifying static_node_feats_col and node_type_col'
                )
            logger.debug('Reading static_node_file_path: %s', node_file_path)
            static_node_feats_reader = _read_csv(static_node_feats_file_path)
            num_nodes = len(static_node_feats_reader)
            if static_node_feats_col is not None:
                static_node_feats = torch.empty((num_nodes, len(static_node_feats_col)))
            if node_type_col is not None:
                node_type = torch.empty(num_nodes, dtype=torch.int32)
            for i, row in enumerate(static_node_feats_reader):
                if static_node_feats_col is not None:
                    for j, col in enumerate(static_node_feats_col):
                        static_node_feats[i, j] = float(row[col])  # type: ignore

                if node_type_col is not None:
                    node_type[i] = int(row[node_type_col])  # type: ignore

        return cls.from_raw(
            time_delta=time_delta,
            edge_timestamps=timestamps,
            edge_index=edge_index,
            edge_feats=edge_feats,
            node_timestamps=node_timestamps,
            node_ids=node_ids,
            dynamic_node_feats=dynamic_node_feats,
            static_node_feats=static_node_feats,
            node_type=node_type,
            edge_type=edge_type,
        )

    @classmethod
    def from_pandas(
        cls,
        edge_df: 'pandas.DataFrame',  # type: ignore
        edge_src_col: str,
        edge_dst_col: str,
        edge_time_col: str,
        edge_feats_col: List[str] | None = None,
        node_df: 'pandas.DataFrame' | None = None,  # type: ignore
        node_id_col: str | None = None,
        node_time_col: str | None = None,
        dynamic_node_feats_col: List[str] | None = None,
        static_node_feats_df: 'pandas.DataFrame' | None = None,  # type: ignore
        static_node_feats_col: List[str] | None = None,
        time_delta: TimeDeltaDG | str = 'r',
        edge_type_col: str | None = None,
        node_type_col: str | None = None,
    ) -> DGData:
        """Construct a DGData from Pandas DataFrames.

        Args:
            edge_df: DataFrame of edges.
            edge_src_col: Column name for src nodes.
            edge_dst_col: Column name for dst nodes.
            edge_time_col: Column name for edge times.
            edge_feats_col: Optional edge feature columns.
            node_df: Optional DataFrame of dynamic node events.
            node_id_col: Column name for dynamic node event ids. Required if node_file_path is specified.
            node_time_col: Column name for dynamic node event times. Required if node_file_path is specified.
            dynamic_node_feats_col: Optional node feature columns.
            static_node_feats_df: Optional static node features DataFrame.
            static_node_feats_col: Required if static_node_feats_df is specified.
            time_delta: Time granularity.
            edge_type_col: Column name for edge types.
            node_type_col: Column name for node types.

        Raises:
            InvalidNodeIDError: If an edge or node ID match `PADDED_NODE_ID`.
            ValueError: If any data attributes have non-well defined tensor shapes.
            ImportError: If the Pandas package is not resolved in the current python environment.
        """

        def _check_pandas_import() -> None:
            try:
                pass
            except ImportError:
                raise ImportError(
                    'User requires pandas to initialize a DGraph from a pd.DataFrame'
                )

        _check_pandas_import()

        # Read in edge data
        edge_index = torch.from_numpy(edge_df[[edge_src_col, edge_dst_col]].to_numpy())
        edge_timestamps = torch.from_numpy(edge_df[edge_time_col].to_numpy())
        edge_feats = None
        if edge_feats_col is not None:
            edge_feats = torch.stack(
                [torch.tensor(row) for row in edge_df[edge_feats_col]]
            )

        edge_type = None
        if edge_type_col is not None:
            edge_type = torch.from_numpy(edge_df[edge_type_col].to_numpy())

        # Read in dynamic node data
        node_timestamps, node_ids, dynamic_node_feats = None, None, None
        if node_df is not None:
            if node_id_col is None or node_time_col is None:
                raise ValueError(
                    'specified node_df without specifying node_id_col and node_time_col'
                )
            node_timestamps = torch.as_tensor(node_df[node_time_col].values)
            node_ids = torch.as_tensor(node_df[node_id_col].values)
            if dynamic_node_feats_col is not None:
                dynamic_node_feats = torch.stack(
                    [torch.tensor(row) for row in node_df[dynamic_node_feats_col]]
                )

        # Read in static node data
        static_node_feats = None
        node_type = None
        if static_node_feats_df is not None:
            if static_node_feats_col is None and node_type_col is None:
                raise ValueError(
                    'specified static_node_feats_df without specifying static_node_feats_col and node_type'
                )

            if static_node_feats_col is not None:
                static_node_feats = torch.stack(
                    [
                        torch.tensor(row)
                        for row in static_node_feats_df[static_node_feats_col]
                    ]
                )

            if node_type_col is not None:
                node_type = torch.from_numpy(
                    static_node_feats_df[node_type_col].to_numpy()
                )

        return cls.from_raw(
            time_delta=time_delta,
            edge_timestamps=edge_timestamps,
            edge_index=edge_index,
            edge_feats=edge_feats,
            node_timestamps=node_timestamps,
            node_ids=node_ids,
            dynamic_node_feats=dynamic_node_feats,
            static_node_feats=static_node_feats,
            edge_type=edge_type,
            node_type=node_type,
        )

    @classmethod
    def from_tgb(cls, name: str, **kwargs: Any) -> DGData:
        """Load a DGData from a TGB dataset.

        Args:
            name (str): Name of the TGB dataset, e.g., 'tgbl-xxxx' or 'tgbn-xxxx'.
            kwargs: Additional dataset-specific arguments.

        Returns:
            DGData: Dataset with `_split_strategy` automatically set to `TGBSplit`.

        Raises:
            ImportError: If the TGB package is not resolved in the current python environment.

        Notes:
            - TGBLinkPrediction (`tgbl-`) and TGBNodePrediction (`tgbn-`) are supported.
            - The split strategy of a TGB dataset cannot be modified.
        """
        logger.debug('Loading DGData from TGB dataset: %s', name)
        try:
            from tgb.linkproppred.dataset import LinkPropPredDataset
            from tgb.nodeproppred.dataset import NodePropPredDataset
        except ImportError:
            raise ImportError('TGB required to load TGB data, try `pip install py-tgb`')

        if name.startswith('tgbl-'):
            dataset = LinkPropPredDataset(name=name, **kwargs)
        elif name.startswith('tgbn-'):
            dataset = NodePropPredDataset(name=name, **kwargs)
        elif name.startswith('tkgl-'):
            dataset = LinkPropPredDataset(name=name, **kwargs)
        elif name.startswith('thgl-'):
            dataset = LinkPropPredDataset(name=name, **kwargs)
        else:
            raise ValueError(f'Unknown dataset: {name}')

        data = dataset.full_data

        # IDs are downcast to int32, and features to float32 preventing any runtime warnings.
        # This is safe for all TGB datasets.
        edge_index = torch.stack(
            [
                torch.from_numpy(data['sources']).to(torch.int32),
                torch.from_numpy(data['destinations']).to(torch.int32),
            ],
            dim=1,
        )
        timestamps = torch.from_numpy(data['timestamps']).to(torch.int64)
        data_edge_feat = (
            data['w'][..., None] if name.startswith('tkgl-') else data['edge_feat']
        )  # @TODO: Need to clarify this
        if data_edge_feat is None:
            edge_feats = None
        else:
            edge_feats = torch.from_numpy(data_edge_feat).to(torch.float32)

        node_timestamps, node_ids, dynamic_node_feats = None, None, None
        if name.startswith('tgbn-'):
            if 'node_label_dict' in data:
                # in TGB, after passing a batch of edges, you find the nearest node event batch in the past
                # in tgbn-trade, validation edge starts at 2010 while the first node event batch starts at 2009.
                # therefore we do (timestamps[0] - 1) to account for this behaviour
                node_label_dict = {
                    t: v
                    for t, v in data['node_label_dict'].items()
                    if (timestamps[0] - 1) <= t < timestamps[-1]
                }
            else:
                raise ValueError(
                    f'Failed to construct TGB dataset. Try updating your TGB package: `pip install --upgrade py-tgb`'
                )

            if len(node_label_dict):
                # Node events could be missing from the current data split (e.g. validation)
                num_node_events, node_label_dim = 0, 0
                for t in node_label_dict:
                    for node_id, label in node_label_dict[t].items():
                        num_node_events += 1
                        node_label_dim = label.shape[0]
                temp_node_timestamps = np.zeros(num_node_events, dtype=np.int64)
                temp_node_ids = np.zeros(num_node_events, dtype=np.int32)
                temp_dynamic_node_feats = np.zeros(
                    (num_node_events, node_label_dim), dtype=np.float32
                )
                idx = 0
                for t in node_label_dict:
                    for node_id, label in node_label_dict[t].items():
                        temp_node_timestamps[idx] = t
                        temp_node_ids[idx] = node_id
                        temp_dynamic_node_feats[idx] = label
                        idx += 1
                node_timestamps = torch.from_numpy(temp_node_timestamps)
                node_ids = torch.from_numpy(temp_node_ids)
                dynamic_node_feats = torch.from_numpy(temp_dynamic_node_feats)

        # Read static node features if they exist
        static_node_feats = None
        if dataset.node_feat is not None:
            static_node_feats = torch.from_numpy(dataset.node_feat).to(torch.float32)

        edge_type = None
        node_type = None
        if name.startswith('thgl'):
            if 'edge_type' not in data or not hasattr(dataset, 'node_type'):
                raise ValueError(
                    f'Failed to construct TGB dataset. Try updating your TGB package: `pip install --upgrade py-tgb`'
                )
            edge_type = torch.from_numpy(data['edge_type']).to(torch.int32)
            node_type = torch.from_numpy(dataset.node_type).to(torch.int32)

        if name.startswith('tkgl'):
            if 'edge_type' not in data:
                raise ValueError(
                    f'Failed to construct TGB dataset. Try updating your TGB package: `pip install --upgrade py-tgb`'
                )
            edge_type = torch.from_numpy(data['edge_type']).to(torch.int32)

        raw_times = torch.from_numpy(data['timestamps'])
        split_bounds = {}
        for split_name, mask in {
            'train': dataset.train_mask,
            'val': dataset.val_mask,
            'test': dataset.test_mask,
        }.items():
            times = raw_times[mask]
            split_bounds[split_name] = (int(times.min()), int(times.max()))

        data = cls.from_raw(
            time_delta=TGB_TIME_DELTAS[name],
            edge_timestamps=timestamps,
            edge_index=edge_index,
            edge_feats=edge_feats,
            node_timestamps=node_timestamps,
            node_ids=node_ids,
            dynamic_node_feats=dynamic_node_feats,
            static_node_feats=static_node_feats,
            edge_type=edge_type,
            node_type=node_type,
        )

        data._split_strategy = TGBSplit(split_bounds)
        logger.debug('Finished loading DGData from TGB dataset: %s', name)
        return data

    @classmethod
    def from_tgb_seq(cls, name: str, **kwargs: Any) -> DGData:
        """Load a DGData from a TGB-SEQ dataset.

        Args:
            name (str): Name of the TGB-SEQ dataset.
            kwargs: Additional dataset-specific arguments.

        Returns:
            DGData: Dataset with `_split_strategy` automatically set to `TGBSplit`.

        Raises:
            ImportError: If the TGB-SEQ package is not resolved in the current python environment.

        Notes:
            - The split strategy of a TGB-SEQ dataset cannot be modified.
            - If a data root location is not specified, we default store location to './data'.
        """
        logger.debug('Loading DGData from TGB-SEQ dataset: %s', name)
        try:
            from tgb_seq.LinkPred.dataloader import TGBSeqLoader
        except ImportError:
            raise ImportError(
                'TGB-SEQ required to load TGB data, try `pip install tgb-seq`'
            )

        if 'root' not in kwargs:
            kwargs['root'] = './data'
        data = TGBSeqLoader(name=name, **kwargs)

        # IDs are downcast to int32, and features to float32 preventing any runtime warnings.
        # This is safe for all TGB datasets.
        edge_index = torch.stack(
            [
                torch.from_numpy(data.src_node_ids).to(torch.int32),
                torch.from_numpy(data.dst_node_ids).to(torch.int32),
            ],
            dim=1,
        )

        timestamps = torch.from_numpy(data.node_interact_times).to(torch.int64)
        edge_feats = None
        if data.edge_features is not None:
            edge_feats = torch.from_numpy(data.edge_features).to(torch.float32)

        # Read static node features if they exist
        static_node_feats = None
        if data.node_features is not None:
            static_node_feats = torch.from_numpy(data.node_features).to(torch.float32)

        split_bounds = {}
        for split_name, mask in {
            'train': data.train_mask,
            'val': data.val_mask,
            'test': data.test_mask,
        }.items():
            times = data.node_interact_times[mask]
            split_bounds[split_name] = (int(times.min()), int(times.max()))

        data = cls.from_raw(
            time_delta=TGB_SEQ_TIME_DELTAS[name],
            edge_timestamps=timestamps,
            edge_index=edge_index,
            edge_feats=edge_feats,
            static_node_feats=static_node_feats,
        )

        data._split_strategy = TGBSplit(split_bounds)
        logger.debug('Finished loading DGData from TGB-SEQ dataset: %s', name)
        return data


def _log_tensor_args(**kwargs: Any) -> None:
    for name, value in kwargs.items():
        if value is None:
            logger.debug('%s: None', name)
        elif isinstance(value, Tensor):
            logger.debug(
                '%s: Tensor, shape: %s, dtype: %s',
                name,
                tuple(value.shape),
                value.dtype,
            )
        else:
            logger.debug('%s: %s = %s', name, type(value).__name__, value)
