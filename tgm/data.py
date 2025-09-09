from __future__ import annotations

import copy
import csv
import pathlib
from dataclasses import dataclass, fields, replace
from typing import Any, Callable, List, Tuple

import numpy as np
import torch
from torch import Tensor

from tgm.constants import PADDED_NODE_ID
from tgm.exceptions import (
    EmptyGraphError,
    EventOrderedConversionError,
    InvalidDiscretizationError,
    InvalidNodeIDError,
)
from tgm.split import SplitStrategy, TemporalRatioSplit, TGBSplit
from tgm.timedelta import TGB_TIME_DELTAS, TimeDeltaDG


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

    Raises:
        InvalidNodeIDError: If an edge or node ID match `PADDED_NODE_ID`.
        ValueError: If any data attributes have non-well defined tensor shapes.
        EmptyGraphError: If attempting to initialize an empty graph.

    Notes:
        - Timestamps must be non-negative and sorted; DGData will sort automatically if necessary.
        - Cloning creates a deep copy of tensors to prevent in-place modifications.
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

    _split_strategy: SplitStrategy | None = None

    def __post_init__(self) -> None:
        if isinstance(self.time_delta, str):
            self.time_delta = TimeDeltaDG(self.time_delta)

        def _assert_is_tensor(x: Any, name: str) -> None:
            if not isinstance(x, Tensor):
                raise TypeError(f'{name} must be a Tensor, got: {type(x)}')

        def _assert_tensor_is_integral(x: Tensor, name: str) -> None:
            int_types = [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]
            if x.dtype not in int_types:
                raise TypeError(f'{name} must have integer dtype but got: {x.dtype}')

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

        num_edges = self.edge_index.shape[0]
        if num_edges == 0:
            raise EmptyGraphError('empty graphs not supported')

        # Validate edge event idx
        _assert_is_tensor(self.edge_event_idx, 'edge_event_idx')
        _assert_tensor_is_integral(self.edge_event_idx, 'edge_event_idx')
        if self.edge_event_idx.ndim != 1 or self.edge_event_idx.shape[0] != num_edges:
            raise ValueError(
                'edge_event_idx must have shape [num_edges], '
                f'got {num_edges} edges and shape {self.edge_event_idx.shape}'
            )

        # Validate edge features
        if self.edge_feats is not None:
            _assert_is_tensor(self.edge_feats, 'edge_feats')
            if self.edge_feats.ndim != 2 or self.edge_feats.shape[0] != num_edges:
                raise ValueError(
                    'edge_feats must have shape [num_edges, D_edge], '
                    f'got {num_edges} edges and shape {self.edge_feats.shape}'
                )

        # Validate node event idx
        num_node_events = 0
        if self.node_event_idx is not None:
            _assert_is_tensor(self.node_event_idx, 'node_event_idx')
            _assert_tensor_is_integral(self.node_event_idx, 'node_event_idx')
            if self.node_event_idx.ndim != 1:
                raise ValueError(
                    f'node_event_idx must have shape [num_node_events], got: {self.node_event_idx.shape}'
                )

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
        else:
            if self.node_ids is not None:
                raise ValueError('must specify node_event_idx if using node_ids')
            if self.dynamic_node_feats is not None:
                raise ValueError(
                    'must specify node_event_idx if using dynamic_node_feats'
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

        # Validate timestamps
        _assert_is_tensor(self.timestamps, 'timestamps')
        _assert_tensor_is_integral(self.timestamps, 'timestamps')
        if not torch.all(self.timestamps >= 0):
            raise ValueError('timestamps must all be non-negative')
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
            # Sort timestamps
            sort_idx = torch.argsort(self.timestamps)
            inverse_sort_idx = torch.empty_like(sort_idx)
            inverse_sort_idx[sort_idx] = torch.arange(len(sort_idx))
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
        buckets = (self.timestamps.float() * time_factor).floor().long()

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

        static_node_feats = None
        if self.static_node_feats is not None:  # Need a deep copy
            static_node_feats = self.static_node_feats.clone()

        return DGData.from_raw(
            time_delta=time_delta,  # new discretized time_delta
            edge_timestamps=edge_timestamps,
            edge_index=edge_index,
            edge_feats=edge_feats,
            node_timestamps=node_timestamps,
            node_ids=node_ids,
            dynamic_node_feats=dynamic_node_feats,
            static_node_feats=static_node_feats,
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
    ) -> DGData:
        """Construct a DGData from raw tensors for edges, nodes, and features.

        Automatically combines edge and node timestamps, computes event indices,
        and validates tensor shapes.

        Raises:
            InvalidNodeIDError: If an edge or node ID match `PADDED_NODE_ID`.
            ValueError: If any data attributes have non-well defined tensor shapes.
            EmptyGraphError: If attempting to initialize an empty graph.
        """
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
        edge_reader = _read_csv(edge_file_path)
        num_edges = len(edge_reader)

        edge_index = torch.empty((num_edges, 2), dtype=torch.long)
        timestamps = torch.empty(num_edges, dtype=torch.long)
        edge_feats = None
        if edge_feats_col is not None:
            edge_feats = torch.empty((num_edges, len(edge_feats_col)))

        for i, row in enumerate(edge_reader):
            edge_index[i, 0] = int(row[edge_src_col])
            edge_index[i, 1] = int(row[edge_dst_col])
            timestamps[i] = int(row[edge_time_col])
            if edge_feats_col is not None:
                for j, col in enumerate(edge_feats_col):
                    edge_feats[i, j] = float(row[col])  # type: ignore

        # Read in dynamic node data
        node_timestamps, node_ids, dynamic_node_feats = None, None, None
        if node_file_path is not None:
            if node_id_col is None or node_time_col is None:
                raise ValueError(
                    'specified node_file_path without specifying node_id_col and node_time_col'
                )
            node_reader = _read_csv(node_file_path)
            num_node_events = len(node_reader)

            node_timestamps = torch.empty(num_node_events, dtype=torch.long)
            node_ids = torch.empty(num_node_events, dtype=torch.long)
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
        if static_node_feats_file_path is not None:
            if static_node_feats_col is None:
                raise ValueError(
                    'specified static_node_feats_file_path without specifying static_node_feats_col'
                )
            static_node_feats_reader = _read_csv(static_node_feats_file_path)
            num_nodes = len(static_node_feats_reader)
            static_node_feats = torch.empty((num_nodes, len(static_node_feats_col)))
            for i, row in enumerate(static_node_feats_reader):
                for j, col in enumerate(static_node_feats_col):
                    static_node_feats[i, j] = float(row[col])

        return cls.from_raw(
            time_delta=time_delta,
            edge_timestamps=timestamps,
            edge_index=edge_index,
            edge_feats=edge_feats,
            node_timestamps=node_timestamps,
            node_ids=node_ids,
            dynamic_node_feats=dynamic_node_feats,
            static_node_feats=static_node_feats,
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

        Raises:
            InvalidNodeIDError: If an edge or node ID match `PADDED_NODE_ID`.
            ValueError: If any data attributes have non-well defined tensor shapes.
            ImportError: If the Pandas package is not resolved in the current python environment.
        """

        def _check_pandas_import(min_version_number: str | None = None) -> None:
            try:
                import pandas

                user_pandas_version = pandas.__version__
            except ImportError:
                user_pandas_version = None

            err_msg = 'User requires pandas '
            if min_version_number is not None:
                err_msg += f'>={min_version_number} '
            err_msg += 'to initialize a DGraph a dataframe'

            if user_pandas_version is None:
                raise ImportError(err_msg)
            elif (
                min_version_number is not None
                and user_pandas_version < min_version_number
            ):
                err_msg += (
                    f', found pandas=={user_pandas_version} < {min_version_number}'
                )
                raise ImportError(err_msg)

        _check_pandas_import()

        # Read in edge data
        edge_index = torch.from_numpy(
            edge_df[[edge_src_col, edge_dst_col]].to_numpy()
        ).long()
        edge_timestamps = torch.from_numpy(edge_df[edge_time_col].to_numpy()).long()
        edge_feats = None
        if edge_feats_col is not None:
            edge_feats = torch.Tensor(edge_df[edge_feats_col].tolist())

        # Read in dynamic node data
        node_timestamps, node_ids, dynamic_node_feats = None, None, None
        if node_df is not None:
            if node_id_col is None or node_time_col is None:
                raise ValueError(
                    'specified node_df without specifying node_id_col and node_time_col'
                )
            node_timestamps = torch.from_numpy(node_df[node_time_col].to_numpy()).long()
            node_ids = torch.from_numpy(node_df[node_id_col].to_numpy()).long()
            if dynamic_node_feats_col is not None:
                dynamic_node_feats = torch.Tensor(
                    node_df[dynamic_node_feats_col].tolist()
                )

        # Read in static node data
        static_node_feats = None
        if static_node_feats_df is not None:
            if static_node_feats_col is None:
                raise ValueError(
                    'specified static_node_feats_df without specifying static_node_feats_col'
                )
            static_node_feats = torch.Tensor(
                static_node_feats_df[static_node_feats_col].tolist()
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
        try:
            from tgb.linkproppred.dataset import LinkPropPredDataset
            from tgb.nodeproppred.dataset import NodePropPredDataset
        except ImportError:
            raise ImportError('TGB required to load TGB data, try `pip install py-tgb`')

        def suppress_output(func: Callable, *args: Any, **kwargs: Any) -> Any:
            # This is a hacky workaround that tries to lower the verbosity on TGB
            # logs which are currently directed to stdout. This should be removed
            # once https://github.com/shenyangHuang/TGB/issues/117 is addressed.
            import builtins

            SILENCE_PREFIXES = [
                'raw file found',
                'Dataset directory is',
                'loading processed file',
            ]

            original_print = builtins.print

            def filtered_print(*p_args: Any, **p_kwargs: Any) -> None:
                if not p_args:
                    return
                msg = str(p_args[0])
                if any(msg.startswith(prefix) for prefix in SILENCE_PREFIXES):
                    return
                original_print(*p_args, **p_kwargs)

            try:
                builtins.print = filtered_print
                return func(*args, **kwargs)
            finally:
                builtins.print = original_print

        if name.startswith('tgbl-'):
            dataset = suppress_output(LinkPropPredDataset, name=name, **kwargs)
        elif name.startswith('tgbn-'):
            dataset = suppress_output(NodePropPredDataset, name=name, **kwargs)
        else:
            raise ValueError(f'Unknown dataset: {name}')

        data = dataset.full_data
        src, dst = data['sources'], data['destinations']
        edge_index = torch.from_numpy(np.stack([src, dst], axis=1)).long()
        timestamps = torch.from_numpy(data['timestamps']).long()
        if data['edge_feat'] is None:
            edge_feats = None
        else:
            edge_feats = torch.from_numpy(data['edge_feat'])

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
                raise ValueError('please update your tgb package or install by source')

            if len(node_label_dict):
                # Node events could be missing from the current data split (e.g. validation)
                num_node_events, node_label_dim = 0, 0
                for t in node_label_dict:
                    for node_id, label in node_label_dict[t].items():
                        num_node_events += 1
                        node_label_dim = label.shape[0]
                temp_node_timestamps = np.zeros(num_node_events, dtype=np.int64)
                temp_node_ids = np.zeros(num_node_events, dtype=np.int64)
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
                node_timestamps = torch.from_numpy(temp_node_timestamps).long()
                node_ids = torch.from_numpy(temp_node_ids).long()
                dynamic_node_feats = torch.from_numpy(temp_dynamic_node_feats).float()

        # Read static node features if they exist
        static_node_feats = None
        if dataset.node_feat is not None:
            static_node_feats = torch.from_numpy(dataset.node_feat)

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
        )

        data._split_strategy = TGBSplit(split_bounds)
        return data
