from __future__ import annotations

import pathlib
from collections.abc import Iterable, Sized
from dataclasses import dataclass
from functools import cached_property
from typing import Any, List, Optional, Set, Tuple

import torch
from torch import Tensor

from tgm._storage import DGSliceTracker, DGStorage
from tgm.data import DGData
from tgm.timedelta import TimeDeltaDG


class DGraph:
    r"""Dynamic Graph Object provides a 'view' over a DGStorage backend."""

    def __init__(
        self,
        data: DGStorage | str,
        discretize_time_delta: TimeDeltaDG | str | None = None,
        reduce_op: str | None = None,
        device: str | torch.device = 'cpu',
        **kwargs: Any,
    ) -> None:
        if isinstance(discretize_time_delta, str):
            discretize_time_delta = TimeDeltaDG(discretize_time_delta)

        if isinstance(data, str):
            dg_data, time_delta = DGData.from_known_dataset(
                data, discretize_time_delta, reduce_op, **kwargs
            )
            self._storage = DGStorage(dg_data)
            self._time_delta = time_delta
        elif isinstance(data, DGStorage):
            if discretize_time_delta is None:
                raise RuntimeError(
                    'Internally tried creating DGraph on existing storage without specifying a time_delta'
                )
            self._storage = data
            self._time_delta = discretize_time_delta
        else:
            raise ValueError(f'Unsupported dataset type: {type(data)}')

        self._device = torch.device(device)
        self._slice = DGSliceTracker()

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
        native_time_delta: TimeDeltaDG | str = 'r',
        discretize_time_delta: TimeDeltaDG | str | None = None,
        reduce_op: str | None = None,
        device: str | torch.device = 'cpu',
    ) -> DGraph:
        r"""Constructs a DGraph from CSV files.

        Args:
            edge_file_path (str | pathlib.Path): Path to the edge CSV file.
            edge_src_col (str): Column name for source nodes in the edge file.
            edge_dst_col (str): Column name for destination nodes in the edge file.
            edge_time_col (str): Column name for edge timestamps in the edge file.
            edge_feats_col (List[str] | None): List of column names for edge features in the edge file. Defaults to None.
            node_file_path (str | pathlib.Path | None): Path to the node CSV file.
            node_id_col (str | None): Column name for node ids in the node file.
            node_time_col (str | None): Column name for node timestamps in the node file.
            dynamic_node_feats_col (List[str] | None): List of column names for dynamic node features in the node file. Defaults to None.
            static_node_feats_file_path (str | pathlib.Path | None): Path to the static node features CSV file.
            static_node_feats_col (List[str] | None): List of column names for static node features in the static node features file.
            native_time_delta (TimeDeltaDG | str): Native Time Delta for the graph. Defaults to 'r' for ordered edgelist.
            discretize_time_delta (TimeDeltaDG | str | None): If specified, determines the time to discretize the underlying data to.
            reduce_op (str | None): If discretize_time_delta is specified, this determines what reduction to apply to the underlying data.
            device (str | torch.device): Device to store the graph on. Defaults to 'cpu'.

        Note:
            - If `discretize_time_delta` is specified:
                1. `native_time_delta` must be an absolute time delta, **not** `'r'`.
                2. The native time delta must be **finer** than `discretize_time_delta`.
                3. A valid `reduce_op` is required.
        """
        data = DGData.from_csv(
            edge_file_path=edge_file_path,
            edge_src_col=edge_src_col,
            edge_dst_col=edge_dst_col,
            edge_time_col=edge_time_col,
            edge_feats_col=edge_feats_col,
            node_file_path=node_file_path,
            node_id_col=node_id_col,
            node_time_col=node_time_col,
            dynamic_node_feats_col=dynamic_node_feats_col,
            static_node_feats_file_path=static_node_feats_file_path,
            static_node_feats_col=static_node_feats_col,
        )
        data = data.discretize(native_time_delta, discretize_time_delta, reduce_op)
        storage = DGStorage(data)
        return cls(storage, discretize_time_delta, device=device)

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
        native_time_delta: TimeDeltaDG | str = 'r',
        discretize_time_delta: TimeDeltaDG | str | None = None,
        reduce_op: str | None = None,
        device: str | torch.device = 'cpu',
    ) -> DGraph:
        r"""Constructs DGraph from raw tensors.

        Args:
            edge_timestamps (Tensor): Timestamps of the edges.
            edge_index (Tensor): Edge index tensor of shape (2, E) where E is the number of edges.
            edge_feats (Tensor | None): Edge features tensor of shape (E, d_edge) or None if no edge features.
            node_timestamps (Tensor | None): Node timestamps tensor of shape (N, T) or None if no node timestamps.
            node_ids (Tensor | None): Node ids tensor of shape (N,) or None if no node ids.
            dynamic_node_feats (Tensor | None): Dynamic node features tensor of shape (T, N, d_node_dynamic) or None if no dynamic node features.
            static_node_feats (Tensor | None): Static node features tensor of shape (N, d_node_static) or None if no static node features.
            native_time_delta (TimeDeltaDG | str): Native Time Delta for the graph. Defaults to 'r' for ordered edgelist.
            discretize_time_delta (TimeDeltaDG | str | None): If specified, determines the time to discretize the underlying data to.
            reduce_op (str | None): If discretize_time_delta is specified, this determines what reduction to apply to the underlying data.
            device (str | torch.device): Device to store the graph on. Defaults to 'cpu'.
        """
        data = DGData.from_raw(
            edge_timestamps=edge_timestamps,
            edge_index=edge_index,
            edge_feats=edge_feats,
            node_timestamps=node_timestamps,
            node_ids=node_ids,
            dynamic_node_feats=dynamic_node_feats,
            static_node_feats=static_node_feats,
        )
        data = data.discretize(native_time_delta, discretize_time_delta, reduce_op)
        storage = DGStorage(data)
        return cls(storage, discretize_time_delta, device=device)

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
        native_time_delta: TimeDeltaDG | str = 'r',
        discretize_time_delta: TimeDeltaDG | str | None = None,
        reduce_op: str | None = None,
        device: str | torch.device = 'cpu',
    ) -> DGraph:
        r"""Constructs a DGraph from pandas DataFrames.

        Args:
            edge_df (pandas.DataFrame): DataFrame containing edges with columns for source, destination, and time.
            edge_src_col (str): Column name for source nodes in the edge DataFrame.
            edge_dst_col (str): Column name for destination nodes in the edge DataFrame.
            edge_time_col (str): Column name for edge timestamps in the edge DataFrame.
            edge_feats_col (List[str] | None): List of column names for edge features in the edge DataFrame. Defaults to None.
            node_df (pandas.DataFrame | None): DataFrame containing nodes with columns for node ids and timestamps.
            node_id_col (str | None): Column name for node ids in the node DataFrame.
            node_time_col (str | None): Column name for node timestamps in the node DataFrame.
            dynamic_node_feats_col (List[str] | None): List of column names for dynamic node features in the node DataFrame. Defaults to None.
            static_node_feats_df (pandas.DataFrame | None): DataFrame containing static node features.
            static_node_feats_col (List[str] | None): List of column names for static node features in the static node features DataFrame.
            native_time_delta (TimeDeltaDG | str): Native Time Delta for the graph. Defaults to 'r' for ordered edgelist.
            discretize_time_delta (TimeDeltaDG | str | None): If specified, determines the time to discretize the underlying data to.
            reduce_op (str | None): If discretize_time_delta is specified, this determines what reduction to apply to the underlying data.
            device (str | torch.device): Device to store the graph on. Defaults to 'cpu'.
        """
        data = DGData.from_pandas(
            edge_df=edge_df,
            edge_src_col=edge_src_col,
            edge_dst_col=edge_dst_col,
            edge_time_col=edge_time_col,
            edge_feats_col=edge_feats_col,
            node_df=node_df,
            node_id_col=node_id_col,
            node_time_col=node_time_col,
            dynamic_node_feats_col=dynamic_node_feats_col,
            static_node_feats_df=static_node_feats_df,
            static_node_feats_col=static_node_feats_col,
        )
        data = data.discretize(native_time_delta, discretize_time_delta, reduce_op)
        storage = DGStorage(data)
        return cls(storage, discretize_time_delta, device=device)

    @classmethod
    def from_tgb(
        cls,
        data_name: str,
        split: str = 'all',
        discretize_time_delta: TimeDeltaDG | str | None = None,
        reduce_op: str | None = None,
        device: str | torch.device = 'cpu',
    ) -> DGraph:
        r"""Constructs a DGraph from a TGB dataset.

        Args:
            data_name (str): Name of the TGB dataset.
            split (str): Split of the dataset to use. Defaults to 'all'.
            discretize_time_delta (TimeDeltaDG | str | None): If specified, determines the time to discretize the underlying data to.
            reduce_op (str | None): If discretize_time_delta is specified, this determines what reduction to apply to the underlying data.
            device (str | torch.device): Device to store the graph on. Defaults to 'cpu'.
        """
        data, time_delta = DGData.from_known_dataset(
            data_name=data_name,
            split=split,
            discretize_time_delta=discretize_time_delta,
            reduce_op=reduce_op,
        )
        storage = DGStorage(data)
        return cls(storage, time_delta, device=device)

    def materialize(self, materialize_features: bool = True) -> DGBatch:
        r"""Materialize dense tensors: src, dst, time, and optionally {'node': dynamic_node_feats, node_times, node_ids, 'edge': edge_features}."""
        batch = DGBatch(*self.edges)
        if materialize_features and self.dynamic_node_feats is not None:
            batch.node_times, batch.node_ids = self.dynamic_node_feats._indices()
            batch.dynamic_node_feats = self.dynamic_node_feats._values()
        if materialize_features and self.edge_feats is not None:
            batch.edge_feats = self.edge_feats._values()
        return batch

    def slice_events(
        self, start_idx: Optional[int] = None, end_idx: Optional[int] = None
    ) -> DGraph:
        r"""Create and return a new view by slicing events (end_idx exclusive)."""
        if start_idx is not None and end_idx is not None and start_idx > end_idx:
            raise ValueError(f'start_idx ({start_idx}) must be <= end_idx ({end_idx})')

        dg = DGraph(
            data=self._storage,
            discretize_time_delta=self.time_delta,
            device=self.device,
        )
        dg._slice.start_time = self._slice.start_time
        dg._slice.end_time = self._slice.end_time
        dg._slice.start_idx = self._maybe_max(start_idx, self._slice.start_idx)
        dg._slice.end_idx = self._maybe_min(end_idx, self._slice.end_idx)
        return dg

    def slice_time(
        self, start_time: Optional[int] = None, end_time: Optional[int] = None
    ) -> DGraph:
        r"""Create and return a new view by slicing temporally (end_time exclusive)."""
        if start_time is not None and end_time is not None and start_time > end_time:
            raise ValueError(
                f'start_time ({start_time}) must be <= end_time ({end_time})'
            )
        if end_time is not None:
            end_time -= 1

        dg = DGraph(
            data=self._storage,
            discretize_time_delta=self.time_delta,
            device=self.device,
        )
        dg._slice.start_time = self._maybe_max(start_time, self.start_time)
        dg._slice.end_time = self._maybe_min(end_time, self.end_time)
        dg._slice.start_idx = self._slice.start_idx
        dg._slice.end_idx = self._slice.end_idx
        return dg

    def __len__(self) -> int:
        r"""The number of timestamps in the dynamic graph."""
        return self.num_timestamps

    def __str__(self) -> str:
        return f'DGraph(storage={self._storage.__class__.__name__}, time_delta={self.time_delta}, device={self.device})'

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def time_delta(self) -> TimeDeltaDG:
        return self._time_delta

    def to(self, device: str | torch.device) -> DGraph:
        return DGraph(data=self._storage, time_delta=self.time_delta, device=device)

    @cached_property
    def start_time(self) -> Optional[int]:
        r"""The start time of the dynamic graph. None if the graph is empty."""
        if self._slice.start_time is None:
            self._slice.start_time = self._storage.get_start_time(self._slice)
        return self._slice.start_time

    @cached_property
    def end_time(self) -> Optional[int]:
        r"""The end time of the dynamic graph. None, if the graph is empty."""
        if self._slice.end_time is None:
            self._slice.end_time = self._storage.get_end_time(self._slice)
        return self._slice.end_time

    @cached_property
    def num_nodes(self) -> int:
        r"""The total number of unique nodes encountered over the dynamic graph."""
        nodes = self._storage.get_nodes(self._slice)
        return max(nodes) + 1 if len(nodes) else 0

    @cached_property
    def num_edges(self) -> int:
        r"""The total number of unique edges encountered over the dynamic graph."""
        src, *_ = self.edges
        return len(src)

    @cached_property
    def num_timestamps(self) -> int:
        r"""The total number of unique timestamps encountered over the dynamic graph."""
        return self._storage.get_num_timestamps(self._slice)

    @cached_property
    def num_events(self) -> int:
        r"""The total number of events encountered over the dynamic graph."""
        return self._storage.get_num_events(self._slice)

    @cached_property
    def nodes(self) -> Set[int]:
        r"""The set of node ids over the dynamic graph."""
        return self._storage.get_nodes(self._slice)

    @cached_property
    def edges(self) -> Tuple[Tensor, Tensor, Tensor]:
        r"""The src, dst, time tensors over the dynamic graph."""
        src, dst, time = self._storage.get_edges(self._slice)
        src, dst, time = src.to(self.device), dst.to(self.device), time.to(self.device)
        return src, dst, time

    @cached_property
    def static_node_feats(self) -> Optional[Tensor]:
        r"""If static node features exist, returns a dense Tensor(num_nodes x d_node_static)."""
        feats = self._storage.get_static_node_feats()
        if feats is not None:
            feats = feats.to(self.device)
        return feats

    @cached_property
    def dynamic_node_feats(self) -> Optional[Tensor]:
        r"""The aggregated dynamic node features over the dynamic graph.

        If dynamic node features exist, returns a Tensor.sparse_coo_tensor(T x V x d_node_dynamic).
        """
        feats = self._storage.get_dynamic_node_feats(self._slice)
        if feats is not None:
            feats = feats.to(self.device)
        return feats

    @cached_property
    def edge_feats(self) -> Optional[Tensor]:
        r"""The aggregated edge features over the dynamic graph.

        If edge features exist, returns a Tensor.sparse_coo_tensor(T x V x V x d_edge).
        """
        feats = self._storage.get_edge_feats(self._slice)
        if feats is not None:
            feats = feats.to(self.device)
        return feats

    @cached_property
    def static_node_feats_dim(self) -> Optional[int]:
        r"""Static Node feature dimension or None if not Node features on the Graph."""
        return self._storage.get_static_node_feats_dim()

    @cached_property
    def dynamic_node_feats_dim(self) -> Optional[int]:
        r"""Dynamic Node feature dimension or None if not Node features on the Graph."""
        return self._storage.get_dynamic_node_feats_dim()

    @cached_property
    def edge_feats_dim(self) -> Optional[int]:
        r"""Edge feature dimension or None if not Node features on the Graph."""
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


@dataclass
class DGBatch:
    src: Tensor
    dst: Tensor
    time: Tensor
    dynamic_node_feats: Optional[Tensor] = None
    edge_feats: Optional[Tensor] = None
    node_times: Optional[Tensor] = None
    node_ids: Optional[Tensor] = None

    def __str__(self) -> str:
        def _get_description(object: Any) -> str:
            description = ''
            if isinstance(object, torch.Tensor):
                description = str(list(object.shape))
            elif isinstance(object, Iterable):
                unique_type = set()
                for element in object:
                    unique_type.add(_get_description(element))
                if isinstance(object, Sized):
                    obj_len = f' x{str(len(object))}'
                else:
                    obj_len = ''

                description = (
                    type(object).__name__ + '(' + '|'.join(unique_type) + obj_len + ')'
                )
            else:
                description = type(object).__name__

            return description

        descriptions = []
        for attr, value in vars(self).items():
            descriptions.append(f'{attr} = {_get_description(value)}')
        return 'DGBatch(' + ', '.join(descriptions) + ')'
