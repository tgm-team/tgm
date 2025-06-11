from __future__ import annotations

import csv
import pathlib
import warnings
from dataclasses import dataclass
from typing import Any, List

import numpy as np
import torch
from torch import Tensor


@dataclass
class DGData:
    r"""Bundles dynamic graph data to be forwarded to DGStorage."""

    timestamps: Tensor  # [num_events]

    edge_event_idx: Tensor  # [num_edge_events]
    edge_index: Tensor  # [num_edge_events, 2]
    edge_feats: Tensor | None = None  # [num_edge_events, D_edge]

    node_event_idx: Tensor | None = None  # [num_node_events]
    node_ids: Tensor | None = None  # [num_node_events]
    dynamic_node_feats: Tensor | None = None  # [num_node_events, D_node_dynamic]

    static_node_feats: Tensor | None = None  # [num_nodes, D_node_static]

    def __post_init__(self) -> None:
        # Validate edge index
        if not isinstance(self.edge_index, Tensor):
            raise TypeError('edge_index must be a Tensor')
        if self.edge_index.ndim != 2 or self.edge_index.shape[1] != 2:
            raise ValueError('edge_index must have shape [num_edges, 2]')

        num_edges = self.edge_index.shape[0]
        if num_edges == 0:
            raise ValueError('empty graphs not supported')

        # Validate edge event idx
        if not isinstance(self.edge_event_idx, Tensor):
            raise TypeError('edge_event_idx must be a Tensor')
        if self.edge_event_idx.ndim != 1 or self.edge_event_idx.shape[0] != num_edges:
            raise ValueError('edge_event_idx must have shape [num_edges]')

        # Validate edge features
        if self.edge_feats is not None:
            if not isinstance(self.edge_feats, Tensor):
                raise TypeError('edge_feats must be a Tensor')
            if self.edge_feats.ndim != 2 or self.edge_feats.shape[0] != num_edges:
                raise ValueError('edge_feats must have shape [num_edges, D_edge]')

        # Validate node event idx
        num_node_events = 0
        if self.node_event_idx is not None:
            if not isinstance(self.node_event_idx, Tensor):
                raise TypeError('node_event_idx must be a Tensor')
            if self.node_event_idx.ndim != 1:
                raise ValueError('node_event_idx must have shape [num_node_events]')
            num_node_events = self.node_event_idx.shape[0]

            # Validate node ids
            if not isinstance(self.node_ids, Tensor):
                raise TypeError('node_ids must be a Tensor')
            if self.node_ids.ndim != 1 or self.node_ids.shape[0] != num_node_events:
                raise ValueError('node_ids must have shape [num_node_events]')

            # Validate dynamic node features (could be None)
            if self.dynamic_node_feats is not None:
                if not isinstance(self.dynamic_node_feats, Tensor):
                    raise TypeError('dynamic_node_feats must be a Tensor')
                if (
                    self.dynamic_node_feats.ndim != 2
                    or self.dynamic_node_feats.shape[0] != num_node_events
                ):
                    raise ValueError(
                        'dynamic_node_feats must have shape [num_node_events, D_node_dynamic]'
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
            if not isinstance(self.static_node_feats, Tensor):
                raise TypeError('static_node_feats must be a Tensor')
            if (
                self.static_node_feats.ndim != 2
                or self.static_node_feats.shape[0] != num_nodes
            ):
                raise ValueError(
                    'static_node_feats must have shape [num_nodes, D_node_static]'
                )

        # Validate timestamps
        if not isinstance(self.timestamps, Tensor):
            raise TypeError('timestamps must be a Tensor')
        if (
            self.timestamps.ndim != 1
            or self.timestamps.shape[0] != num_edges + num_node_events
        ):
            raise ValueError('timestamps must have shape [num_edges + num_node_events]')
        if not torch.all(self.timestamps >= 0):
            raise ValueError('timestamps must be non-negative integers')

        # Sort if necessary
        if not torch.all(torch.diff(self.timestamps) >= 0):
            warnings.warn('received non-chronological events, sorting by time')

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
    ) -> DGData:
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
    ) -> DGData:
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
    ) -> DGData:
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
            edge_timestamps=edge_timestamps,
            edge_index=edge_index,
            edge_feats=edge_feats,
            node_timestamps=node_timestamps,
            node_ids=node_ids,
            dynamic_node_feats=dynamic_node_feats,
            static_node_feats=static_node_feats,
        )

    @classmethod
    def from_tgb(cls, name: str, split: str = 'all', **kwargs: Any) -> DGData:
        def _check_tgb_import() -> tuple['LinkPropPredDataset', 'NodePropPredDataset']:  # type: ignore
            try:
                from tgb.linkproppred.dataset import LinkPropPredDataset
                from tgb.nodeproppred.dataset import NodePropPredDataset

                return LinkPropPredDataset, NodePropPredDataset
            except ImportError:
                err_msg = 'User requires tgb to initialize a DGraph from a tgb dataset '
                raise ImportError(err_msg)

        LinkPropPredDataset, NodePropPredDataset = _check_tgb_import()

        if name.startswith('tgbl-'):
            dataset = LinkPropPredDataset(name=name, **kwargs)  # type: ignore
        elif name.startswith('tgbn-'):
            dataset = NodePropPredDataset(name=name, **kwargs)  # type: ignore
        else:
            raise ValueError(f'Unknown dataset: {name}')
        data = dataset.full_data

        split_masks = {
            'train': dataset.train_mask,
            'valid': dataset.val_mask,
            'test': dataset.test_mask,
        }

        if split == 'all':
            mask = slice(None)  # selects everything
        elif split in split_masks:
            mask = split_masks[split]
        else:
            raise ValueError(f'Unknown split: {split}')

        src = data['sources'][mask]
        dst = data['destinations'][mask]
        edge_index = torch.from_numpy(np.stack([src, dst], axis=1)).long()
        timestamps = torch.from_numpy(data['timestamps'][mask]).long()
        if data['edge_feat'] is None:
            edge_feats = None
        else:
            edge_feats = torch.from_numpy(data['edge_feat'][mask])

        node_timestamps, node_ids, dynamic_node_feats = None, None, None
        if name.startswith('tgbn-'):
            if 'node_label_dict' in data:
                node_label_dict = data['node_label_dict']
            else:
                raise ValueError('please update your tgb package or install by source')

            num_node_events = 0
            node_label_dim = 0
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

        return cls.from_raw(
            edge_timestamps=timestamps,
            edge_index=edge_index,
            edge_feats=edge_feats,
            node_timestamps=node_timestamps,
            node_ids=node_ids,
            dynamic_node_feats=dynamic_node_feats,
            static_node_feats=static_node_feats,
        )

    @classmethod
    def from_any(
        cls,
        data: str | pathlib.Path | 'pandas.DataFrame',  # type: ignore
        **kwargs: Any,
    ) -> DGData:
        if isinstance(data, (str, pathlib.Path)):
            data_str = str(data)
            if data_str.startswith('tgbl-') or data_str.startswith('tgbn-'):
                return cls.from_tgb(name=data_str, **kwargs)
            if data_str.endswith('.csv'):
                return cls.from_csv(data, **kwargs)
            raise ValueError(
                f'Unsupported file format or dataset identifier: {data_str}'
            )
        else:
            return cls.from_pandas(data, **kwargs)
