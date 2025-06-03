from __future__ import annotations

import csv
import pathlib
import warnings
from dataclasses import dataclass
from typing import Any, List

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
        file_path: str | pathlib.Path,
        src_col: str,
        dst_col: str,
        time_col: str,
        edge_feature_col: List[str] | None = None,
    ) -> DGData:
        # TODO: Node Events not supported
        file_path = str(file_path) if isinstance(file_path, pathlib.Path) else file_path
        with open(file_path, newline='') as f:
            reader = list(csv.DictReader(f))  # Assumes the whole things fits in memory
            num_edges = len(reader)

        edge_index = torch.empty((num_edges, 2), dtype=torch.long)
        timestamps = torch.empty(num_edges, dtype=torch.long)
        edge_feats = None
        if edge_feature_col is not None:
            edge_feats = torch.empty((num_edges, len(edge_feature_col)))

        for i, row in enumerate(reader):
            edge_index[i, 0] = int(row[src_col])
            edge_index[i, 1] = int(row[dst_col])
            timestamps[i] = int(row[time_col])
            if edge_feature_col is not None:
                # This is likely better than creating a tensor copy for every event
                for j, col in enumerate(edge_feature_col):
                    edge_feats[i, j] = float(row[col])  # type: ignore
        return cls.from_raw(
            edge_timestamps=timestamps, edge_index=edge_index, edge_feats=edge_feats
        )

    @classmethod
    def from_pandas(
        cls,
        df: 'pandas.DataFrame',  # type: ignore
        src_col: str,
        dst_col: str,
        time_col: str,
        edge_feature_col: str | None = None,
    ) -> DGData:
        # TODO: Node Events not supported

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

        edge_index = torch.from_numpy(df[[src_col, dst_col]].to_numpy()).long()
        timestamps = torch.from_numpy(df[time_col].to_numpy()).long()
        if edge_feature_col is None:
            edge_feats = None
        else:
            edge_feats = torch.Tensor(df[edge_feature_col].tolist())
        return cls.from_raw(
            edge_timestamps=timestamps, edge_index=edge_index, edge_feats=edge_feats
        )

    @classmethod
    def from_tgb(cls, name: str, split: str = 'all', **kwargs: Any) -> DGData:
        def _check_tgb_import() -> 'LinkPropPredDataset':  # type: ignore
            try:
                from tgb.linkproppred.dataset import LinkPropPredDataset

                return LinkPropPredDataset
            except ImportError:
                err_msg = 'User requires tgb to initialize a DGraph from a tgb dataset '
                raise ImportError(err_msg)

        LinkPropPredDataset = _check_tgb_import()

        # TODO: Node Events not supported
        if name.startswith('tgbl-'):
            dataset = LinkPropPredDataset(name=name, **kwargs)  # type: ignore
        elif name.startswith('tgbn-'):
            raise ValueError(f'Not Implemented dataset: {name}')
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
        return cls.from_raw(
            edge_timestamps=timestamps, edge_index=edge_index, edge_feats=edge_feats
        )

    @classmethod
    def from_any(
        cls,
        data: str | pathlib.Path | 'pandas.DataFrame',  # type: ignore
        **kwargs: Any,
    ) -> DGData:
        if isinstance(data, (str, pathlib.Path)):
            data_str = str(data)
            if data_str.startswith('tgbl-'):
                return cls.from_tgb(name=data_str, **kwargs)
            if data_str.endswith('.csv'):
                return cls.from_csv(data, **kwargs)
            raise ValueError(
                f'Unsupported file format or dataset identifier: {data_str}'
            )
        else:
            return cls.from_pandas(data, **kwargs)
