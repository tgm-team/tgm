from __future__ import annotations

import warnings
from dataclasses import dataclass

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

        # Sort if necessary
        if not torch.all(torch.diff(self.timestamps) >= 0):
            warnings.warn('received non-chronological events, sorting by time')
            sorted_idx = torch.argsort(self.timestamps)
            self.timestamps = self.timestamps[sorted_idx]

            # Build reverse mapping from old to new indices
            inverse_sort_idx = torch.empty_like(sorted_idx)
            inverse_sort_idx[sorted_idx] = torch.arange(len(sorted_idx))

            # Reorder edge and node events
            self.edge_event_idx = inverse_sort_idx[self.edge_event_idx]
            if self.node_event_idx is not None:
                self.node_event_idx = inverse_sort_idx[self.node_event_idx]

            # Reorder edge_index and edge_feats using edge_event_idx
            self.edge_index = self.edge_index[self.edge_event_idx]
            if self.edge_feats is not None:
                self.edge_feats = self.edge_feats[self.edge_event_idx]

            # Reorder dynamic node features
            if self.node_event_idx is not None:
                self.node_ids = self.node_ids[self.node_event_idx]  # type: ignore
                if self.dynamic_node_feats is not None:
                    self.dynamic_node_feats = self.dynamic_node_feats[
                        self.node_event_idx
                    ]

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

        # Sort event timeline
        sorted_idx = torch.argsort(timestamps)
        timestamps = timestamps[sorted_idx]
        event_types = event_types[sorted_idx]

        # Compute event masks
        edge_event_idx = (event_types == 0).nonzero(as_tuple=True)[0]
        node_event_idx = (
            (event_types == 1).nonzero(as_tuple=True)[0]
            if node_timestamps is not None
            else None
        )

        # Reorder data
        edge_perm = torch.argsort(edge_timestamps)
        edge_index = edge_index[edge_perm]
        edge_feats = edge_feats[edge_perm] if edge_feats is not None else None

        if node_timestamps is not None:
            node_perm = torch.argsort(node_timestamps)
            node_ids = node_ids[node_perm] if node_ids is not None else None
            dynamic_node_feats = (
                dynamic_node_feats[node_perm]
                if dynamic_node_feats is not None
                else None
            )
        else:
            node_ids = None
            dynamic_node_feats = None

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
