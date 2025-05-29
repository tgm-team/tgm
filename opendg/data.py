from dataclasses import dataclass

import torch


@dataclass
class DGData:
    r"""Bundles dynamic graph data to be forwarded to DGStorage."""

    edge_index: torch.Tensor  # [num_edges, 2]
    timestamps: torch.Tensor  # [num_edges]
    edge_feats: torch.Tensor | None = None  # [num_edges, D_edge]
    node_feats: torch.Tensor | None = None  # [num_nodes, D_node]

    def __post_init__(self) -> None:
        # Validate edge index
        if not isinstance(self.edge_index, torch.Tensor):
            raise TypeError('edge_index must be a torch.Tensor')
        if self.edge_index.ndim != 2 or self.edge_index.shape[1] != 2:
            raise ValueError('edge_index must have shape [num_edges, 2]')

        num_edges = self.edge_index.shape[1]
        if num_edges == 0:
            raise ValueError('empty graphs not supported')

        # Validate edge features
        if self.edge_feats is not None:
            if not isinstance(self.edge_feats, torch.Tensor):
                raise TypeError('edge_feats must be a torch.Tensor')
            if self.edge_feats.shape[0] != num_edges:
                raise ValueError('edge_feats must have shape [num_edges, D_edge]')

        # Validate node features
        if self.node_feats is not None:
            if not isinstance(self.node_feats, torch.Tensor):
                raise TypeError('node_feats must be a torch.Tensor')

        # Validate timestamps
        if not isinstance(self.timestamps, torch.Tensor):
            raise TypeError('timestamps must be a torch.Tensor')
        if self.timestamps.ndim != 1 or self.timestamps.shape[0] != num_edges:
            raise ValueError('timestamps must have shape [num_edges]')

        # Sort if necessary
        if not torch.all(torch.diff(self.timestamps) >= 0):
            sorted_idx = torch.argsort(self.timestamps)
            self.timestamps = self.timestamps[sorted_idx]
            self.edge_index = self.edge_index[sorted_idx]
            if self.edge_feats is not None:
                self.edge_feats = self.edge_feats[sorted_idx]
