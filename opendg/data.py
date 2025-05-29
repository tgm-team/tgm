from dataclasses import dataclass

from torch import Tensor


@dataclass
class DGData:
    r"""Bundles dynamic graph data to be forwarded to DGStorage."""

    edge_index: Tensor  # [num_events, 2]
    timestamps: Tensor  # [num_events]
    edge_features: Tensor | None = None  # [num_events, D_edge]
    node_features: Tensor | None = None  # [num_nodes, D_node]

    def __post_init__(self) -> None:
        # TODO:Validate tensor shapes, ensure sorted by time
        pass
