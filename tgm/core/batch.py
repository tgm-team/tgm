from __future__ import annotations

from collections.abc import Iterable, Sized
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor


@dataclass
class DGBatch:
    """Container for a batch of events/materialized data from a DGraph.

    Each `DGBatch` holds edge and node information for a slice of a dynamic graph,
    including optional dynamic node features and edge features. Hooks read and write
    additional attributes to the container transparently during dataloading.

    Args:
        src (Tensor): Source node indices for edges in the batch. Shape `(E,)`.
        dst (Tensor): Destination node indices for edges in the batch. Shape `(E,)`.
        time (Tensor): Timestamps of each edge event. Shape `(E,)`.
        dynamic_node_feats (Tensor | None, optional): Dynamic node features for nodes
            in the batch. Tensor of shape `(T x V x d_node_dynamic)`.
        edge_feats (Tensor | None, optional): Edge features for the batch. Tensor
            of shape `(T x V x V x d_edge)`.
        node_times (Tensor | None, optional): Timestamps corresponding to dynamic node features.
        node_ids (Tensor | None, optional): Node IDs corresponding to dynamic node features.
    """

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
            elif isinstance(object, str):
                description = object
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
