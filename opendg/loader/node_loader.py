from typing import Any, List

import torch

from opendg.graph import DGraph


class NodeLoader(torch.utils.data.DataLoader):
    r"""A data loader which samples DGraph mini-batches according to a node sampler.

    Args:
        dg (DGraph): The dynamic graph from which to load the data.
        node_sampler (TODO): TODO
        batch_size (int, optional): How many samples per batch to load (default = 1).
        **kwargs (optional): Additional arguments to torch.utils.data.DataLoader.
    """

    def __init__(
        self, dg: DGraph, node_sampler: None, batch_size: int = 1, **kwargs: Any
    ):
        self._dg = dg
        self._sampler = node_sampler
        self._batch_size = batch_size

        # TODO: Initialize node batches properly
        node_batch = range(dg.num_nodes)
        super().__init__(node_batch, 1, shuffle=False, collate_fn=self, **kwargs)

    def __call__(self, node_batch: List[int]) -> DGraph:
        # TODO: Apply node sampler, then slice nodes from DGraph
        # nodes_batch = self._sampler.sample(node_batch, df, **some additional stuff)
        return self._dg.slice_nodes(
            node_batch
        )  # TODO: Figure out copy / in-place semantics
