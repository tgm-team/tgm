from __future__ import annotations

import torch
from typing import List

from tgm import DGBatch, DGraph
from tgm.constants import PADDED_NODE_ID
from tgm.hooks import StatelessHook
from tgm.util.logging import _get_logger

logger = _get_logger(__name__)


class DeduplicationHook(StatelessHook):
    """Deduplicate node IDs from batch fields and create index mappings to unique node embeddings.

    Note: Supports batches with or without negative samples and multi-hop neighbors.
    """

    def __init__(self, seed_nodes_keys: List[str] = []) -> None:
        super().__init__()
        self.requires = {'edge_src', 'edge_dst'} | set(seed_nodes_keys)
        self.produces = {'unique_nids', 'global_to_local'}

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        nids = [batch.edge_src, batch.edge_dst] # TODO: Need to retrieve seed nodes. Have NodeSeedable?
        for node_attr in self.requires:
            if not hasattr(batch,node_attr):
                raise ValueError(f'Missing seed node attribut {node_attr}')
            
            if  'nbr_nids' in node_attr :
                for hop in range(len(batch.nbr_nids)):
                    nbr_nodes = batch.nbr_nids[hop].flatten()
                    nbr_mask = nbr_nodes != PADDED_NODE_ID
                    nids.append(nbr_nodes[nbr_mask].flatten().to(batch.edge_src.device))
            else:
                seed_node = getattr(batch,node_attr)
                if seed_node is not None:
                    nids.append(seed_node)

        all_nids = torch.cat(nids, dim=0)
        unique_nids = torch.unique(all_nids, sorted=True)

        batch.unique_nids = unique_nids  # type: ignore
        batch.global_to_local = lambda x: torch.searchsorted(unique_nids, x).int()  # type: ignore
        logger.debug(
            'Deduplicated batch: %d ids to %d unique ids',
            all_nids.numel(),
            unique_nids.numel(),
        )

        return batch
