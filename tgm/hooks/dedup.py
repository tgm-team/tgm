from __future__ import annotations

from typing import Set

import torch

from tgm import DGBatch, DGraph
from tgm.hooks import StatelessHook
from tgm.util.logging import _get_logger

logger = _get_logger(__name__)


class DeduplicationHook(StatelessHook):
    """Deduplicate node IDs from batch fields and create index mappings to unique node embeddings.

    Note: Supports batches with or without negative samples and multi-hop neighbors.
    """

    requires: Set[str] = set()
    produces = {'unique_nids', 'global_to_local'}

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        nids = [batch.src, batch.dst]
        if hasattr(batch, 'neg'):
            batch.neg = batch.neg.to(batch.src.device)
            nids.append(batch.neg)
        if hasattr(batch, 'nbr_nids'):
            for hop in range(len(batch.nbr_nids)):
                nids.append(batch.nbr_nids[hop].to(batch.src.device))
        nids.append(
            batch.node_ids.to(batch.src.device)
        ) if batch.node_ids is not None else None

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
