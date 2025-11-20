from typing import Set

import torch

from tgm.core.batch import DGBatch
from tgm.core.graph import DGraph
from tgm.hooks.base import StatelessHook
from tgm.util.logging import _get_logger

logger = _get_logger(__name__)


class BasicBatchAnalyticsHook(StatelessHook):
    """Compute simple batch-level statistics.

    This is a refactor of the example hook from
    `examples/analytics/basic_stats.py` into an official package module.
    """

    requires: Set[str] = set()

    produces = {
        'num_edge_events',
        'num_node_events',
        'num_timestamps',
        'num_unique_nodes',
        'avg_degree',
        'num_repeated_events',
    }

    def _count_edge_events(self, batch: DGBatch) -> int:
        return int(batch.src.numel()) if batch.src is not None else 0

    def _count_node_events(self, batch: DGBatch) -> int:
        return int(batch.node_ids.numel()) if batch.node_ids is not None else 0

    def _count_timestamps(self, batch: DGBatch) -> int:
        num_edge_ts = batch.time.numel() if batch.time is not None else 0
        num_node_ts = batch.node_times.numel() if batch.node_times is not None else 0
        return num_edge_ts + num_node_ts

    def _compute_unique_nodes(self, batch: DGBatch) -> int:
        node_tensors = []
        if batch.src is not None and batch.src.numel() > 0:
            node_tensors.append(batch.src)
        if batch.dst is not None and batch.dst.numel() > 0:
            node_tensors.append(batch.dst)
        if batch.node_ids is not None and batch.node_ids.numel() > 0:
            node_tensors.append(batch.node_ids)

        if len(node_tensors) == 0:
            return 0

        all_nodes = torch.cat(node_tensors, dim=0)
        unique_nodes = torch.unique(all_nodes)
        return int(unique_nodes.numel())

    def _compute_avg_degree(self, batch: DGBatch) -> float:
        src, dst = batch.src, batch.dst
        if src is None or dst is None or src.numel() == 0:
            return 0.0

        edge_nodes = torch.cat([src, dst], dim=0)
        edge_unique_nodes, edge_inverse = torch.unique(edge_nodes, return_inverse=True)
        degree_per_node = torch.bincount(
            edge_inverse, minlength=edge_unique_nodes.numel()
        )
        return float(degree_per_node.float().mean().item())

    def _count_repeated_edge_events(self, batch: DGBatch) -> int:
        src, dst, time = batch.src, batch.dst, batch.time
        if src is None or dst is None or time is None or src.numel() == 0:
            return 0

        edge_triples = torch.stack(
            [src.to(torch.long), dst.to(torch.long), time.to(torch.long)], dim=1
        )

        _, edge_counts = torch.unique(edge_triples, dim=0, return_counts=True)
        return int((edge_counts - 1).clamp(min=0).sum().item())

    def _count_repeated_node_events(self, batch: DGBatch) -> int:
        if batch.node_ids is None or batch.node_ids.numel() == 0:
            return 0

        node_ids = batch.node_ids.to(torch.long)
        if batch.node_times is None:
            return 0  # cannot detect duplicates without time

        node_times = batch.node_times.to(torch.long)
        node_pairs = torch.stack([node_ids, node_times], dim=1)

        _, node_counts = torch.unique(node_pairs, dim=0, return_counts=True)
        return int((node_counts - 1).clamp(min=0).sum().item())

    def __call__(
        self, dg: DGraph, batch: DGBatch
    ) -> DGBatch:  # NOTE: temporary Any to avoid mypy attribute error
        batch.num_edge_events = self._count_edge_events(batch)  # type: ignore[attr-defined]
        batch.num_node_events = self._count_node_events(batch)  # type: ignore[attr-defined]
        batch.num_timestamps = self._count_timestamps(batch)  # type: ignore[attr-defined]
        batch.num_unique_nodes = self._compute_unique_nodes(batch)  # type: ignore[attr-defined]
        batch.avg_degree = self._compute_avg_degree(batch)  # type: ignore[attr-defined]
        edge_rep = self._count_repeated_edge_events(batch)
        node_rep = self._count_repeated_node_events(batch)
        batch.num_repeated_events = edge_rep + node_rep  # type: ignore[attr-defined]

        return batch
