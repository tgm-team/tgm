import torch

from tgm.core.batch import DGBatch
from tgm.core.graph import DGraph
from tgm.hooks.base import StatelessHook
from tgm.util.logging import _get_logger

logger = _get_logger(__name__)


class BatchAnalyticsHook(StatelessHook):
    """Compute simple batch-level statistics."""

    requires = {
        'edge_src',
        'edge_dst',
        'edge_time',
        'node_x_time',
        'node_x_nids',
    }
    produces = {
        'num_edge_events',
        'num_node_events',
        'num_unique_timestamps',
        'num_unique_nodes',
        'avg_degree',
        'num_repeated_edge_events',
        'num_repeated_node_events',
    }

    def _count_edge_events(self, batch: DGBatch) -> int:
        return int(batch.edge_src.numel()) if batch.edge_src is not None else 0

    def _count_node_events(self, batch: DGBatch) -> int:
        return int(batch.node_x_nids.numel()) if batch.node_x_nids is not None else 0

    def _count_unique_timestamps(self, batch: DGBatch) -> int:
        edge_ts = batch.edge_time if batch.edge_time is not None else torch.tensor([])
        node_ts = (
            batch.node_x_time if batch.node_x_time is not None else torch.tensor([])
        )

        all_ts = torch.cat([edge_ts, node_ts], dim=0)
        unique_ts = torch.unique(all_ts)

        return int(unique_ts.numel())

    def _compute_unique_nodes(self, batch: DGBatch) -> int:
        fields = (batch.edge_src, batch.edge_dst, batch.node_x_nids)
        node_tensors = [t for t in fields if t is not None and t.numel() > 0]
        if not node_tensors:
            return 0
        all_nodes = torch.cat(node_tensors, dim=0)
        return int(torch.unique(all_nodes).numel())

    def _compute_avg_degree(self, batch: DGBatch) -> float:
        edge_src, edge_dst = batch.edge_src, batch.edge_dst
        if edge_src is None or edge_dst is None or edge_src.numel() == 0:
            return 0.0

        edge_nodes = torch.cat([edge_src, edge_dst], dim=0)
        edge_unique_nodes, edge_inverse = torch.unique(edge_nodes, return_inverse=True)
        degree_per_node = torch.bincount(
            edge_inverse, minlength=edge_unique_nodes.numel()
        )
        return float(degree_per_node.float().mean().item())

    def _count_repeated_edge_events(self, batch: DGBatch) -> int:
        edge_src, edge_dst, time = batch.edge_src, batch.edge_dst, batch.edge_time
        if (
            edge_src is None
            or edge_dst is None
            or time is None
            or edge_src.numel() == 0
        ):
            return 0

        edge_triples = torch.stack([edge_src, edge_dst, time], dim=1)

        _, edge_counts = torch.unique(edge_triples, dim=0, return_counts=True)
        return int((edge_counts - 1).clamp(min=0).sum().item())

    def _count_repeated_node_events(self, batch: DGBatch) -> int:
        if batch.node_x_nids is None or batch.node_x_nids.numel() == 0:
            return 0

        node_ids = batch.node_x_nids
        if batch.node_x_time is None:
            logger.debug('node_times are None, cannot detect repeated node events.')
            return 0

        node_times = batch.node_x_time
        node_pairs = torch.stack([node_ids, node_times], dim=1)

        _, node_counts = torch.unique(node_pairs, dim=0, return_counts=True)
        return int((node_counts - 1).clamp(min=0).sum().item())

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        batch.num_edge_events = self._count_edge_events(batch)  # type: ignore[attr-defined]
        batch.num_node_events = self._count_node_events(batch)  # type: ignore[attr-defined]
        batch.num_unique_timestamps = self._count_unique_timestamps(batch)  # type: ignore[attr-defined]
        batch.num_unique_nodes = self._compute_unique_nodes(batch)  # type: ignore[attr-defined]
        batch.avg_degree = self._compute_avg_degree(batch)  # type: ignore[attr-defined]
        batch.num_repeated_edge_events = self._count_repeated_edge_events(batch)  # type: ignore[attr-defined]
        batch.num_repeated_node_events = self._count_repeated_node_events(batch)  # type: ignore[attr-defined]

        return batch
