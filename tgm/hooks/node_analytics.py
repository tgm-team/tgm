from typing import Dict, Set

import torch
from torch import Tensor

from tgm.core.batch import DGBatch
from tgm.core.graph import DGraph
from tgm.hooks.base import StatefulHook
from tgm.util.logging import _get_logger

logger = _get_logger(__name__)


class NodeAnalyticsHook(StatefulHook):
    """Compute node-centric statistics for a specific set of tracked nodes.

    This hook maintains state across batches to compute temporal statistics
    for a specified set of nodes.

    Args:
        tracked_nodes (Tensor): 1D tensor of node IDs to track statistics for.
        num_nodes (int): Total number of nodes in the graph.

    Produces:
        node_stats (Dict[int, Dict[str, float]]): Dictionary mapping node_id to statistics:
            - degree: Number of edges connected to the node in the current batch.
            - activity: Fraction of unique timesteps in which the node has appeared.
            - new_neighbors: Number of new neighbors in which the node encountered in the current batch.
            - lifetime: Time since the node was first seen.
            - time_since_last_seen: Time since the node was last seen.
            - appearances: Total number of unique timesteps the node has appeared in.
        node_macro_stats (Dict): Batch-level node statistics:
            - node_novelty: Fraction of tracked nodes in the batch that are appearing for the first time.
            - new_node_count: Number of tracked nodes in the batch that are appearing for the first time.
        edge_stats (Dict): Batch-level edge statistics:
            - edge_novelty: Fraction of new edges in the batch, that is not seen in previous batches.
            - edge_density: Edges this batch / possible edges based on unique nodes
            - new_edge_count: Number of new edges in the batch, that is not seen in previous batches.
    """

    requires = {
        'edge_src',
        'edge_dst',
        'edge_time',
        'node_x_time',
        'node_x_nids',
    }
    produces = {'node_stats', 'node_macro_stats', 'edge_stats'}

    def __init__(self, tracked_nodes: Tensor, num_nodes: int) -> None:
        if num_nodes <= 0:
            raise ValueError('num_nodes must be positive')
        self.tracked_nodes = tracked_nodes.unique()
        self.num_nodes = num_nodes

        # Create a mask for fast lookup of tracked nodes
        self._tracked_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self._tracked_mask[self.tracked_nodes] = True

        # State dictionaries for each tracked node
        self._first_seen: Dict[int, float] = {}
        self._last_seen: Dict[int, float] = {}
        self._appearances: Dict[int, int] = {}  # Count of unique timesteps per node
        self._total_timesteps: Set[float] = set()  # Track all unique timesteps seen
        self._node_timesteps: Dict[int, Set[float]] = {  # Track timesteps per node
            int(node): set() for node in self.tracked_nodes.tolist()
        }

        # Neighbor tracking
        self._all_neighbors: Dict[int, Set[int]] = {
            int(node): set() for node in self.tracked_nodes.tolist()
        }
        self._engagement_sum: Dict[int, float] = {}

        # Edge tracking
        self._seen_edges: Set[tuple] = set()

    def _compute_node_degrees(self, batch: DGBatch, nodes: Tensor) -> Dict[int, int]:
        """Compute degree for each node in the given set."""
        if batch.edge_src is None or batch.edge_dst is None:
            return {int(n.item()): 0 for n in nodes}

        # Concatenate edge_src and edge_dst to count all edge connections
        edge_nodes = torch.cat([batch.edge_src, batch.edge_dst], dim=0)

        # Vectorized degree computation
        unique_nodes, inv = torch.unique(edge_nodes, return_inverse=True)
        counts = torch.bincount(inv)
        deg_map = dict(zip(unique_nodes.tolist(), counts.tolist()))
        return {int(n): deg_map.get(int(n), 0) for n in nodes.tolist()}

    def _compute_node_neighbors(
        self, batch: DGBatch, nodes: Tensor
    ) -> Dict[int, Set[int]]:
        """Get all neighbors for each node in this batch."""
        # Convert to numpy for faster iteration
        nodes_u = nodes.unique()
        nodes_set = set(int(n) for n in nodes_u)
        neighbors: Dict[int, Set[int]] = {int(n): set() for n in nodes_set}

        if batch.edge_src is None or batch.edge_dst is None:
            return neighbors

        # Convert to numpy arrays for faster iteration
        src_np = (
            batch.edge_src.cpu().numpy()
            if batch.edge_src.is_cuda
            else batch.edge_src.numpy()
        )
        dst_np = (
            batch.edge_dst.cpu().numpy()
            if batch.edge_dst.is_cuda
            else batch.edge_dst.numpy()
        )

        for edge_src, edge_dst in zip(src_np, dst_np):
            src_int, dst_int = int(edge_src), int(edge_dst)
            if src_int in nodes_set:
                neighbors[src_int].add(dst_int)
            if dst_int in nodes_set:
                neighbors[dst_int].add(src_int)

        return neighbors

    def _get_batch_timestamp(self, batch: DGBatch) -> float:
        """Extract a representative timestamp from the batch."""
        max_edge_time = 0.0
        max_node_time = 0.0

        if batch.edge_time is not None and batch.edge_time.numel() > 0:
            # Use the latest timestamp in the batch
            max_edge_time = batch.edge_time.max().item()
        if batch.node_x_time is not None and batch.node_x_time.numel() > 0:
            max_node_time = batch.node_x_time.max().item()

        return max(max_edge_time, max_node_time)

    def _compute_node_statistics(self, batch: DGBatch) -> Dict[str, float]:
        """Compute node-level statistics for the batch."""
        node_stats = {
            'node_novelty': 0.0,
            'new_node_count': 0,
        }

        if batch.node_x_nids is None or batch.node_x_nids.numel() == 0:
            return node_stats

        new_nodes = 0
        for node in batch.node_x_nids:
            node_id = int(node.item())
            if node_id not in self._first_seen:
                new_nodes += 1

            node_stats['new_node_count'] = new_nodes
            node_stats['node_novelty'] = (
                new_nodes / batch.node_x_nids.numel()
                if batch.node_x_nids.numel() > 0
                else 0.0
            )

        return node_stats

    def _compute_edge_statistics(self, batch: DGBatch) -> Dict[str, float]:
        """Compute edge-level statistics for the batch."""
        edge_stats = {
            'edge_novelty': 0.0,
            'edge_density': 0.0,
            'new_edge_count': 0,
        }

        if (
            batch.edge_src is None
            or batch.edge_dst is None
            or batch.edge_src.numel() == 0
        ):
            return edge_stats

        # Convert to numpy for faster iteration
        src_np = (
            batch.edge_src.cpu().numpy()
            if batch.edge_src.is_cuda
            else batch.edge_src.numpy()
        )
        dst_np = (
            batch.edge_dst.cpu().numpy()
            if batch.edge_dst.is_cuda
            else batch.edge_dst.numpy()
        )

        # Count new edges
        new_edges = 0
        for edge_src, edge_dst in zip(src_np, dst_np):
            edge_tuple = (int(edge_src), int(edge_dst))
            if edge_tuple not in self._seen_edges:
                new_edges += 1
                self._seen_edges.add(edge_tuple)

        edge_stats['new_edge_count'] = new_edges
        edge_stats['edge_novelty'] = (
            new_edges / batch.edge_src.numel() if batch.edge_src.numel() > 0 else 0.0
        )

        # Compute edge density
        unique_nodes = torch.unique(torch.cat([batch.edge_src, batch.edge_dst]))
        num_unique = unique_nodes.numel()
        possible_edges = num_unique * (num_unique - 1)  # Directed graph

        edge_stats['edge_density'] = (
            batch.edge_src.numel() / possible_edges if possible_edges > 0 else 0.0
        )

        return edge_stats

    def reset_state(self) -> None:
        """Reset internal state."""
        self._first_seen.clear()
        self._last_seen.clear()
        self._appearances.clear()
        self._total_timesteps.clear()
        self._node_timesteps = {
            int(node): set() for node in self.tracked_nodes.tolist()
        }
        self._all_neighbors = {int(node): set() for node in self.tracked_nodes.tolist()}
        self._engagement_sum.clear()
        self._seen_edges.clear()

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        """Compute node-centric statistics for tracked nodes in the batch."""
        # Get current timestamp
        current_time = self._get_batch_timestamp(batch)

        # Track unique timesteps in this batch (vectorized)
        time_tensors = [
            batch.edge_time,
            batch.node_x_time,
        ]

        for times in time_tensors:
            if times is None or times.numel() == 0:
                continue

            times_np = times.cpu().numpy() if times.is_cuda else times.numpy()
            self._total_timesteps.update(float(t) for t in times_np)

        # Find which tracked nodes appear in this batch
        all_batch_nodes = []
        if batch.edge_src is not None:
            all_batch_nodes.append(batch.edge_src)
        if batch.edge_dst is not None:
            all_batch_nodes.append(batch.edge_dst)
        if batch.node_x_nids is not None:
            all_batch_nodes.append(batch.node_x_nids)

        # No nodes in batch, return empty stats
        if not all_batch_nodes:
            node_batch_stats = self._compute_node_statistics(batch)
            edge_batch_stats = self._compute_edge_statistics(batch)
            batch.node_stats = {}  # type: ignore[attr-defined]
            batch.node_macro_stats = node_batch_stats  # type: ignore[attr-defined]
            batch.edge_stats = {**edge_batch_stats}  # type: ignore[attr-defined]
            return batch

        batch_nodes = torch.cat(all_batch_nodes, dim=0).unique()

        # Filter to only tracked nodes
        is_tracked = self._tracked_mask[batch_nodes]
        present_tracked_nodes = batch_nodes[is_tracked]

        # Compute degrees and neighbors for present tracked nodes
        node_degrees = self._compute_node_degrees(batch, present_tracked_nodes)
        node_neighbors = self._compute_node_neighbors(batch, present_tracked_nodes)

        # Compute statistics for each present tracked node
        node_stats = {}
        for node in present_tracked_nodes:
            node = int(node.item())
            # Update temporal tracking
            if node not in self._first_seen:
                self._first_seen[node] = current_time

            self._last_seen.get(node, None)
            self._last_seen[node] = current_time

            # Track unique timesteps for this node (vectorized)
            node_timesteps_in_batch: Set[float] = set()

            pairs = [
                (batch.edge_src, batch.edge_time),
                (batch.edge_dst, batch.edge_time),
                (batch.node_x_nids, batch.node_x_time),
            ]

            for ids, times in pairs:
                if ids is None or times is None:
                    continue

                ids_np = ids.cpu().numpy() if ids.is_cuda else ids.numpy()
                times_np = times.cpu().numpy() if times.is_cuda else times.numpy()

                mask = ids_np == node
                node_timesteps_in_batch.update(float(t) for t in times_np[mask])

            # Add new timesteps to node's timestep set
            self._node_timesteps[node].update(node_timesteps_in_batch)
            self._appearances[node] = len(self._node_timesteps[node])

            # Degree statistics
            degree = node_degrees[node]

            # Engagement statistics (distinct neighbors)
            current_neighbors = node_neighbors[node]
            previous_neighbors = self._all_neighbors[node]
            new_neighbors = current_neighbors - previous_neighbors

            # Update all neighbors
            self._all_neighbors[node].update(current_neighbors)

            # Compile statistics
            total_timesteps = len(self._total_timesteps) if self._total_timesteps else 1
            stats = {
                'degree': degree,
                'activity': self._appearances[node] / total_timesteps,
                'new_neighbors': len(new_neighbors),
                'lifetime': current_time - self._first_seen[node]
                if self._first_seen[node] is not None
                else 0.0,
                'time_since_last_seen': 0.0,
                'appearances': self._appearances[node],
            }

            node_stats[node] = stats

        # Add statistics for tracked nodes not in this batch
        for node_tensor in self.tracked_nodes:
            node = int(node_tensor.item())
            if node not in node_stats and node in self._last_seen:
                # Node has been seen before but not in this batch
                total_timesteps = (
                    len(self._total_timesteps) if self._total_timesteps else 1
                )
                stats = {
                    'degree': 0,
                    'activity': self._appearances.get(node, 0) / total_timesteps,
                    'new_neighbors': 0,
                    'lifetime': self._last_seen[node] - self._first_seen[node],
                    'time_since_last_seen': current_time - self._last_seen[node],
                    'appearances': self._appearances.get(node, 0),
                }

                node_stats[node] = stats

        # Compute batch-level statistics
        node_batch_stats = self._compute_node_statistics(batch)
        edge_batch_stats = self._compute_edge_statistics(batch)

        batch.node_stats = node_stats  # type: ignore[attr-defined]
        batch.node_macro_stats = node_batch_stats  # type: ignore[attr-defined]
        batch.edge_stats = edge_batch_stats  # type: ignore[attr-defined]

        return batch
