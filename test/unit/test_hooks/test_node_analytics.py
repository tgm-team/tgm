import pytest
import torch

from tgm import DGBatch, DGraph
from tgm.data import DGData
from tgm.hooks.node_analytics import NodeAnalyticsHook


@pytest.fixture
def simple_dgraph():
    """Create a simple dynamic graph for testing."""
    # Create a small temporal graph with 5 nodes
    src = torch.tensor([0, 0, 1, 2, 1, 3, 2, 0, 4], dtype=torch.int32)
    dst = torch.tensor([1, 2, 2, 3, 0, 4, 1, 3, 0], dtype=torch.int32)
    time = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4], dtype=torch.int32)

    edge_index = torch.stack([src, dst], dim=1)
    data = DGData.from_raw(time, edge_index)
    return DGraph(data)


def test_node_analytics_hook_initialization():
    """Test that NodeAnalyticsHook initializes correctly."""
    tracked_nodes = torch.tensor([0, 1, 2])
    hook = NodeAnalyticsHook(tracked_nodes=tracked_nodes, num_nodes=5)

    assert hook.tracked_nodes.numel() == 3
    assert hook.num_nodes == 5
    assert hook.compute_moving_avg is True
    assert hook.alpha == 0.3


def test_node_analytics_hook_initialization_errors():
    """Test that NodeAnalyticsHook raises errors for invalid inputs."""
    tracked_nodes = torch.tensor([0, 1, 2])

    with pytest.raises(ValueError, match='num_nodes must be positive'):
        NodeAnalyticsHook(tracked_nodes=tracked_nodes, num_nodes=0)

    with pytest.raises(ValueError, match='alpha must be in range'):
        NodeAnalyticsHook(tracked_nodes=tracked_nodes, num_nodes=5, alpha=0.0)

    with pytest.raises(ValueError, match='alpha must be in range'):
        NodeAnalyticsHook(tracked_nodes=tracked_nodes, num_nodes=5, alpha=1.5)


def test_node_analytics_hook_basic_stats(simple_dgraph):
    """Test basic statistics computation."""
    tracked_nodes = torch.tensor([0, 1, 2])
    hook = NodeAnalyticsHook(
        tracked_nodes=tracked_nodes, num_nodes=5, compute_moving_avg=False
    )

    # Materialize first batch (2 edges at time 0)
    batch1 = simple_dgraph.slice_events(0, 2).materialize()
    batch1 = hook(simple_dgraph, batch1)

    # Check that node_stats is populated
    assert hasattr(batch1, 'node_stats')
    assert len(batch1.node_stats) > 0

    # Node 0 should have appeared in this batch
    assert 0 in batch1.node_stats
    node_0_stats = batch1.node_stats[0]

    # Check expected fields
    assert 'degree' in node_0_stats
    assert 'activity' in node_0_stats
    assert 'engagement' in node_0_stats
    assert 'lifetime' in node_0_stats
    assert 'appearances' in node_0_stats

    # First batch: activity should be 1.0
    assert node_0_stats['activity'] == 1.0
    assert node_0_stats['appearances'] == 1


def test_node_analytics_hook_degree_computation(simple_dgraph):
    """Test degree computation."""
    tracked_nodes = torch.tensor([0, 1, 2])
    hook = NodeAnalyticsHook(
        tracked_nodes=tracked_nodes, num_nodes=5, compute_moving_avg=False
    )

    # First batch has edges: (0->1), (0->2)
    # Node 0 has degree 2 (2 outgoing)
    # Node 1 has degree 1 (1 incoming)
    # Node 2 has degree 1 (1 incoming)
    batch = simple_dgraph.slice_events(0, 2).materialize()
    batch = hook(simple_dgraph, batch)

    assert batch.node_stats[0]['degree'] == 2
    assert batch.node_stats[1]['degree'] == 1
    assert batch.node_stats[2]['degree'] == 1


def test_node_analytics_hook_activity_over_batches(simple_dgraph):
    """Test activity tracking over multiple batches."""
    tracked_nodes = torch.tensor([0, 1])
    hook = NodeAnalyticsHook(
        tracked_nodes=tracked_nodes, num_nodes=5, compute_moving_avg=False
    )

    # Process multiple batches
    batch1 = simple_dgraph.slice_events(0, 2).materialize()
    batch1 = hook(simple_dgraph, batch1)

    batch2 = simple_dgraph.slice_events(2, 4).materialize()
    batch2 = hook(simple_dgraph, batch2)

    # Node 0 appeared in batch 1, node 1 appeared in both batches
    # Activity for node 1 should be 2/2 = 1.0
    assert batch2.node_stats[1]['activity'] == 1.0
    assert batch2.node_stats[1]['appearances'] == 2


def test_node_analytics_hook_edge_statistics(simple_dgraph):
    """Test edge-level statistics."""
    tracked_nodes = torch.tensor([0, 1, 2])
    hook = NodeAnalyticsHook(
        tracked_nodes=tracked_nodes, num_nodes=5, compute_moving_avg=False
    )

    # First batch - all edges should be new
    batch1 = simple_dgraph.slice_events(0, 2).materialize()
    batch1 = hook(simple_dgraph, batch1)

    assert hasattr(batch1, 'edge_stats')
    assert 'edge_novelty' in batch1.edge_stats
    assert 'edge_density' in batch1.edge_stats
    assert 'new_edge_count' in batch1.edge_stats

    # First batch: all edges are new
    assert batch1.edge_stats['new_edge_count'] == 2
    assert batch1.edge_stats['edge_novelty'] == 1.0  # 2/2 = 1.0

    # Second batch with different edges
    batch2 = simple_dgraph.slice_events(2, 4).materialize()
    batch2 = hook(simple_dgraph, batch2)

    # All edges in batch2 should also be new (different edge pairs)
    assert batch2.edge_stats['new_edge_count'] == 2


def test_node_analytics_hook_moving_average(simple_dgraph):
    """Test exponential moving average computation."""
    tracked_nodes = torch.tensor([0])
    hook = NodeAnalyticsHook(
        tracked_nodes=tracked_nodes, num_nodes=5, compute_moving_avg=True, alpha=0.5
    )

    # First batch
    batch1 = simple_dgraph.slice_events(0, 2).materialize()
    batch1 = hook(simple_dgraph, batch1)

    degree1 = batch1.node_stats[0]['degree']
    degree_ema1 = batch1.node_stats[0]['degree_ema']

    # First observation: EMA should equal the actual value
    assert degree_ema1 == degree1

    # Second batch
    batch2 = simple_dgraph.slice_events(2, 4).materialize()
    batch2 = hook(simple_dgraph, batch2)

    if 0 in batch2.node_stats:
        # EMA should be updated
        assert 'degree_ema' in batch2.node_stats[0]


def test_node_analytics_hook_neighbor_tracking(simple_dgraph):
    """Test neighbor tracking and new neighbor detection."""
    tracked_nodes = torch.tensor([0])
    hook = NodeAnalyticsHook(
        tracked_nodes=tracked_nodes, num_nodes=5, compute_moving_avg=False
    )

    # First batch: edges (0->1), (0->2)
    # Node 0 has 2 new neighbors: 1 and 2
    batch1 = simple_dgraph.slice_events(0, 2).materialize()
    batch1 = hook(simple_dgraph, batch1)

    assert batch1.node_stats[0]['new_neighbors'] == 2

    # Second batch: edges (1->2), (2->3)
    # Node 0 doesn't appear, but we can check batch3

    # Third batch: edges (1->0), (3->4)
    # Node 0 connects with node 1 (already seen)
    batch3 = simple_dgraph.slice_events(4, 6).materialize()
    batch3 = hook(simple_dgraph, batch3)

    # Node 0 connects to node 1, which is not a new neighbor
    assert batch3.node_stats[0]['new_neighbors'] == 0


def test_node_analytics_hook_reset_state(simple_dgraph):
    """Test that reset_state clears all internal state."""
    tracked_nodes = torch.tensor([0, 1])
    hook = NodeAnalyticsHook(
        tracked_nodes=tracked_nodes, num_nodes=5, compute_moving_avg=False
    )

    # Process a batch
    batch1 = simple_dgraph.slice_events(0, 2).materialize()
    batch1 = hook(simple_dgraph, batch1)

    assert len(batch1.node_stats) > 0

    # Reset state
    hook.reset_state()

    # Process another batch - should start fresh
    batch2 = simple_dgraph.slice_events(0, 2).materialize()
    batch2 = hook(simple_dgraph, batch2)

    # After reset, appearances should be 1 again
    assert batch2.node_stats[0]['appearances'] == 1


def test_node_analytics_hook_empty_batch(simple_dgraph):
    """Test behavior with empty batch."""
    tracked_nodes = torch.tensor([0, 1])
    hook = NodeAnalyticsHook(
        tracked_nodes=tracked_nodes, num_nodes=5, compute_moving_avg=False
    )

    # Create an empty batch
    empty_batch = DGBatch(
        src=torch.tensor([], dtype=torch.int32),
        dst=torch.tensor([], dtype=torch.int32),
        time=torch.tensor([], dtype=torch.int32),
    )

    result = hook(simple_dgraph, empty_batch)

    # Should handle empty batch gracefully
    assert hasattr(result, 'node_stats')
    assert hasattr(result, 'edge_stats')


def test_node_analytics_hook_unique_tracked_nodes():
    """Test that duplicate tracked nodes are deduplicated."""
    tracked_nodes = torch.tensor([0, 1, 1, 2, 0])
    hook = NodeAnalyticsHook(tracked_nodes=tracked_nodes, num_nodes=5)

    # Should have only 3 unique tracked nodes
    assert hook.tracked_nodes.numel() == 3
    assert set(hook.tracked_nodes.tolist()) == {0, 1, 2}
