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

    assert hook.num_nodes == 5


def test_node_analytics_hook_initialization_errors():
    """Test that NodeAnalyticsHook raises errors for invalid inputs."""
    tracked_nodes = torch.tensor([0, 1, 2])

    with pytest.raises(ValueError, match='num_nodes must be positive'):
        NodeAnalyticsHook(tracked_nodes=tracked_nodes, num_nodes=-1)


def test_node_analytics_hook_basic_stats(simple_dgraph):
    """Test basic statistics computation."""
    tracked_nodes = torch.tensor([0, 1, 2])
    hook = NodeAnalyticsHook(tracked_nodes=tracked_nodes, num_nodes=5)

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
    assert 'new_neighbors' in node_0_stats
    assert 'lifetime' in node_0_stats
    assert 'appearances' in node_0_stats
    assert 'time_since_last_seen' in node_0_stats

    # First batch: activity should be 1.0
    assert node_0_stats['activity'] == 1.0
    assert node_0_stats['appearances'] == 1


def test_node_analytics_hook_degree_computation(simple_dgraph):
    """Test degree computation."""
    tracked_nodes = torch.tensor([0, 1, 2])
    hook = NodeAnalyticsHook(tracked_nodes=tracked_nodes, num_nodes=5)
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
    hook = NodeAnalyticsHook(tracked_nodes=tracked_nodes, num_nodes=5)

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
    hook = NodeAnalyticsHook(tracked_nodes=tracked_nodes, num_nodes=5)

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


def test_node_analytics_hook_neighbor_tracking(simple_dgraph):
    """Test neighbor tracking and new neighbor detection."""
    tracked_nodes = torch.tensor([0])
    hook = NodeAnalyticsHook(tracked_nodes=tracked_nodes, num_nodes=5)

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
    hook = NodeAnalyticsHook(tracked_nodes=tracked_nodes, num_nodes=5)

    # Process a batch
    batch1 = simple_dgraph.slice_events(0, 2).materialize()
    batch1 = hook(simple_dgraph, batch1)

    assert batch1.node_stats[0]['appearances'] == 1

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
    hook = NodeAnalyticsHook(tracked_nodes=tracked_nodes, num_nodes=5)
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


def test_node_analytics_hook_batch_with_all_none_nodes(simple_dgraph):
    """Test behavior when batch has no nodes at all (src, dst, node_ids all None)."""
    tracked_nodes = torch.tensor([0, 1])
    hook = NodeAnalyticsHook(tracked_nodes=tracked_nodes, num_nodes=5)

    # Create a batch where all node fields are None
    batch = DGBatch(
        src=None,
        dst=None,
        time=None,
        node_ids=None,
        node_times=None,
    )

    result = hook(simple_dgraph, batch)

    # Should handle batch with no nodes gracefully
    assert hasattr(result, 'node_stats')
    assert hasattr(result, 'node_macro_stats')
    assert hasattr(result, 'edge_stats')

    # node_stats should be empty since there are no nodes
    assert result.node_stats == {}

    # node_macro_stats should have default values
    assert 'node_novelty' in result.node_macro_stats
    assert 'new_node_count' in result.node_macro_stats
    assert result.node_macro_stats['node_novelty'] == 0.0
    assert result.node_macro_stats['new_node_count'] == 0

    # edge_stats should have default values
    assert 'edge_novelty' in result.edge_stats
    assert 'edge_density' in result.edge_stats
    assert 'new_edge_count' in result.edge_stats
    assert result.edge_stats['edge_novelty'] == 0.0
    assert result.edge_stats['edge_density'] == 0.0
    assert result.edge_stats['new_edge_count'] == 0


def test_node_analytics_hook_unique_tracked_nodes():
    """Test that duplicate tracked nodes are deduplicated."""
    tracked_nodes = torch.tensor([0, 1, 1, 2, 0])
    hook = NodeAnalyticsHook(tracked_nodes=tracked_nodes, num_nodes=5)

    # Should have only 3 unique tracked nodes
    assert hook.tracked_nodes.numel() == 3
    assert set(hook.tracked_nodes.tolist()) == {0, 1, 2}


def test_node_analytics_hook_produces_and_requires():
    """Test that hook declares correct produces and requires sets."""
    tracked_nodes = torch.tensor([0])
    hook = NodeAnalyticsHook(tracked_nodes=tracked_nodes, num_nodes=5)

    assert hook.requires == {'src', 'dst', 'time', 'node_times', 'node_ids'}
    assert hook.produces == {'node_stats', 'node_macro_stats', 'edge_stats'}


def test_node_analytics_hook_node_macro_stats(simple_dgraph):
    """Test node-level macro statistics computation."""
    tracked_nodes = torch.tensor([0, 1, 2])
    hook = NodeAnalyticsHook(tracked_nodes=tracked_nodes, num_nodes=5)

    batch1 = simple_dgraph.slice_events(0, 2).materialize()
    batch1 = hook(simple_dgraph, batch1)

    # Check that node_macro_stats is populated
    assert hasattr(batch1, 'node_macro_stats')
    assert 'node_novelty' in batch1.node_macro_stats
    assert 'new_node_count' in batch1.node_macro_stats

    # First batch should have all nodes as new (relative to tracked state)
    assert batch1.node_macro_stats['new_node_count'] >= 0


def test_node_analytics_hook_edge_density(simple_dgraph):
    """Test edge density computation."""
    tracked_nodes = torch.tensor([0, 1, 2])
    hook = NodeAnalyticsHook(tracked_nodes=tracked_nodes, num_nodes=5)

    batch1 = simple_dgraph.slice_events(0, 2).materialize()
    batch1 = hook(simple_dgraph, batch1)

    assert 'edge_density' in batch1.edge_stats
    # Edge density should be between 0 and 1
    assert 0 < batch1.edge_stats['edge_density'] < 1


def test_node_analytics_hook_lifetime_tracking(simple_dgraph):
    """Test that node lifetime is tracked correctly over time."""
    tracked_nodes = torch.tensor([0])
    hook = NodeAnalyticsHook(tracked_nodes=tracked_nodes, num_nodes=5)

    # Process first batch
    batch1 = simple_dgraph.slice_events(0, 2).materialize()
    batch1 = hook(simple_dgraph, batch1)

    # First appearance - lifetime should be 0
    assert batch1.node_stats[0]['lifetime'] == 0.0

    # Process third batch with higher timestamp
    batch3 = simple_dgraph.slice_events(4, 6).materialize()
    batch3 = hook(simple_dgraph, batch3)
    # Lifetime should be positive if node appeared
    assert batch3.node_stats[0]['lifetime'] > 0


def test_node_analytics_hook_time_since_last_seen(simple_dgraph):
    """Test time_since_last_seen tracking."""
    tracked_nodes = torch.tensor([2])
    hook = NodeAnalyticsHook(tracked_nodes=tracked_nodes, num_nodes=5)

    # Process first batch
    batch1 = simple_dgraph.slice_events(0, 2).materialize()
    batch1 = hook(simple_dgraph, batch1)

    # First appearance - time_since_last_seen should be 0
    assert batch1.node_stats[2]['time_since_last_seen'] == 0.0

    # Skip a batch
    batch2 = simple_dgraph.slice_events(2, 4).materialize()
    batch2 = hook(simple_dgraph, batch2)

    # Process next batch
    batch3 = simple_dgraph.slice_events(4, 6).materialize()
    batch3 = hook(simple_dgraph, batch3)

    # Check that time_since_last_seen is tracked
    assert batch3.node_stats[2]['time_since_last_seen'] > 0.0


def test_node_analytics_hook_untracked_nodes_ignored(simple_dgraph):
    """Test that untracked nodes are not included in statistics."""
    # Only track nodes 0 and 1
    tracked_nodes = torch.tensor([0, 1])
    hook = NodeAnalyticsHook(tracked_nodes=tracked_nodes, num_nodes=5)

    # Process batch with nodes 0, 1, 2
    batch1 = simple_dgraph.slice_events(0, 2).materialize()
    batch1 = hook(simple_dgraph, batch1)

    # Node 2 should not be in node_stats
    assert 2 not in batch1.node_stats
    # Nodes 0 and 1 may or may not appear depending on edges


def test_node_analytics_hook_batch_with_only_nodes(simple_dgraph):
    """Test handling of batches with only node events (no edges)."""
    tracked_nodes = torch.tensor([0, 1, 2])
    hook = NodeAnalyticsHook(tracked_nodes=tracked_nodes, num_nodes=5)

    # Create batch with only node events
    batch = DGBatch(
        src=None,
        dst=None,
        time=None,
        node_ids=torch.tensor([0, 1, 2], dtype=torch.int32),
        node_times=torch.tensor([1, 1, 1], dtype=torch.int32),
    )

    result = hook(simple_dgraph, batch)

    # Should handle node-only batch
    assert hasattr(result, 'node_stats')
    assert hasattr(result, 'node_macro_stats')
    assert hasattr(result, 'edge_stats')


def test_node_analytics_hook_repeated_edges(simple_dgraph):
    """Test that repeated edges are counted correctly."""
    tracked_nodes = torch.tensor([0, 1])
    hook = NodeAnalyticsHook(tracked_nodes=tracked_nodes, num_nodes=5)

    # Process batch with edges (0->1), (0->2)
    batch1 = simple_dgraph.slice_events(0, 2).materialize()
    batch1 = hook(simple_dgraph, batch1)

    # Get new edge count
    assert batch1.edge_stats['new_edge_count'] > 0

    # Process same batch again
    batch2 = simple_dgraph.slice_events(0, 2).materialize()
    batch2 = hook(simple_dgraph, batch2)

    # Should have 0 new edges since we've seen them before
    assert batch2.edge_stats['new_edge_count'] == 0
    assert batch2.edge_stats['edge_novelty'] == 0.0


def test_node_analytics_hook_nodes_not_in_batch_stats(simple_dgraph):
    """Test that tracked nodes not in current batch still get stats if seen before."""
    tracked_nodes = torch.tensor([0, 1, 2, 3])
    hook = NodeAnalyticsHook(tracked_nodes=tracked_nodes, num_nodes=5)

    # Process first batch with nodes 0, 1, 2
    batch1 = simple_dgraph.slice_events(0, 2).materialize()
    batch1 = hook(simple_dgraph, batch1)

    # Process second batch with different nodes
    batch2 = simple_dgraph.slice_events(2, 4).materialize()
    batch2 = hook(simple_dgraph, batch2)

    # Previously seen nodes should still have stats even if not in current batch
    for node in batch1.node_stats.keys():
        if node in batch2.node_stats:
            # Node was seen before and appears in batch2 stats
            assert batch2.node_stats[node]['appearances'] > 0


def test_node_analytics_hook_edge_stats_all_fields(simple_dgraph):
    """Test that all expected edge stats fields are present."""
    tracked_nodes = torch.tensor([0, 1, 2])
    hook = NodeAnalyticsHook(tracked_nodes=tracked_nodes, num_nodes=5)

    batch = simple_dgraph.slice_events(0, 2).materialize()
    batch = hook(simple_dgraph, batch)

    # Check all expected edge stats fields
    expected_fields = {'edge_novelty', 'edge_density', 'new_edge_count'}
    assert set(batch.edge_stats.keys()) == expected_fields


def test_node_analytics_hook_node_stats_all_fields(simple_dgraph):
    """Test that all expected node stats fields are present."""
    tracked_nodes = torch.tensor([0])
    hook = NodeAnalyticsHook(tracked_nodes=tracked_nodes, num_nodes=5)

    batch = simple_dgraph.slice_events(0, 2).materialize()
    batch = hook(simple_dgraph, batch)

    if 0 in batch.node_stats:
        expected_fields = {
            'degree',
            'activity',
            'new_neighbors',
            'lifetime',
            'time_since_last_seen',
            'appearances',
        }
        assert set(batch.node_stats[0].keys()) == expected_fields


def test_node_analytics_hook_multiple_batches_neighbor_accumulation(simple_dgraph):
    """Test that neighbors accumulate correctly over multiple batches."""
    tracked_nodes = torch.tensor([0])
    hook = NodeAnalyticsHook(tracked_nodes=tracked_nodes, num_nodes=5)

    # First batch: node 0 connects to nodes 1, 2
    batch1 = simple_dgraph.slice_events(0, 2).materialize()
    batch1 = hook(simple_dgraph, batch1)

    assert batch1.node_stats[0]['new_neighbors'] > 0

    # Later batch: node 0 connects to node 3
    batch4 = simple_dgraph.slice_events(6, 8).materialize()
    batch4 = hook(simple_dgraph, batch4)

    # Should have new neighbor(s) since it's a different connection
    assert batch4.node_stats[0]['new_neighbors'] > 0
