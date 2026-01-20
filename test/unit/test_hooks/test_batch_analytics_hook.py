import pytest
import torch

from tgm import DGraph
from tgm.data import DGData
from tgm.hooks import BatchAnalyticsHook


@pytest.fixture
def dg():
    edge_index = torch.IntTensor(
        [
            [1, 2],
            [1, 2],
            [2, 3],
        ]
    )
    edge_time = torch.IntTensor([1, 1, 2])

    node_x_time = torch.IntTensor([5, 5, 6])
    node_x_nids = torch.IntTensor([2, 2, 3])
    node_x = torch.rand(3, 3)

    data = DGData.from_raw(
        edge_time=edge_time,
        edge_index=edge_index,
        node_x_time=node_x_time,
        node_x_nids=node_x_nids,
        node_x=node_x,
    )
    return DGraph(data)


def test_hook_dependancies():
    hook = BatchAnalyticsHook()
    assert hook.requires == {
        'edge_src',
        'edge_dst',
        'edge_time',
        'node_x_time',
        'node_x_nids',
    }
    assert hook.produces == {
        'num_edge_events',
        'num_node_events',
        'num_unique_timestamps',
        'num_unique_nodes',
        'avg_degree',
        'num_repeated_edge_events',
        'num_repeated_node_events',
    }


def test_hook_reset_state():
    assert BatchAnalyticsHook.has_state is False


def test_basic_analytics_num_events_and_timestamps(dg):
    hook = BatchAnalyticsHook()
    batch = dg.materialize()
    processed_batch = hook(dg, batch)

    # assert batch.node_x_nids is not None

    # edge and node events
    assert processed_batch.num_edge_events == 3
    assert processed_batch.num_node_events == 3

    # timestamps
    assert processed_batch.num_unique_timestamps == 4


def test_basic_analytics_unique_nodes_and_degree(dg):
    hook = BatchAnalyticsHook()
    batch = dg.materialize()
    processed_batch = hook(dg, batch)

    # unique nodes = {1, 2, 3}
    assert processed_batch.num_unique_nodes == 3

    # avg degree:
    # edges: (1,2), (2,3), (1,2)
    # degree: node1=2, node2=3, node3=1 -> avg=2.0
    assert processed_batch.avg_degree == pytest.approx(2.0)


def test_basic_analytics_repeated_events(dg):
    hook = BatchAnalyticsHook()
    batch = dg.materialize()
    processed_batch = hook(dg, batch)

    # repeated events:
    # repeated edge: (1,2,1) → appears twice → 1 duplicate
    # repeated node: (2,5)   → appears twice → 1 duplicate
    # total = 2
    assert processed_batch.num_repeated_edge_events == 1
    assert processed_batch.num_repeated_node_events == 1


def test_basic_analytics_empty_edges_and_nodes(dg):
    hook = BatchAnalyticsHook()
    batch = dg.materialize()

    # Force edges and nodes to be "present but empty"
    batch.edge_src = torch.empty(0, dtype=batch.edge_src.dtype)
    batch.edge_dst = torch.empty(0, dtype=batch.edge_dst.dtype)
    batch.edge_time = torch.empty(0, dtype=batch.edge_time.dtype)
    batch.node_x_nids = torch.empty(0, dtype=batch.node_x_nids.dtype)
    batch.node_x_time = torch.empty(0, dtype=batch.node_x_time.dtype)

    processed_batch = hook(dg, batch)

    # _compute_unique_nodes: node_tensors is empty -> returns 0
    assert processed_batch.num_unique_nodes == 0

    # _compute_avg_degree: edge_src.numel() == 0 -> 0.0
    assert processed_batch.avg_degree == 0.0

    # _count_repeated_edge_events: edge_src.numel() == 0 -> 0
    assert processed_batch.num_repeated_edge_events == 0

    # _count_repeated_node_events: node_x_nids.numel() == 0 -> 0
    assert processed_batch.num_repeated_node_events == 0
