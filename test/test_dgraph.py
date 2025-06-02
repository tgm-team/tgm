from dataclasses import asdict

import pytest
import torch

from opendg.events import EdgeEvent, NodeEvent
from opendg.graph import DGBatch, DGraph
from opendg.timedelta import TimeDeltaDG


@pytest.fixture
def events():
    return [
        NodeEvent(t=1, src=2, features=torch.rand(5)),
        EdgeEvent(t=1, src=2, dst=2, features=torch.rand(5)),
        NodeEvent(t=5, src=4, features=torch.rand(5)),
        EdgeEvent(t=5, src=2, dst=4, features=torch.rand(5)),
        NodeEvent(t=10, src=6, features=torch.rand(5)),
        EdgeEvent(t=20, src=1, dst=8, features=torch.rand(5)),
    ]


@pytest.mark.parametrize(
    'time_delta', [TimeDeltaDG('Y'), TimeDeltaDG('s'), TimeDeltaDG('r')]
)
def test_attempt_init_empty(time_delta):
    with pytest.raises(ValueError):
        DGraph(data=[], time_delta=time_delta)


def test_init_from_events(events):
    dg = DGraph(events)

    assert dg.time_delta.is_ordered

    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edges == 3
    assert dg.num_timestamps == 4
    assert dg.num_events == 6
    assert dg.nodes == {1, 2, 4, 6, 8}
    assert dg.node_feats_dim == 5
    assert dg.edge_feats_dim == 5

    expected_edges = (
        torch.tensor([2, 2, 1], dtype=torch.int64),
        torch.tensor([2, 4, 8], dtype=torch.int64),
        torch.tensor([1, 5, 20], dtype=torch.int64),
    )
    torch.testing.assert_close(dg.edges, expected_edges)

    exp_node_feats = torch.zeros(dg.end_time + 1, dg.num_nodes, 5)
    exp_node_feats[1, 2] = events[0].features
    exp_node_feats[5, 4] = events[2].features
    exp_node_feats[10, 6] = events[4].features
    torch.testing.assert_close(dg.node_feats.to_dense(), exp_node_feats)

    exp_edge_feats = torch.zeros(dg.end_time + 1, dg.num_nodes, dg.num_nodes, 5)
    exp_edge_feats[1, 2, 2] = events[1].features
    exp_edge_feats[5, 2, 4] = events[3].features
    exp_edge_feats[20, 1, 8] = events[-1].features
    torch.testing.assert_close(dg.edge_feats.to_dense(), exp_edge_feats)


def test_init_from_storage(events):
    dg_tmp = DGraph(events)
    dg = DGraph(dg_tmp._storage)
    assert id(dg_tmp._storage) == id(dg._storage)


def test_materialize():
    events = [
        EdgeEvent(t=1, src=2, dst=2),
        EdgeEvent(t=5, src=2, dst=4),
        EdgeEvent(t=20, src=1, dst=8),
    ]
    dg = DGraph(events)

    exp_src = torch.tensor([2, 2, 1], dtype=torch.int64)
    exp_dst = torch.tensor([2, 4, 8], dtype=torch.int64)
    exp_t = torch.tensor([1, 5, 20], dtype=torch.int64)
    exp = DGBatch(exp_src, exp_dst, exp_t, dg.node_feats, dg.edge_feats)
    torch.testing.assert_close(asdict(dg.materialize()), asdict(exp))


def test_materialize_with_features(events):
    dg = DGraph(events)
    exp_src = torch.tensor([2, 2, 1], dtype=torch.int64)
    exp_dst = torch.tensor([2, 4, 8], dtype=torch.int64)
    exp_t = torch.tensor([1, 5, 20], dtype=torch.int64)
    exp = DGBatch(
        exp_src, exp_dst, exp_t, dg.node_feats._values(), dg.edge_feats._values()
    )
    torch.testing.assert_close(asdict(dg.materialize()), asdict(exp))


def test_materialize_skip_feature_materialization(events):
    dg = DGraph(events)
    exp_src = torch.tensor([2, 2, 1], dtype=torch.int64)
    exp_dst = torch.tensor([2, 4, 8], dtype=torch.int64)
    exp_t = torch.tensor([1, 5, 20], dtype=torch.int64)
    exp = DGBatch(exp_src, exp_dst, exp_t, None, None)
    torch.testing.assert_close(
        asdict(dg.materialize(materialize_features=False)), asdict(exp)
    )


def test_slice_time_full_graph(events):
    dg = DGraph(events)

    dg1 = dg.slice_time(1, 21)
    assert id(dg1._storage) == id(dg._storage)

    assert len(dg1) == 4
    assert dg1.start_time == 1
    assert dg1.end_time == 20
    assert dg1.num_nodes == 9
    assert dg1.num_edges == 3
    assert dg1.num_timestamps == 4
    assert dg.nodes == {1, 2, 4, 6, 8}
    assert dg.node_feats_dim == 5
    assert dg.edge_feats_dim == 5

    exp_edges = (
        torch.LongTensor([2, 2, 1]),
        torch.LongTensor([2, 4, 8]),
        torch.LongTensor([1, 5, 20]),
    )
    torch.testing.assert_close(dg.edges, exp_edges)

    exp_node_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, 5)
    exp_node_feats[1, 2] = events[0].features
    exp_node_feats[5, 4] = events[2].features
    exp_node_feats[10, 6] = events[4].features
    assert torch.equal(dg1.node_feats.to_dense(), exp_node_feats)

    exp_edge_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, dg1.num_nodes, 5)
    exp_edge_feats[1, 2, 2] = events[1].features
    exp_edge_feats[5, 2, 4] = events[3].features
    exp_edge_feats[20, 1, 8] = events[-1].features
    assert torch.equal(dg1.edge_feats.to_dense(), exp_edge_feats)


def test_slice_time_no_time_bounds(events):
    dg = DGraph(events)

    dg1 = dg.slice_time()
    assert id(dg1._storage) == id(dg._storage)

    assert len(dg1) == 4
    assert dg1.start_time == 1
    assert dg1.end_time == 20
    assert dg1.num_nodes == 9
    assert dg1.num_edges == 3
    assert dg1.num_timestamps == 4
    assert dg.nodes == {1, 2, 4, 6, 8}
    assert dg.node_feats_dim == 5
    assert dg.edge_feats_dim == 5

    exp_edges = (
        torch.LongTensor([2, 2, 1]),
        torch.LongTensor([2, 4, 8]),
        torch.LongTensor([1, 5, 20]),
    )
    torch.testing.assert_close(dg.edges, exp_edges)

    exp_node_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, 5)
    exp_node_feats[1, 2] = events[0].features
    exp_node_feats[5, 4] = events[2].features
    exp_node_feats[10, 6] = events[4].features
    assert torch.equal(dg1.node_feats.to_dense(), exp_node_feats)

    exp_edge_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, dg1.num_nodes, 5)
    exp_edge_feats[1, 2, 2] = events[1].features
    exp_edge_feats[5, 2, 4] = events[3].features
    exp_edge_feats[20, 1, 8] = events[-1].features
    assert torch.equal(dg1.edge_feats.to_dense(), exp_edge_feats)


def test_slice_time_no_upper_bound(events):
    dg = DGraph(events)

    dg1 = dg.slice_time(5)
    assert id(dg1._storage) == id(dg._storage)

    assert len(dg1) == 3
    assert dg1.start_time == 5
    assert dg1.end_time == 20
    assert dg1.num_nodes == 9
    assert dg1.num_edges == 2
    assert dg1.num_timestamps == 3
    assert dg.nodes == {1, 2, 4, 6, 8}
    assert dg.node_feats_dim == 5
    assert dg.edge_feats_dim == 5

    exp_edges = (
        torch.LongTensor([2, 1]),
        torch.LongTensor([4, 8]),
        torch.LongTensor([5, 20]),
    )
    torch.testing.assert_close(dg1.edges, exp_edges)

    exp_node_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, 5)
    exp_node_feats[5, 4] = events[2].features
    exp_node_feats[10, 6] = events[4].features
    assert torch.equal(dg1.node_feats.to_dense(), exp_node_feats)

    exp_edge_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, dg1.num_nodes, 5)
    exp_edge_feats[5, 2, 4] = events[3].features
    exp_edge_feats[20, 1, 8] = events[-1].features
    assert torch.equal(dg1.edge_feats.to_dense(), exp_edge_feats)


def test_slice_time_no_lower_bound(events):
    dg = DGraph(events)

    dg1 = dg.slice_time(end_time=4)
    assert id(dg1._storage) == id(dg._storage)

    assert len(dg1) == 1
    assert dg1.start_time == 1
    assert dg1.end_time == 4
    assert dg1.num_nodes == 3
    assert dg1.num_edges == 1
    assert dg1.num_timestamps == 1
    assert dg.nodes == {1, 2, 4, 6, 8}
    assert dg.node_feats_dim == 5
    assert dg.edge_feats_dim == 5

    exp_edges = (
        torch.LongTensor([2]),
        torch.LongTensor([2]),
        torch.LongTensor([1]),
    )
    torch.testing.assert_close(dg1.edges, exp_edges)

    exp_node_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, 5)
    exp_node_feats[1, 2] = events[0].features
    assert torch.equal(dg1.node_feats.to_dense(), exp_node_feats)

    exp_edge_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, dg1.num_nodes, 5)
    exp_edge_feats[1, 2, 2] = events[1].features
    assert torch.equal(dg1.edge_feats.to_dense(), exp_edge_feats)


def test_slice_time_no_cache_refresh(events):
    dg = DGraph(events)

    dg1 = dg.slice_time(0, 100)
    assert id(dg1._storage) == id(dg._storage)

    assert len(dg1) == 4
    assert dg1.start_time == 1
    assert dg1.end_time == 20
    assert dg1.num_nodes == 9
    assert dg1.num_edges == 3
    assert dg1.num_timestamps == 4
    assert dg.nodes == {1, 2, 4, 6, 8}
    assert dg.node_feats_dim == 5
    assert dg.edge_feats_dim == 5

    exp_edges = (
        torch.LongTensor([2, 2, 1]),
        torch.LongTensor([2, 4, 8]),
        torch.LongTensor([1, 5, 20]),
    )
    torch.testing.assert_close(dg1.edges, exp_edges)

    exp_node_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, 5)
    exp_node_feats[1, 2] = events[0].features
    exp_node_feats[5, 4] = events[2].features
    exp_node_feats[10, 6] = events[4].features
    assert torch.equal(dg1.node_feats.to_dense(), exp_node_feats)

    exp_edge_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, dg1.num_nodes, 5)
    exp_edge_feats[1, 2, 2] = events[1].features
    exp_edge_feats[5, 2, 4] = events[3].features
    exp_edge_feats[20, 1, 8] = events[-1].features
    assert torch.equal(dg1.edge_feats.to_dense(), exp_edge_feats)


def test_slice_time_at_end_time(events):
    dg = DGraph(events)

    dg1 = dg.slice_time(1, 19)
    assert id(dg1._storage) == id(dg._storage)

    assert len(dg1) == 3
    assert dg1.start_time == 1
    assert dg1.end_time == 19  # Note: this is 19 despite no events in [10, 19)
    assert dg1.num_nodes == 7
    assert dg1.num_edges == 2
    assert dg1.num_timestamps == 3
    assert dg1.nodes == {2, 4, 6}

    exp_edges = (
        torch.LongTensor([2, 2]),
        torch.LongTensor([2, 4]),
        torch.LongTensor([1, 5]),
    )
    torch.testing.assert_close(dg1.edges, exp_edges)

    exp_node_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, 5)
    exp_node_feats[1, 2] = events[0].features
    exp_node_feats[5, 4] = events[2].features
    exp_node_feats[10, 6] = events[4].features
    assert torch.equal(dg1.node_feats.to_dense(), exp_node_feats)

    exp_edge_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, dg1.num_nodes, 5)
    exp_edge_feats[1, 2, 2] = events[1].features
    exp_edge_feats[5, 2, 4] = events[3].features
    assert torch.equal(dg1.edge_feats.to_dense(), exp_edge_feats)

    # Check original graph cache is not updated
    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edges == 3
    assert dg.num_timestamps == 4
    assert dg.num_events == 6
    assert dg.nodes == {1, 2, 4, 6, 8}

    exp_edges = (
        torch.LongTensor([2, 2, 1]),
        torch.LongTensor([2, 4, 8]),
        torch.LongTensor([1, 5, 20]),
    )
    torch.testing.assert_close(dg.edges, exp_edges)

    exp_node_feats = torch.zeros(dg.end_time + 1, dg.num_nodes, 5)
    exp_node_feats[1, 2] = events[0].features
    exp_node_feats[5, 4] = events[2].features
    exp_node_feats[10, 6] = events[4].features
    assert torch.equal(dg.node_feats.to_dense(), exp_node_feats)

    exp_edge_feats = torch.zeros(dg.end_time + 1, dg.num_nodes, dg.num_nodes, 5)
    exp_edge_feats[1, 2, 2] = events[1].features
    exp_edge_feats[5, 2, 4] = events[3].features
    exp_edge_feats[20, 1, 8] = events[-1].features
    assert torch.equal(dg.edge_feats.to_dense(), exp_edge_feats)


def test_slice_time_to_empty(events):
    dg = DGraph(events)

    original_node_feats = torch.zeros(dg.end_time + 1, dg.num_nodes, 5)
    original_node_feats[1, 2] = events[0].features
    original_node_feats[5, 4] = events[2].features
    original_node_feats[10, 6] = events[4].features

    original_edge_feats = torch.zeros(dg.end_time + 1, dg.num_nodes, dg.num_nodes, 5)
    original_edge_feats[1, 2, 2] = events[1].features
    original_edge_feats[5, 2, 4] = events[3].features
    original_edge_feats[20, 1, 8] = events[-1].features

    # Slice Number 1
    dg1 = dg.slice_time(1, 14)
    assert id(dg1._storage) == id(dg._storage)

    assert len(dg1) == 3
    assert dg1.start_time == 1
    assert dg1.end_time == 14
    assert dg1.num_nodes == 7
    assert dg1.num_edges == 2
    assert dg1.num_timestamps == 3
    assert dg1.nodes == {2, 4, 6}

    exp_edges = (
        torch.LongTensor([2, 2]),
        torch.LongTensor([2, 4]),
        torch.LongTensor([1, 5]),
    )
    torch.testing.assert_close(dg1.edges, exp_edges)

    exp_node_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, 5)
    exp_node_feats[1, 2] = events[0].features
    exp_node_feats[5, 4] = events[2].features
    exp_node_feats[10, 6] = events[4].features
    assert torch.equal(dg1.node_feats.to_dense(), exp_node_feats)

    exp_edge_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, dg1.num_nodes, 5)
    exp_edge_feats[1, 2, 2] = events[1].features
    exp_edge_feats[5, 2, 4] = events[3].features
    assert torch.equal(dg1.edge_feats.to_dense(), exp_edge_feats)

    # Slice Number 2
    dg2 = dg1.slice_time(5, 14)
    assert id(dg2._storage) == id(dg._storage)

    assert len(dg2) == 2
    assert dg2.start_time == 5
    assert dg2.end_time == 14
    assert dg2.num_nodes == 7
    assert dg2.num_edges == 1
    assert dg2.num_timestamps == 2
    assert dg2.nodes == {2, 4, 6}

    exp_edges = (
        torch.LongTensor([2]),
        torch.LongTensor([4]),
        torch.LongTensor([5]),
    )
    torch.testing.assert_close(dg2.edges, exp_edges)

    exp_node_feats = torch.zeros(dg2.end_time + 1, dg2.num_nodes, 5)
    exp_node_feats[5, 4] = events[2].features
    exp_node_feats[10, 6] = events[4].features
    assert torch.equal(dg2.node_feats.to_dense(), exp_node_feats)

    exp_edge_feats = torch.zeros(dg2.end_time + 1, dg2.num_nodes, dg2.num_nodes, 5)
    exp_edge_feats[5, 2, 4] = events[3].features
    assert torch.equal(dg2.edge_feats.to_dense(), exp_edge_feats)

    # Slice number 3
    dg3 = dg2.slice_time(7, 10)
    assert id(dg3._storage) == id(dg._storage)

    assert len(dg3) == 1
    assert dg3.start_time == 7
    assert dg3.end_time == 10
    assert dg3.num_nodes == 7
    assert dg3.num_edges == 0
    assert dg3.num_timestamps == 1
    assert dg3.nodes == {6}

    exp_edges = (torch.LongTensor([]), torch.LongTensor([]), torch.LongTensor([]))
    torch.testing.assert_close(dg3.edges, exp_edges)

    exp_node_feats = torch.zeros(dg3.end_time + 1, dg3.num_nodes, 5)
    exp_node_feats[10, 6] = events[4].features
    assert torch.equal(dg3.node_feats.to_dense(), exp_node_feats)

    assert dg3.edge_feats is None

    # Slice number 4 (to empty)
    dg4 = dg3.slice_time(0, 7)
    assert id(dg4._storage) == id(dg._storage)

    assert len(dg4) == 0
    assert dg4.start_time == 7
    assert dg4.end_time == 7
    assert dg4.num_nodes == 0
    assert dg4.num_edges == 0
    assert dg4.num_timestamps == 0
    assert dg4.nodes == set()
    assert dg4.node_feats is None
    assert dg4.edge_feats is None

    exp_edges = (torch.LongTensor([]), torch.LongTensor([]), torch.LongTensor([]))
    torch.testing.assert_close(dg4.edges, exp_edges)

    # Check original graph cache is not updated
    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edges == 3
    assert dg.num_timestamps == 4
    assert dg.num_events == 6
    assert dg.nodes == {1, 2, 4, 6, 8}
    assert torch.equal(dg.node_feats.to_dense(), original_node_feats)
    assert torch.equal(dg.edge_feats.to_dense(), original_edge_feats)


def test_slice_time_bad_args(events):
    dg = DGraph(data=events)
    with pytest.raises(ValueError):
        dg.slice_time(2, 1)


def test_slice_nodes_full_graph(events):
    dg1 = DGraph(events)

    dg = dg1.slice_nodes({1, 2, 3, 4, 6, 8})  # Extra node (3) should just be ignored
    assert id(dg._storage) == id(dg1._storage)

    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edges == 3
    assert dg.num_timestamps == 4
    assert dg.num_events == 6
    assert dg.nodes == {1, 2, 4, 6, 8}

    exp_edges = (
        torch.LongTensor([2, 2, 1]),
        torch.LongTensor([2, 4, 8]),
        torch.LongTensor([1, 5, 20]),
    )
    torch.testing.assert_close(dg.edges, exp_edges)

    exp_node_feats = torch.zeros(dg.end_time + 1, dg.num_nodes, 5)
    exp_node_feats[1, 2] = events[0].features
    exp_node_feats[5, 4] = events[2].features
    exp_node_feats[10, 6] = events[4].features
    assert torch.equal(dg.node_feats.to_dense(), exp_node_feats)

    exp_edge_feats = torch.zeros(dg.end_time + 1, dg.num_nodes, dg.num_nodes, 5)
    exp_edge_feats[1, 2, 2] = events[1].features
    exp_edge_feats[5, 2, 4] = events[3].features
    exp_edge_feats[20, 1, 8] = events[-1].features
    assert torch.equal(dg.edge_feats.to_dense(), exp_edge_feats)


def test_slice_nodes_to_empty(events):
    dg = DGraph(events)

    original_node_feats = torch.zeros(dg.end_time + 1, dg.num_nodes, 5)
    original_node_feats[1, 2] = events[0].features
    original_node_feats[5, 4] = events[2].features
    original_node_feats[10, 6] = events[4].features

    original_edge_feats = torch.zeros(dg.end_time + 1, dg.num_nodes, dg.num_nodes, 5)
    original_edge_feats[1, 2, 2] = events[1].features
    original_edge_feats[5, 2, 4] = events[3].features
    original_edge_feats[20, 1, 8] = events[-1].features

    # Slice Number 1
    dg1 = dg.slice_nodes({1, 2, 4, 8})
    assert id(dg1._storage) == id(dg._storage)

    assert len(dg1) == 3
    assert dg1.start_time == 1
    assert dg1.end_time == 20
    assert dg1.num_nodes == 9
    assert dg1.num_edges == 3
    assert dg1.num_timestamps == 3
    assert dg.nodes == {1, 2, 4, 6, 8}

    exp_edges = (
        torch.LongTensor([2, 2, 1]),
        torch.LongTensor([2, 4, 8]),
        torch.LongTensor([1, 5, 20]),
    )
    torch.testing.assert_close(dg1.edges, exp_edges)

    exp_node_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, 5)
    exp_node_feats[1, 2] = events[0].features
    exp_node_feats[5, 4] = events[2].features
    assert torch.equal(dg1.node_feats.to_dense(), exp_node_feats)

    exp_edge_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, dg1.num_nodes, 5)
    exp_edge_feats[1, 2, 2] = events[1].features
    exp_edge_feats[5, 2, 4] = events[3].features
    exp_edge_feats[20, 1, 8] = events[-1].features
    assert torch.equal(dg1.edge_feats.to_dense(), exp_edge_feats)

    # Slice Number 2
    dg2 = dg1.slice_nodes({1, 2, 4, 6})  # 6 should be gone since we sliced it away

    assert id(dg2._storage) == id(dg._storage)

    assert len(dg2) == 3
    assert dg2.start_time == 1
    assert dg2.end_time == 20
    assert dg2.num_nodes == 9
    assert dg2.num_edges == 3
    assert dg2.num_timestamps == 3
    assert dg2.nodes == {1, 2, 4, 8}

    exp_edges = (
        torch.LongTensor([2, 2, 1]),
        torch.LongTensor([2, 4, 8]),
        torch.LongTensor([1, 5, 20]),
    )
    torch.testing.assert_close(dg2.edges, exp_edges)

    exp_node_feats = torch.zeros(dg2.end_time + 1, dg2.num_nodes, 5)
    exp_node_feats[1, 2] = events[0].features
    exp_node_feats[5, 4] = events[2].features
    assert torch.equal(dg2.node_feats.to_dense(), exp_node_feats)

    exp_edge_feats = torch.zeros(dg2.end_time + 1, dg2.num_nodes, dg2.num_nodes, 5)
    exp_edge_feats[1, 2, 2] = events[1].features
    exp_edge_feats[5, 2, 4] = events[3].features
    exp_edge_feats[20, 1, 8] = events[-1].features
    assert torch.equal(dg2.edge_feats.to_dense(), exp_edge_feats)

    # Slice Number 3: Edge 4 -> 2 should cause node 2 to come back
    dg3 = dg2.slice_nodes({4})
    assert id(dg3._storage) == id(dg._storage)

    assert len(dg3) == 1
    assert dg3.start_time == 5
    assert dg3.end_time == 5
    assert dg3.num_nodes == 5
    assert dg3.num_edges == 1
    assert dg3.num_timestamps == 1
    assert dg3.nodes == {2, 4}

    exp_edges = (
        torch.LongTensor([2]),
        torch.LongTensor([4]),
        torch.LongTensor([5]),
    )
    torch.testing.assert_close(dg3.edges, exp_edges)

    exp_node_feats = torch.zeros(dg3.end_time + 1, dg3.num_nodes, 5)
    exp_node_feats[5, 4] = events[2].features
    assert torch.equal(dg3.node_feats.to_dense(), exp_node_feats)

    exp_edge_feats = torch.zeros(dg3.end_time + 1, dg3.num_nodes, dg3.num_nodes, 5)
    exp_edge_feats[5, 2, 4] = events[3].features
    assert torch.equal(dg3.edge_feats.to_dense(), exp_edge_feats)

    # Slice number 4 (to empty)
    dg4 = dg3.slice_nodes({5})  # Should be empty since 2 was previously sliced

    assert len(dg4) == 0
    assert dg4.start_time == 5
    assert dg4.end_time == 5
    assert dg4.num_nodes == 0
    assert dg4.num_edges == 0
    assert dg4.num_timestamps == 0
    assert dg4.nodes == set()
    assert dg4.node_feats is None
    assert dg4.edge_feats is None

    exp_edges = (torch.LongTensor([]), torch.LongTensor([]), torch.LongTensor([]))
    torch.testing.assert_close(dg4.edges, exp_edges)

    # Check original graph cache is not updated
    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edges == 3
    assert dg.num_timestamps == 4
    assert dg.num_events == 6
    assert dg.nodes == {1, 2, 4, 6, 8}
    assert torch.equal(dg.node_feats.to_dense(), original_node_feats)
    assert torch.equal(dg.edge_feats.to_dense(), original_edge_feats)


def test_interleave_slice_time_slice_nodes(events):
    dg = DGraph(events)

    original_node_feats = torch.zeros(dg.end_time + 1, dg.num_nodes, 5)
    original_node_feats[1, 2] = events[0].features
    original_node_feats[5, 4] = events[2].features
    original_node_feats[10, 6] = events[4].features

    original_edge_feats = torch.zeros(dg.end_time + 1, dg.num_nodes, dg.num_nodes, 5)
    original_edge_feats[1, 2, 2] = events[1].features
    original_edge_feats[5, 2, 4] = events[3].features
    original_edge_feats[20, 1, 8] = events[-1].features

    # Slice Number 1
    dg1 = dg.slice_nodes({1, 2, 4, 8})
    assert id(dg1._storage) == id(dg._storage)

    assert len(dg1) == 3
    assert dg1.start_time == 1
    assert dg1.end_time == 20
    assert dg1.num_nodes == 9
    assert dg1.num_edges == 3
    assert dg1.num_timestamps == 3
    assert dg1.nodes == {1, 2, 4, 8}

    exp_edges = (
        torch.LongTensor([2, 2, 1]),
        torch.LongTensor([2, 4, 8]),
        torch.LongTensor([1, 5, 20]),
    )
    torch.testing.assert_close(dg1.edges, exp_edges)

    exp_node_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, 5)
    exp_node_feats[1, 2] = events[0].features
    exp_node_feats[5, 4] = events[2].features
    assert torch.equal(dg1.node_feats.to_dense(), exp_node_feats)

    exp_edge_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, dg1.num_nodes, 5)
    exp_edge_feats[1, 2, 2] = events[1].features
    exp_edge_feats[5, 2, 4] = events[3].features
    exp_edge_feats[20, 1, 8] = events[-1].features
    assert torch.equal(dg1.edge_feats.to_dense(), exp_edge_feats)

    # Slice Number 2
    dg2 = dg1.slice_time(1, 14)
    assert id(dg1._storage) == id(dg._storage)

    assert len(dg2) == 2
    assert dg2.start_time == 1
    assert dg2.end_time == 14
    assert dg2.num_nodes == 5
    assert dg2.num_edges == 2
    assert dg2.num_timestamps == 2
    assert dg2.nodes == {2, 4}

    exp_edges = (
        torch.LongTensor([2, 2]),
        torch.LongTensor([2, 4]),
        torch.LongTensor([1, 5]),
    )
    torch.testing.assert_close(dg2.edges, exp_edges)

    exp_node_feats = torch.zeros(dg2.end_time + 1, dg2.num_nodes, 5)
    exp_node_feats[1, 2] = events[0].features
    exp_node_feats[5, 4] = events[2].features
    assert torch.equal(dg2.node_feats.to_dense(), exp_node_feats)

    exp_edge_feats = torch.zeros(dg2.end_time + 1, dg2.num_nodes, dg2.num_nodes, 5)
    exp_edge_feats[1, 2, 2] = events[1].features
    exp_edge_feats[5, 2, 4] = events[3].features
    assert torch.equal(dg2.edge_feats.to_dense(), exp_edge_feats)

    # Slice Number 3
    dg3 = dg2.slice_nodes({1, 4})  # Node 1 at time 20 was previously sliced off
    assert id(dg3._storage) == id(dg._storage)
    assert len(dg3) == 1
    assert dg3.start_time == 5
    assert dg3.end_time == 5
    assert dg3.num_nodes == 5
    assert dg3.num_edges == 1
    assert dg3.num_timestamps == 1
    assert dg3.nodes == {2, 4}

    exp_edges = (torch.LongTensor([2]), torch.LongTensor([4]), torch.LongTensor([5]))
    torch.testing.assert_close(dg3.edges, exp_edges)

    exp_node_feats = torch.zeros(dg3.end_time + 1, dg3.num_nodes, 5)
    exp_node_feats[5, 4] = events[2].features
    assert torch.equal(dg3.node_feats.to_dense(), exp_node_feats)

    exp_edge_feats = torch.zeros(dg3.end_time + 1, dg3.num_nodes, dg3.num_nodes, 5)
    exp_edge_feats[5, 2, 4] = events[3].features
    assert torch.equal(dg3.edge_feats.to_dense(), exp_edge_feats)

    # Slice Number 4
    dg4 = dg3.slice_time(0, 19)  # Ensure previous slice not changed
    assert id(dg4._storage) == id(dg._storage)

    assert len(dg4) == 1
    assert dg4.start_time == 5
    assert dg4.end_time == 5
    assert dg4.num_nodes == 5
    assert dg4.num_edges == 1
    assert dg4.num_timestamps == 1
    assert dg4.nodes == {2, 4}

    exp_edges = (torch.LongTensor([2]), torch.LongTensor([4]), torch.LongTensor([5]))
    torch.testing.assert_close(dg3.edges, exp_edges)

    exp_node_feats = torch.zeros(dg4.end_time + 1, dg4.num_nodes, 5)
    exp_node_feats[5, 4] = events[2].features
    assert torch.equal(dg4.node_feats.to_dense(), exp_node_feats)

    exp_edge_feats = torch.zeros(dg4.end_time + 1, dg4.num_nodes, dg4.num_nodes, 5)
    exp_edge_feats[5, 2, 4] = events[3].features
    assert torch.equal(dg4.edge_feats.to_dense(), exp_edge_feats)

    # Slice Number 5
    dg5 = dg4.slice_nodes({2})
    assert id(dg5._storage) == id(dg._storage)

    assert len(dg5) == 1  # Broken: this should only add back the node 2
    assert dg5.start_time == 5
    assert dg4.end_time == 5
    assert dg4.num_nodes == 5
    assert dg4.num_edges == 1
    assert dg4.num_timestamps == 1

    exp_edges = (torch.LongTensor([2]), torch.LongTensor([4]), torch.LongTensor([5]))
    torch.testing.assert_close(dg3.edges, exp_edges)

    exp_node_feats = torch.zeros(dg4.end_time + 1, dg4.num_nodes, 5)
    exp_node_feats[5, 4] = events[2].features
    assert torch.equal(dg4.node_feats.to_dense(), exp_node_feats)

    exp_edge_feats = torch.zeros(dg4.end_time + 1, dg4.num_nodes, dg4.num_nodes, 5)
    exp_edge_feats[5, 2, 4] = events[3].features
    assert torch.equal(dg4.edge_feats.to_dense(), exp_edge_feats)

    # Slice 6 (to empty)
    dg6 = dg5.slice_time(1, 4)
    assert id(dg6._storage) == id(dg._storage)

    assert len(dg6) == 0
    assert dg6.start_time == 5
    assert dg6.end_time is 4
    assert dg6.num_nodes == 0
    assert dg6.num_edges == 0
    assert dg6.num_timestamps == 0
    assert dg6.nodes == set()
    assert dg6.node_feats is None
    assert dg6.edge_feats is None

    exp_edges = (torch.LongTensor([]), torch.LongTensor([]), torch.LongTensor([]))
    torch.testing.assert_close(dg6.edges, exp_edges)

    # Check original graph cache is not updated
    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edges == 3
    assert dg.num_timestamps == 4
    assert dg.nodes == {1, 2, 4, 6, 8}
    assert torch.equal(dg.node_feats.to_dense(), original_node_feats)
    assert torch.equal(dg.edge_feats.to_dense(), original_edge_feats)


@pytest.mark.skip(reason='TODO: Add test for dg slice events!')
def test_slice_events():
    pass


def _assert_events_equal(expected_events, actual_events):
    expected = [asdict(e) for e in expected_events]
    actual = [asdict(e) for e in actual_events]
    torch.testing.assert_close(expected, actual)
