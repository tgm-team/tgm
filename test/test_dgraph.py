import pytest
import torch

from opendg.events import EdgeEvent, NodeEvent
from opendg.graph import DGraph
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
def test_init_empty(time_delta):
    dg = DGraph(data=[], time_delta=time_delta)
    if time_delta is None:
        assert dg.time_delta.is_ordered
    else:
        assert dg.time_delta == time_delta

    assert len(dg) == 0
    assert dg.start_time is None
    assert dg.end_time is None
    assert dg.num_nodes == 0
    assert dg.num_edges == 0
    assert dg.num_timestamps == 0
    assert dg.node_feats is None
    assert dg.edge_feats is None

    expected_edges = torch.Tensor([]), torch.Tensor([]), torch.Tensor([])
    _assert_edge_eq(dg.edges, expected_edges)


def test_init_from_events(events):
    dg = DGraph(events)

    assert dg.time_delta.is_ordered

    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edges == 3
    assert dg.num_timestamps == 4

    expected_edges = (
        torch.tensor([2, 2, 1], dtype=torch.int64),
        torch.tensor([2, 4, 8], dtype=torch.int64),
        torch.tensor([1, 5, 20], dtype=torch.int64),
    )
    _assert_edge_eq(dg.edges, expected_edges)

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


def test_init_from_storage(events):
    dg_tmp = DGraph(events)
    dg = DGraph(dg_tmp._storage)
    assert id(dg_tmp._storage) == id(dg._storage)


def test_to_events():
    events = [
        EdgeEvent(t=1, src=2, dst=2),
        EdgeEvent(t=5, src=2, dst=4),
        EdgeEvent(t=20, src=1, dst=8),
    ]
    dg = DGraph(events)
    assert dg.to_events() == events


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
    exp_features = {'node': dg.node_feats, 'edge': dg.edge_feats}
    exp = (exp_src, exp_dst, exp_t, exp_features)
    _assert_batch_eq(exp, dg.materialize)


def test_materialize_with_features(events):
    dg = DGraph(events)
    exp_src = torch.tensor([2, 2, 1], dtype=torch.int64)
    exp_dst = torch.tensor([2, 4, 8], dtype=torch.int64)
    exp_t = torch.tensor([1, 5, 20], dtype=torch.int64)
    exp_features = {'node': dg.node_feats.to_dense(), 'edge': dg.edge_feats.to_dense()}
    exp = (exp_src, exp_dst, exp_t, exp_features)
    _assert_batch_eq(exp, dg.materialize)


@pytest.mark.skip('Node feature IO not implemented')
def test_to_events_with_node_events():
    events = [
        NodeEvent(t=1, src=2),
        EdgeEvent(t=1, src=2, dst=2),
        NodeEvent(t=5, src=4),
        EdgeEvent(t=5, src=2, dst=4),
        NodeEvent(t=10, src=6),
        EdgeEvent(t=20, src=1, dst=8),
    ]
    dg = DGraph(events)
    assert dg.to_events() == events


def test_to_events_with_cache():
    events = [
        EdgeEvent(t=1, src=2, dst=2),
        EdgeEvent(t=5, src=2, dst=4),
        EdgeEvent(t=20, src=1, dst=8),
    ]
    dg = DGraph(events)
    dg._slice.start_time = 5
    assert dg.to_events() == events[1:]

    dg._slice.start_time = None  # reset
    dg._slice.node_slice = set([2])
    assert dg.to_events() == events[:2]


def test_to_events_with_features():
    events = [
        EdgeEvent(t=1, src=2, dst=2, features=torch.rand(2)),
        EdgeEvent(t=5, src=2, dst=2, features=torch.rand(2)),
        EdgeEvent(t=20, src=1, dst=8, features=torch.rand(2)),
    ]
    dg = DGraph(events)
    _assert_events_equal(dg.to_events(), events)


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

    expected_edges = (
        torch.tensor([2, 2, 1], dtype=torch.int64),
        torch.tensor([2, 4, 8], dtype=torch.int64),
        torch.tensor([1, 5, 20], dtype=torch.int64),
    )
    _assert_edge_eq(dg.edges, expected_edges)

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

    expected_edges = (
        torch.tensor([2, 2, 1], dtype=torch.int64),
        torch.tensor([2, 4, 8], dtype=torch.int64),
        torch.tensor([1, 5, 20], dtype=torch.int64),
    )
    _assert_edge_eq(dg.edges, expected_edges)

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

    expected_edges = (
        torch.tensor([2, 1], dtype=torch.int64),
        torch.tensor([4, 8], dtype=torch.int64),
        torch.tensor([5, 20], dtype=torch.int64),
    )
    _assert_edge_eq(dg1.edges, expected_edges)

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

    expected_edges = (
        torch.tensor([2], dtype=torch.int64),
        torch.tensor([2], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int64),
    )
    _assert_edge_eq(dg1.edges, expected_edges)

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

    expected_edges = (
        torch.tensor([2, 2, 1], dtype=torch.int64),
        torch.tensor([2, 4, 8], dtype=torch.int64),
        torch.tensor([1, 5, 20], dtype=torch.int64),
    )
    _assert_edge_eq(dg1.edges, expected_edges)

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

    expected_edges = (
        torch.tensor([2, 2], dtype=torch.int64),
        torch.tensor([2, 4], dtype=torch.int64),
        torch.tensor([1, 5], dtype=torch.int64),
    )
    _assert_edge_eq(dg1.edges, expected_edges)

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

    expected_edges = (
        torch.tensor([2, 2, 1], dtype=torch.int64),
        torch.tensor([2, 4, 8], dtype=torch.int64),
        torch.tensor([1, 5, 20], dtype=torch.int64),
    )
    _assert_edge_eq(dg.edges, expected_edges)

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

    expected_edges = (
        torch.tensor([2, 2], dtype=torch.int64),
        torch.tensor([2, 4], dtype=torch.int64),
        torch.tensor([1, 5], dtype=torch.int64),
    )
    _assert_edge_eq(dg1.edges, expected_edges)

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
    assert torch.equal(dg.node_feats.to_dense(), original_node_feats)
    assert torch.equal(dg.edge_feats.to_dense(), original_edge_feats)

    # Slice Number 2
    dg2 = dg1.slice_time(5, 14)
    assert id(dg2._storage) == id(dg._storage)

    assert len(dg2) == 2
    assert dg2.start_time == 5
    assert dg2.end_time == 14
    assert dg2.num_nodes == 7
    assert dg2.num_edges == 1
    assert dg2.num_timestamps == 2

    expected_edges = (
        torch.tensor([2], dtype=torch.int64),
        torch.tensor([4], dtype=torch.int64),
        torch.tensor([5], dtype=torch.int64),
    )
    _assert_edge_eq(dg2.edges, expected_edges)

    exp_node_feats = torch.zeros(dg2.end_time + 1, dg2.num_nodes, 5)
    exp_node_feats[5, 4] = events[2].features
    exp_node_feats[10, 6] = events[4].features
    assert torch.equal(dg2.node_feats.to_dense(), exp_node_feats)

    exp_edge_feats = torch.zeros(dg2.end_time + 1, dg2.num_nodes, dg2.num_nodes, 5)
    exp_edge_feats[5, 2, 4] = events[3].features
    assert torch.equal(dg2.edge_feats.to_dense(), exp_edge_feats)

    # Check original graph cache is not updated
    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edges == 3
    assert dg.num_timestamps == 4
    assert torch.equal(dg.node_feats.to_dense(), original_node_feats)
    assert torch.equal(dg.edge_feats.to_dense(), original_edge_feats)

    # Slice number 3
    dg3 = dg2.slice_time(7, 10)
    assert id(dg3._storage) == id(dg._storage)

    assert len(dg3) == 1
    assert dg3.start_time == 7
    assert dg3.end_time == 10
    assert dg3.num_nodes == 7
    assert dg3.num_edges == 0
    assert dg3.num_timestamps == 1

    expected_edges = torch.Tensor([]), torch.Tensor([]), torch.Tensor([])
    _assert_edge_eq(dg3.edges, expected_edges)

    exp_node_feats = torch.zeros(dg3.end_time + 1, dg3.num_nodes, 5)
    exp_node_feats[10, 6] = events[4].features
    assert torch.equal(dg3.node_feats.to_dense(), exp_node_feats)

    assert dg3.edge_feats is None

    # Check original graph cache is not updated
    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edges == 3
    assert dg.num_timestamps == 4
    assert torch.equal(dg.node_feats.to_dense(), original_node_feats)
    assert torch.equal(dg.edge_feats.to_dense(), original_edge_feats)

    # Slice number 4 (to empty)
    dg4 = dg3.slice_time(0, 7)
    assert id(dg4._storage) == id(dg._storage)

    assert len(dg4) == 0
    assert dg4.start_time == 7
    assert dg4.end_time == 7
    assert dg4.num_nodes == 0
    assert dg4.num_edges == 0
    assert dg4.num_timestamps == 0
    assert dg4.node_feats is None
    assert dg4.edge_feats is None

    expected_edges = torch.Tensor([]), torch.Tensor([]), torch.Tensor([])
    _assert_edge_eq(dg4.edges, expected_edges)

    # Check original graph cache is not updated
    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edges == 3
    assert dg.num_timestamps == 4
    assert torch.equal(dg.node_feats.to_dense(), original_node_feats)
    assert torch.equal(dg.edge_feats.to_dense(), original_edge_feats)


def test_slice_time_bad_args():
    dg = DGraph(data=[])
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

    expected_edges = (
        torch.tensor([2, 2, 1], dtype=torch.int64),
        torch.tensor([2, 4, 8], dtype=torch.int64),
        torch.tensor([1, 5, 20], dtype=torch.int64),
    )
    _assert_edge_eq(dg.edges, expected_edges)

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

    expected_edges = (
        torch.tensor([2, 2, 1], dtype=torch.int64),
        torch.tensor([2, 4, 8], dtype=torch.int64),
        torch.tensor([1, 5, 20], dtype=torch.int64),
    )
    _assert_edge_eq(dg1.edges, expected_edges)

    exp_node_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, 5)
    exp_node_feats[1, 2] = events[0].features
    exp_node_feats[5, 4] = events[2].features
    assert torch.equal(dg1.node_feats.to_dense(), exp_node_feats)

    exp_edge_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, dg1.num_nodes, 5)
    exp_edge_feats[1, 2, 2] = events[1].features
    exp_edge_feats[5, 2, 4] = events[3].features
    exp_edge_feats[20, 1, 8] = events[-1].features
    assert torch.equal(dg1.edge_feats.to_dense(), exp_edge_feats)

    # Check original graph cache is not updated
    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edges == 3
    assert dg.num_timestamps == 4
    assert torch.equal(dg.node_feats.to_dense(), original_node_feats)
    assert torch.equal(dg.edge_feats.to_dense(), original_edge_feats)

    # Slice Number 2
    dg2 = dg1.slice_nodes({1, 2, 4, 6})  # 6 should be gone since we sliced it away

    assert id(dg2._storage) == id(dg._storage)

    assert len(dg2) == 3
    assert dg2.start_time == 1
    assert dg2.end_time == 20
    assert dg2.num_nodes == 9
    assert dg2.num_edges == 3
    assert dg2.num_timestamps == 3

    expected_edges = (
        torch.tensor([2, 2, 1], dtype=torch.int64),
        torch.tensor([2, 4, 8], dtype=torch.int64),
        torch.tensor([1, 5, 20], dtype=torch.int64),
    )
    _assert_edge_eq(dg2.edges, expected_edges)

    exp_node_feats = torch.zeros(dg2.end_time + 1, dg2.num_nodes, 5)
    exp_node_feats[1, 2] = events[0].features
    exp_node_feats[5, 4] = events[2].features
    assert torch.equal(dg2.node_feats.to_dense(), exp_node_feats)

    exp_edge_feats = torch.zeros(dg2.end_time + 1, dg2.num_nodes, dg2.num_nodes, 5)
    exp_edge_feats[1, 2, 2] = events[1].features
    exp_edge_feats[5, 2, 4] = events[3].features
    exp_edge_feats[20, 1, 8] = events[-1].features
    assert torch.equal(dg2.edge_feats.to_dense(), exp_edge_feats)

    # Check original graph cache is not updated
    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edges == 3
    assert dg.num_timestamps == 4
    assert torch.equal(dg.node_feats.to_dense(), original_node_feats)
    assert torch.equal(dg.edge_feats.to_dense(), original_edge_feats)

    # Slice Number 3: Edge 4 -> 2 should cause node 2 to come back
    dg3 = dg2.slice_nodes({4})
    assert id(dg3._storage) == id(dg._storage)

    assert len(dg3) == 1
    assert dg3.start_time == 5
    assert dg3.end_time == 5
    assert dg3.num_nodes == 5
    assert dg3.num_edges == 1
    assert dg3.num_timestamps == 1

    expected_edges = (
        torch.tensor([2], dtype=torch.int64),
        torch.tensor([4], dtype=torch.int64),
        torch.tensor([5], dtype=torch.int64),
    )
    _assert_edge_eq(dg3.edges, expected_edges)

    exp_node_feats = torch.zeros(dg3.end_time + 1, dg3.num_nodes, 5)
    exp_node_feats[5, 4] = events[2].features
    assert torch.equal(dg3.node_feats.to_dense(), exp_node_feats)

    exp_edge_feats = torch.zeros(dg3.end_time + 1, dg3.num_nodes, dg3.num_nodes, 5)
    exp_edge_feats[5, 2, 4] = events[3].features
    assert torch.equal(dg3.edge_feats.to_dense(), exp_edge_feats)

    # Check original graph cache is not updated
    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edges == 3
    assert dg.num_timestamps == 4
    assert torch.equal(dg.node_feats.to_dense(), original_node_feats)
    assert torch.equal(dg.edge_feats.to_dense(), original_edge_feats)

    # Slice number 4 (to empty)
    dg4 = dg3.slice_nodes({5})  # Should be empty since 2 was previously sliced

    assert len(dg4) == 0
    assert dg4.start_time == 5
    assert dg4.end_time == 5
    assert dg4.num_nodes == 0
    assert dg4.num_edges == 0
    assert dg4.num_timestamps == 0
    assert dg4.node_feats is None
    assert dg4.edge_feats is None

    expected_edges = torch.Tensor([]), torch.Tensor([]), torch.Tensor([])
    _assert_edge_eq(dg4.edges, expected_edges)

    # Check original graph cache is not updated
    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edges == 3
    assert dg.num_timestamps == 4
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

    expected_edges = (
        torch.tensor([2, 2, 1], dtype=torch.int64),
        torch.tensor([2, 4, 8], dtype=torch.int64),
        torch.tensor([1, 5, 20], dtype=torch.int64),
    )
    _assert_edge_eq(dg1.edges, expected_edges)

    exp_node_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, 5)
    exp_node_feats[1, 2] = events[0].features
    exp_node_feats[5, 4] = events[2].features
    assert torch.equal(dg1.node_feats.to_dense(), exp_node_feats)

    exp_edge_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, dg1.num_nodes, 5)
    exp_edge_feats[1, 2, 2] = events[1].features
    exp_edge_feats[5, 2, 4] = events[3].features
    exp_edge_feats[20, 1, 8] = events[-1].features
    assert torch.equal(dg1.edge_feats.to_dense(), exp_edge_feats)

    # Check original graph cache is not updated
    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edges == 3
    assert dg.num_timestamps == 4
    assert torch.equal(dg.node_feats.to_dense(), original_node_feats)
    assert torch.equal(dg.edge_feats.to_dense(), original_edge_feats)

    # Slice Number 2
    dg2 = dg1.slice_time(1, 14)
    assert id(dg1._storage) == id(dg._storage)

    assert len(dg2) == 2
    assert dg2.start_time == 1
    assert dg2.end_time == 14
    assert dg2.num_nodes == 5
    assert dg2.num_edges == 2
    assert dg2.num_timestamps == 2

    expected_edges = (
        torch.tensor([2, 2], dtype=torch.int64),
        torch.tensor([2, 4], dtype=torch.int64),
        torch.tensor([1, 5], dtype=torch.int64),
    )
    _assert_edge_eq(dg2.edges, expected_edges)

    exp_node_feats = torch.zeros(dg2.end_time + 1, dg2.num_nodes, 5)
    exp_node_feats[1, 2] = events[0].features
    exp_node_feats[5, 4] = events[2].features
    assert torch.equal(dg2.node_feats.to_dense(), exp_node_feats)

    exp_edge_feats = torch.zeros(dg2.end_time + 1, dg2.num_nodes, dg2.num_nodes, 5)
    exp_edge_feats[1, 2, 2] = events[1].features
    exp_edge_feats[5, 2, 4] = events[3].features
    assert torch.equal(dg2.edge_feats.to_dense(), exp_edge_feats)

    # Check original graph cache is not updated
    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edges == 3
    assert dg.num_timestamps == 4
    assert torch.equal(dg.node_feats.to_dense(), original_node_feats)
    assert torch.equal(dg.edge_feats.to_dense(), original_edge_feats)

    # Slice Number 3
    dg3 = dg2.slice_nodes({1, 4})  # Node 1 at time 20 was previously sliced off
    assert id(dg3._storage) == id(dg._storage)
    assert len(dg3) == 1
    assert dg3.start_time == 5
    assert dg3.end_time == 5
    assert dg3.num_nodes == 5
    assert dg3.num_edges == 1
    assert dg3.num_timestamps == 1

    expected_edges = (
        torch.tensor([2], dtype=torch.int64),
        torch.tensor([4], dtype=torch.int64),
        torch.tensor([5], dtype=torch.int64),
    )
    _assert_edge_eq(dg3.edges, expected_edges)

    exp_node_feats = torch.zeros(dg3.end_time + 1, dg3.num_nodes, 5)
    exp_node_feats[5, 4] = events[2].features
    assert torch.equal(dg3.node_feats.to_dense(), exp_node_feats)

    exp_edge_feats = torch.zeros(dg3.end_time + 1, dg3.num_nodes, dg3.num_nodes, 5)
    exp_edge_feats[5, 2, 4] = events[3].features
    assert torch.equal(dg3.edge_feats.to_dense(), exp_edge_feats)

    # Check original graph cache is not updated
    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edges == 3
    assert dg.num_timestamps == 4
    assert torch.equal(dg.node_feats.to_dense(), original_node_feats)
    assert torch.equal(dg.edge_feats.to_dense(), original_edge_feats)

    # Slice Number 4
    dg4 = dg3.slice_time(0, 19)  # Ensure previous slice not changed
    assert id(dg4._storage) == id(dg._storage)

    assert len(dg4) == 1
    assert dg4.start_time == 5
    assert dg4.end_time == 5
    assert dg4.num_nodes == 5
    assert dg4.num_edges == 1
    assert dg4.num_timestamps == 1

    expected_edges = (
        torch.tensor([2], dtype=torch.int64),
        torch.tensor([4], dtype=torch.int64),
        torch.tensor([5], dtype=torch.int64),
    )
    _assert_edge_eq(dg3.edges, expected_edges)

    exp_node_feats = torch.zeros(dg4.end_time + 1, dg4.num_nodes, 5)
    exp_node_feats[5, 4] = events[2].features
    assert torch.equal(dg4.node_feats.to_dense(), exp_node_feats)

    exp_edge_feats = torch.zeros(dg4.end_time + 1, dg4.num_nodes, dg4.num_nodes, 5)
    exp_edge_feats[5, 2, 4] = events[3].features
    assert torch.equal(dg4.edge_feats.to_dense(), exp_edge_feats)

    # Check original graph cache is not updated
    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edges == 3
    assert dg.num_timestamps == 4
    assert torch.equal(dg.node_feats.to_dense(), original_node_feats)
    assert torch.equal(dg.edge_feats.to_dense(), original_edge_feats)

    # Slice Number 5
    dg5 = dg4.slice_nodes({2})
    assert id(dg5._storage) == id(dg._storage)

    assert len(dg5) == 1  # Broken: this should only add back the node 2
    assert dg5.start_time == 5
    assert dg4.end_time == 5
    assert dg4.num_nodes == 5
    assert dg4.num_edges == 1
    assert dg4.num_timestamps == 1

    expected_edges = (
        torch.tensor([2], dtype=torch.int64),
        torch.tensor([4], dtype=torch.int64),
        torch.tensor([5], dtype=torch.int64),
    )
    _assert_edge_eq(dg3.edges, expected_edges)

    exp_node_feats = torch.zeros(dg4.end_time + 1, dg4.num_nodes, 5)
    exp_node_feats[5, 4] = events[2].features
    assert torch.equal(dg4.node_feats.to_dense(), exp_node_feats)

    exp_edge_feats = torch.zeros(dg4.end_time + 1, dg4.num_nodes, dg4.num_nodes, 5)
    exp_edge_feats[5, 2, 4] = events[3].features
    assert torch.equal(dg4.edge_feats.to_dense(), exp_edge_feats)

    # Check original graph cache is not updated
    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edges == 3
    assert dg.num_timestamps == 4
    assert torch.equal(dg.node_feats.to_dense(), original_node_feats)
    assert torch.equal(dg.edge_feats.to_dense(), original_edge_feats)

    # Slice 6 (to empty)
    dg6 = dg5.slice_time(1, 4)
    assert id(dg6._storage) == id(dg._storage)

    assert len(dg6) == 0
    assert dg6.start_time == 5
    assert dg6.end_time is 4
    assert dg6.num_nodes == 0
    assert dg6.num_edges == 0
    assert dg6.num_timestamps == 0
    assert dg6.node_feats is None
    assert dg6.edge_feats is None

    expected_edges = torch.Tensor([]), torch.Tensor([]), torch.Tensor([])
    _assert_edge_eq(dg6.edges, expected_edges)

    # Check original graph cache is not updated
    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edges == 3
    assert dg.num_timestamps == 4
    assert torch.equal(dg.node_feats.to_dense(), original_node_feats)
    assert torch.equal(dg.edge_feats.to_dense(), original_edge_feats)


def _assert_events_equal(expected_events, actual_events):
    assert len(expected_events) == len(actual_events)
    for i in range(len(expected_events)):
        expected_event = expected_events[i]
        actual_event = actual_events[i]

        assert isinstance(expected_event, EdgeEvent)
        assert isinstance(actual_event, EdgeEvent)

        assert expected_event.t == actual_event.t
        assert expected_event.edge == actual_event.edge
        torch.testing.assert_close(expected_event.features, actual_event.features)


def _assert_edge_eq(actual, expected_edges):
    src_actual, dst_actual, t_actual = actual
    src_exp, dst_exp, t_exp = expected_edges

    def _eq_check(actual_tensor, exp_tensor):
        assert isinstance(actual_tensor, torch.Tensor)
        assert actual_tensor.tolist() == exp_tensor.tolist()

    _eq_check(src_actual, src_exp)
    _eq_check(dst_actual, dst_exp)
    _eq_check(t_actual, t_exp)


def _assert_batch_eq(actual, expected_edges):
    *actual_edge, actual_features = actual
    *exp_edge, exp_features = expected_edges
    _assert_edge_eq(actual_edge, exp_edge)
    assert isinstance(exp_features, dict)
    if exp_features['node'] is None:
        assert actual_features['node'] is None
    else:
        assert torch.equal(exp_features['node'], actual_features['node'])
    if exp_features['edge'] is None:
        assert actual_features['node'] is None
    else:
        assert torch.equal(exp_features['edge'], actual_features['edge'])
