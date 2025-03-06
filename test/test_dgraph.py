import tempfile

import pytest
import torch

from opendg.events import EdgeEvent, NodeEvent
from opendg.graph import DGraph
from opendg.timedelta import TimeDeltaDG


@pytest.mark.parametrize(
    'time_delta', [TimeDeltaDG('Y'), TimeDeltaDG('s'), TimeDeltaDG('r'), None]
)
def test_init_empty(time_delta):
    dg = DGraph(time_delta=time_delta)
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


def test_init_from_events():
    events = [
        NodeEvent(time=1, node_id=2, features=torch.rand(2, 5)),
        EdgeEvent(time=1, edge=(2, 2), features=torch.rand(2, 5)),
        NodeEvent(time=5, node_id=4, features=torch.rand(2, 5)),
        EdgeEvent(time=5, edge=(2, 4), features=torch.rand(2, 5)),
        NodeEvent(time=10, node_id=6, features=torch.rand(2, 5)),
        EdgeEvent(time=20, edge=(1, 8), features=torch.rand(2, 5)),
    ]
    dg = DGraph(events=events)

    assert dg.time_delta.is_ordered

    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edges == 3
    assert dg.num_timestamps == 4

    expected_node_feats = torch.zeros(20 + 1, 8 + 1, 2, 5)
    expected_node_feats[1, 2] = events[0].features
    expected_node_feats[5, 4] = events[2].features
    expected_node_feats[10, 6] = events[4].features
    assert torch.equal(dg.node_feats.to_dense(), expected_node_feats)

    expected_edge_feats = torch.zeros(20 + 1, 8 + 1, 8 + 1, 2, 5)
    expected_edge_feats[1, 2, 2] = events[1].features
    expected_edge_feats[5, 2, 4] = events[3].features
    expected_edge_feats[20, 1, 8] = events[-1].features
    assert torch.equal(dg.edge_feats.to_dense(), expected_edge_feats)


def test_init_from_storage():
    events = [
        NodeEvent(time=1, node_id=2, features=torch.rand(2, 5)),
        EdgeEvent(time=1, edge=(2, 2), features=torch.rand(2, 5)),
        NodeEvent(time=5, node_id=4, features=torch.rand(2, 5)),
        EdgeEvent(time=5, edge=(2, 4), features=torch.rand(2, 5)),
        NodeEvent(time=10, node_id=6, features=torch.rand(2, 5)),
        EdgeEvent(time=20, edge=(1, 8), features=torch.rand(2, 5)),
    ]
    dg_tmp = DGraph(events=events)
    dg = DGraph(_storage=dg_tmp._storage)
    assert id(dg_tmp._storage) == id(dg._storage)


def test_init_bad_constructor_args():
    events = [
        NodeEvent(time=1, node_id=2, features=torch.rand(2, 5)),
        EdgeEvent(time=1, edge=(2, 2), features=torch.rand(2, 5)),
        NodeEvent(time=5, node_id=4, features=torch.rand(2, 5)),
        EdgeEvent(time=5, edge=(2, 4), features=torch.rand(2, 5)),
        NodeEvent(time=10, node_id=6, features=torch.rand(2, 5)),
        EdgeEvent(time=20, edge=(1, 8), features=torch.rand(2, 5)),
    ]
    dg_tmp = DGraph(events=events)

    with pytest.raises(ValueError):
        DGraph(events=events, _storage=dg_tmp._storage)


def test_to_csv():
    events = [
        EdgeEvent(time=1, edge=(2, 2)),
        EdgeEvent(time=5, edge=(2, 4)),
        EdgeEvent(time=20, edge=(1, 8)),
    ]
    dg = DGraph(events=events)

    col_names = {'src_col': 'src_id', 'dst_col': 'dst_id', 'time_col': 'time'}
    with tempfile.NamedTemporaryFile() as f:
        dg.to_csv(f.name, **col_names)
        dg_recovered = dg.from_csv(f.name, **col_names)

    _assert_events_equal(dg_recovered._storage.to_events(), dg._storage.to_events())


@pytest.mark.skip('Node feature IO not implemented')
def test_to_csv_with_node_events():
    events = [
        NodeEvent(time=1, node_id=2),
        EdgeEvent(time=1, edge=(2, 2)),
        NodeEvent(time=5, node_id=4),
        EdgeEvent(time=5, edge=(2, 4)),
        NodeEvent(time=10, node_id=6),
        EdgeEvent(time=20, edge=(1, 8)),
    ]
    dg = DGraph(events=events)

    col_names = {'src_col': 'src_id', 'dst_col': 'dst_id', 'time_col': 'time'}
    with tempfile.NamedTemporaryFile() as f:
        dg.to_csv(f.name, **col_names)
        dg_recovered = dg.from_csv(f.name, **col_names)

    _assert_events_equal(dg_recovered._storage.to_events(), events)


def test_to_csv_with_cache():
    events = [
        EdgeEvent(time=1, edge=(2, 2)),
        EdgeEvent(time=5, edge=(2, 4)),
        EdgeEvent(time=20, edge=(1, 8)),
    ]
    dg = DGraph(events=events)
    dg._cache['start_time'] = 5

    col_names = {'src_col': 'src_id', 'dst_col': 'dst_id', 'time_col': 'time'}
    with tempfile.NamedTemporaryFile() as f:
        dg.to_csv(f.name, **col_names)
        dg_recovered = dg.from_csv(f.name, **col_names)

    _assert_events_equal(dg_recovered._storage.to_events(), events[1:])

    dg._cache.clear()
    dg._cache['node_slice'] = set([2])

    col_names = {'src_col': 'src_id', 'dst_col': 'dst_id', 'time_col': 'time'}
    with tempfile.NamedTemporaryFile() as f:
        dg.to_csv(f.name, **col_names)
        dg_recovered = dg.from_csv(f.name, **col_names)

    _assert_events_equal(dg_recovered._storage.to_events(), events[:2])


def test_to_csv_with_features():
    events = [
        EdgeEvent(time=1, edge=(2, 2), features=torch.rand(2)),
        EdgeEvent(time=5, edge=(2, 4), features=torch.rand(2)),
        EdgeEvent(time=20, edge=(1, 8), features=torch.rand(2)),
    ]
    dg = DGraph(events=events)
    col_names = {
        'src_col': 'src_id',
        'dst_col': 'dst_id',
        'time_col': 'time',
        'edge_feature_col': ['foo', 'bar'],
    }

    with tempfile.NamedTemporaryFile() as f:
        dg.to_csv(f.name, **col_names)
        dg_recovered = dg.from_csv(f.name, **col_names)

    _assert_events_equal(dg_recovered._storage.to_events(), events)


def test_to_csv_with_multi_dim_features():
    events = [
        EdgeEvent(time=1, edge=(2, 2), features=torch.rand(2, 5)),
        EdgeEvent(time=5, edge=(2, 4), features=torch.rand(2, 5)),
        EdgeEvent(time=20, edge=(1, 8), features=torch.rand(2, 5)),
    ]
    dg = DGraph(events=events)
    col_names = {
        'src_col': 'src_id',
        'dst_col': 'dst_id',
        'time_col': 'time',
        'edge_feature_col': ['foo', 'bar'],
    }

    with tempfile.NamedTemporaryFile() as f:
        with pytest.raises(ValueError):
            dg.to_csv(f.name, **col_names)


def test_slice_time_full_graph():
    events = [
        NodeEvent(time=1, node_id=2, features=torch.rand(2, 5)),
        EdgeEvent(time=1, edge=(2, 2), features=torch.rand(2, 5)),
        NodeEvent(time=5, node_id=4, features=torch.rand(2, 5)),
        EdgeEvent(time=5, edge=(2, 4), features=torch.rand(2, 5)),
        NodeEvent(time=10, node_id=6, features=torch.rand(2, 5)),
        EdgeEvent(time=20, edge=(1, 8), features=torch.rand(2, 5)),
    ]
    dg1 = DGraph(events=events)

    dg = dg1.slice_time(1, 21)
    assert id(dg._storage) == id(dg1._storage)

    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edges == 3
    assert dg.num_timestamps == 4

    expected_node_feats = torch.zeros(20 + 1, 8 + 1, 2, 5)
    expected_node_feats[1, 2] = events[0].features
    expected_node_feats[5, 4] = events[2].features
    expected_node_feats[10, 6] = events[4].features
    assert torch.equal(dg.node_feats.to_dense(), expected_node_feats)

    expected_edge_feats = torch.zeros(20 + 1, 8 + 1, 8 + 1, 2, 5)
    expected_edge_feats[1, 2, 2] = events[1].features
    expected_edge_feats[5, 2, 4] = events[3].features
    expected_edge_feats[20, 1, 8] = events[-1].features
    assert torch.equal(dg.edge_feats.to_dense(), expected_edge_feats)


def test_slice_time_no_cache_refresh():
    events = [
        NodeEvent(time=1, node_id=2, features=torch.rand(2, 5)),
        EdgeEvent(time=1, edge=(2, 2), features=torch.rand(2, 5)),
        NodeEvent(time=5, node_id=4, features=torch.rand(2, 5)),
        EdgeEvent(time=5, edge=(2, 4), features=torch.rand(2, 5)),
        NodeEvent(time=10, node_id=6, features=torch.rand(2, 5)),
        EdgeEvent(time=20, edge=(1, 8), features=torch.rand(2, 5)),
    ]
    dg1 = DGraph(events=events)

    dg = dg1.slice_time(0, 100)
    assert id(dg._storage) == id(dg1._storage)

    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edges == 3
    assert dg.num_timestamps == 4

    expected_node_feats = torch.zeros(20 + 1, 8 + 1, 2, 5)
    expected_node_feats[1, 2] = events[0].features
    expected_node_feats[5, 4] = events[2].features
    expected_node_feats[10, 6] = events[4].features
    assert torch.equal(dg.node_feats.to_dense(), expected_node_feats)

    expected_edge_feats = torch.zeros(20 + 1, 8 + 1, 8 + 1, 2, 5)
    expected_edge_feats[1, 2, 2] = events[1].features
    expected_edge_feats[5, 2, 4] = events[3].features
    expected_edge_feats[20, 1, 8] = events[-1].features
    assert torch.equal(dg.edge_feats.to_dense(), expected_edge_feats)


def test_slice_time_at_end_time():
    events = [
        NodeEvent(time=1, node_id=2, features=torch.rand(2, 5)),
        EdgeEvent(time=1, edge=(2, 2), features=torch.rand(2, 5)),
        NodeEvent(time=5, node_id=4, features=torch.rand(2, 5)),
        EdgeEvent(time=5, edge=(2, 4), features=torch.rand(2, 5)),
        NodeEvent(time=10, node_id=6, features=torch.rand(2, 5)),
        EdgeEvent(time=20, edge=(1, 8), features=torch.rand(2, 5)),
    ]
    dg1 = DGraph(events=events)

    dg = dg1.slice_time(1, 20)
    assert id(dg._storage) == id(dg1._storage)

    assert len(dg) == 3
    assert dg.start_time == 1
    assert dg.end_time == 19  # Note: this is 19 despite no events in [10, 19)
    assert dg.num_nodes == 7
    assert dg.num_edges == 2
    assert dg.num_timestamps == 3

    expected_node_feats = torch.zeros(19 + 1, 6 + 1, 2, 5)
    expected_node_feats[1, 2] = events[0].features
    expected_node_feats[5, 4] = events[2].features
    expected_node_feats[10, 6] = events[4].features
    assert torch.equal(dg.node_feats.to_dense(), expected_node_feats)

    expected_edge_feats = torch.zeros(19 + 1, 6 + 1, 6 + 1, 2, 5)
    expected_edge_feats[1, 2, 2] = events[1].features
    expected_edge_feats[5, 2, 4] = events[3].features
    assert torch.equal(dg.edge_feats.to_dense(), expected_edge_feats)

    # Check original graph cache is not updated
    assert len(dg1) == 4
    assert dg1.start_time == 1
    assert dg1.end_time == 20
    assert dg1.num_nodes == 9
    assert dg1.num_edges == 3
    assert dg1.num_timestamps == 4

    expected_node_feats = torch.zeros(20 + 1, 8 + 1, 2, 5)
    expected_node_feats[1, 2] = events[0].features
    expected_node_feats[5, 4] = events[2].features
    expected_node_feats[10, 6] = events[4].features
    assert torch.equal(dg1.node_feats.to_dense(), expected_node_feats)

    expected_edge_feats = torch.zeros(20 + 1, 8 + 1, 8 + 1, 2, 5)
    expected_edge_feats[1, 2, 2] = events[1].features
    expected_edge_feats[5, 2, 4] = events[3].features
    expected_edge_feats[20, 1, 8] = events[-1].features
    assert torch.equal(dg1.edge_feats.to_dense(), expected_edge_feats)


def test_slice_time_to_empty():
    events = [
        NodeEvent(time=1, node_id=2, features=torch.rand(2, 5)),
        EdgeEvent(time=1, edge=(2, 2), features=torch.rand(2, 5)),
        NodeEvent(time=5, node_id=4, features=torch.rand(2, 5)),
        EdgeEvent(time=5, edge=(2, 4), features=torch.rand(2, 5)),
        NodeEvent(time=10, node_id=6, features=torch.rand(2, 5)),
        EdgeEvent(time=20, edge=(1, 8), features=torch.rand(2, 5)),
    ]
    dg = DGraph(events=events)

    # Slice Number 1
    dg1 = dg.slice_time(1, 15)
    assert id(dg1._storage) == id(dg._storage)

    assert len(dg1) == 3
    assert dg1.start_time == 1
    assert dg1.end_time == 14
    assert dg1.num_nodes == 7
    assert dg1.num_edges == 2
    assert dg1.num_timestamps == 3

    expected_node_feats = torch.zeros(14 + 1, 6 + 1, 2, 5)
    expected_node_feats[1, 2] = events[0].features
    expected_node_feats[5, 4] = events[2].features
    expected_node_feats[10, 6] = events[4].features
    assert torch.equal(dg1.node_feats.to_dense(), expected_node_feats)

    expected_edge_feats = torch.zeros(14 + 1, 6 + 1, 6 + 1, 2, 5)
    expected_edge_feats[1, 2, 2] = events[1].features
    expected_edge_feats[5, 2, 4] = events[3].features
    assert torch.equal(dg1.edge_feats.to_dense(), expected_edge_feats)

    # Check original graph cache is not updated
    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edges == 3
    assert dg.num_timestamps == 4

    expected_node_feats = torch.zeros(20 + 1, 8 + 1, 2, 5)
    expected_node_feats[1, 2] = events[0].features
    expected_node_feats[5, 4] = events[2].features
    expected_node_feats[10, 6] = events[4].features
    assert torch.equal(dg.node_feats.to_dense(), expected_node_feats)

    expected_edge_feats = torch.zeros(20 + 1, 8 + 1, 8 + 1, 2, 5)
    expected_edge_feats[1, 2, 2] = events[1].features
    expected_edge_feats[5, 2, 4] = events[3].features
    expected_edge_feats[20, 1, 8] = events[-1].features
    assert torch.equal(dg.edge_feats.to_dense(), expected_edge_feats)

    # Slice Number 2
    dg2 = dg1.slice_time(5, 15)
    assert id(dg2._storage) == id(dg._storage)

    assert len(dg2) == 2
    assert dg2.start_time == 5
    assert dg2.end_time == 14
    assert dg2.num_nodes == 7
    assert dg2.num_edges == 1
    assert dg2.num_timestamps == 2

    expected_node_feats = torch.zeros(14 + 1, 6 + 1, 2, 5)
    expected_node_feats[5, 4] = events[2].features
    expected_node_feats[10, 6] = events[4].features
    assert torch.equal(dg2.node_feats.to_dense(), expected_node_feats)

    expected_edge_feats = torch.zeros(14 + 1, 6 + 1, 6 + 1, 2, 5)
    expected_edge_feats[5, 2, 4] = events[3].features
    assert torch.equal(dg2.edge_feats.to_dense(), expected_edge_feats)

    # Check original graph cache is not updated
    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edges == 3
    assert dg.num_timestamps == 4

    expected_node_feats = torch.zeros(20 + 1, 8 + 1, 2, 5)
    expected_node_feats[1, 2] = events[0].features
    expected_node_feats[5, 4] = events[2].features
    expected_node_feats[10, 6] = events[4].features
    assert torch.equal(dg.node_feats.to_dense(), expected_node_feats)

    expected_edge_feats = torch.zeros(20 + 1, 8 + 1, 8 + 1, 2, 5)
    expected_edge_feats[1, 2, 2] = events[1].features
    expected_edge_feats[5, 2, 4] = events[3].features
    expected_edge_feats[20, 1, 8] = events[-1].features
    assert torch.equal(dg.edge_feats.to_dense(), expected_edge_feats)

    # Slice number 3
    dg3 = dg2.slice_time(7, 11)
    assert id(dg3._storage) == id(dg._storage)

    assert len(dg3) == 1
    assert dg3.start_time == 7
    assert dg3.end_time == 10
    assert dg3.num_nodes == 7
    assert dg3.num_edges == 0
    assert dg3.num_timestamps == 1

    expected_node_feats = torch.zeros(10 + 1, 6 + 1, 2, 5)
    expected_node_feats[10, 6] = events[4].features
    assert torch.equal(dg3.node_feats.to_dense(), expected_node_feats)

    assert dg3.edge_feats is None

    # Check original graph cache is not updated
    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edges == 3
    assert dg.num_timestamps == 4

    expected_node_feats = torch.zeros(20 + 1, 8 + 1, 2, 5)
    expected_node_feats[1, 2] = events[0].features
    expected_node_feats[5, 4] = events[2].features
    expected_node_feats[10, 6] = events[4].features
    assert torch.equal(dg.node_feats.to_dense(), expected_node_feats)

    expected_edge_feats = torch.zeros(20 + 1, 8 + 1, 8 + 1, 2, 5)
    expected_edge_feats[1, 2, 2] = events[1].features
    expected_edge_feats[5, 2, 4] = events[3].features
    expected_edge_feats[20, 1, 8] = events[-1].features
    assert torch.equal(dg.edge_feats.to_dense(), expected_edge_feats)

    # Slice number 4 (to empty)
    dg4 = dg3.slice_time(0, 8)
    assert id(dg4._storage) == id(dg._storage)

    assert len(dg4) == 0
    assert dg4.start_time == 7
    assert dg4.end_time == 7
    assert dg4.num_nodes == 0
    assert dg4.num_edges == 0
    assert dg4.num_timestamps == 0
    assert dg4.node_feats is None
    assert dg4.edge_feats is None

    # Check original graph cache is not updated
    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edges == 3
    assert dg.num_timestamps == 4

    expected_node_feats = torch.zeros(20 + 1, 8 + 1, 2, 5)
    expected_node_feats[1, 2] = events[0].features
    expected_node_feats[5, 4] = events[2].features
    expected_node_feats[10, 6] = events[4].features
    assert torch.equal(dg.node_feats.to_dense(), expected_node_feats)

    expected_edge_feats = torch.zeros(20 + 1, 8 + 1, 8 + 1, 2, 5)
    expected_edge_feats[1, 2, 2] = events[1].features
    expected_edge_feats[5, 2, 4] = events[3].features
    expected_edge_feats[20, 1, 8] = events[-1].features
    assert torch.equal(dg.edge_feats.to_dense(), expected_edge_feats)


def test_slice_time_bad_args():
    dg = DGraph()
    with pytest.raises(ValueError):
        dg.slice_time(2, 1)


@pytest.mark.skip('TODO')
def test_slice_nodes():
    pass


@pytest.mark.skip('TODO')
def test_slice_nodes_to_empty():
    pass


@pytest.mark.skip('TODO')
def test_interleave_slice_time_slice_nodes():
    pass


def test_append_single_event():
    dg = DGraph()

    event = NodeEvent(time=1, node_id=2, features=torch.rand(2, 5))

    dg.append(event)

    assert len(dg) == 1
    assert dg.start_time == 1
    assert dg.end_time == 1
    assert dg.num_nodes == 3
    assert dg.num_edges == 0
    assert dg.num_timestamps == 1

    expected_node_feats = torch.zeros(1 + 1, 2 + 1, 2, 5)
    expected_node_feats[1, 2] = event.features
    assert torch.equal(dg.node_feats.to_dense(), expected_node_feats)

    assert dg.edge_feats is None


def test_append_multiple_events():
    dg = DGraph()

    events = [
        NodeEvent(time=1, node_id=2, features=torch.rand(2, 5)),
        EdgeEvent(time=1, edge=(2, 2), features=torch.rand(2, 5)),
        NodeEvent(time=5, node_id=4, features=torch.rand(2, 5)),
        EdgeEvent(time=5, edge=(2, 4), features=torch.rand(2, 5)),
        NodeEvent(time=10, node_id=6, features=torch.rand(2, 5)),
        EdgeEvent(time=20, edge=(1, 8), features=torch.rand(2, 5)),
    ]

    dg.append(events)

    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edges == 3
    assert dg.num_timestamps == 4

    expected_node_feats = torch.zeros(20 + 1, 8 + 1, 2, 5)
    expected_node_feats[1, 2] = events[0].features
    expected_node_feats[5, 4] = events[2].features
    expected_node_feats[10, 6] = events[4].features
    assert torch.equal(dg.node_feats.to_dense(), expected_node_feats)

    expected_edge_feats = torch.zeros(20 + 1, 8 + 1, 8 + 1, 2, 5)
    expected_edge_feats[1, 2, 2] = events[1].features
    expected_edge_feats[5, 2, 4] = events[3].features
    expected_edge_feats[20, 1, 8] = events[-1].features
    assert torch.equal(dg.edge_feats.to_dense(), expected_edge_feats)


@pytest.mark.skip('Append on view not implemented')
def test_slice_then_append():
    pass


@pytest.mark.skip('Append on view not implemented')
def test_append_then_slice():
    pass


@pytest.mark.skip('Append on view not implemented')
def test_slice_then_append_twice_then_slice():
    pass


def test_append_bad_args():
    events = [
        NodeEvent(time=1, node_id=2, features=torch.rand(2, 5)),
        EdgeEvent(time=1, edge=(2, 2), features=torch.rand(2, 5)),
        NodeEvent(time=5, node_id=4, features=torch.rand(2, 5)),
        EdgeEvent(time=5, edge=(2, 4), features=torch.rand(2, 5)),
        NodeEvent(time=10, node_id=6, features=torch.rand(2, 5)),
        EdgeEvent(time=20, edge=(1, 8), features=torch.rand(2, 5)),
    ]
    dg = DGraph(events=events)

    new_event = EdgeEvent(time=15, edge=(1, 1))  # Time 15 < 20
    with pytest.raises(ValueError):
        dg.append(new_event)

    # First event is still invalid despite the second one being valid
    new_events = [new_event, EdgeEvent(time=25, edge=(1, 1))]
    with pytest.raises(ValueError):
        dg.append(new_events)


def test_append_empty():
    dg = DGraph()
    dg.append([])

    assert len(dg) == 0
    assert dg.start_time is None
    assert dg.end_time is None
    assert dg.num_nodes == 0
    assert dg.num_edges == 0
    assert dg.num_timestamps == 0
    assert dg.node_feats is None
    assert dg.edge_feats is None


@pytest.mark.skip('Temporal Coarsening not implemented')
def test_temporal_coarsening_with_cache_sum_no_features():
    pass


@pytest.mark.skip('Temporal Coarsening not implemented')
def test_temporal_coarsening_sum_with_features():
    pass


@pytest.mark.skip('Temporal Coarsening not implemented')
def test_temporal_coarsening_with_cache_sum_with_features():
    pass


@pytest.mark.skip('Temporal Coarsening not implemented')
def test_temporal_coarsening_with_cache_concat_no_features():
    pass


@pytest.mark.skip('Temporal Coarsening not implemented')
def test_temporal_coarsening_concat_with_features():
    pass


@pytest.mark.skip('Temporal Coarsening not implemented')
def test_temporal_coarsening_concat_with_cache_sum_with_features():
    pass


@pytest.mark.skip('Temporal Coarsening not implemented')
def test_temporal_coarsening_empty_graph():
    pass


@pytest.mark.skip('Temporal Coarsening not implemented')
def test_temporal_coarsening_bad_time_delta():
    pass


@pytest.mark.skip('Temporal Coarsening not implemented')
def test_temporal_coarsening_bad_agg_func():
    pass


@pytest.mark.skip('Temporal Coarsening not implemented')
def test_temporal_coarsening_causes_time_gap():
    pass


def _assert_events_equal(expected_events, actual_events):
    assert len(expected_events) == len(actual_events)
    for i in range(len(expected_events)):
        expected_event = expected_events[i]
        actual_event = actual_events[i]

        assert isinstance(expected_event, EdgeEvent)
        assert isinstance(actual_event, EdgeEvent)

        assert expected_event.time == actual_event.time
        assert expected_event.edge == actual_event.edge
        torch.testing.assert_close(expected_event.features, actual_event.features)
