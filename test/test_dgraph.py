from dataclasses import asdict
from unittest.mock import patch

import pytest
import torch

from tgm.data import DGData
from tgm.graph import DGBatch, DGraph


@pytest.fixture
def data():
    edge_index = torch.LongTensor([[2, 2], [2, 4], [1, 8]])
    edge_timestamps = torch.LongTensor([1, 5, 20])
    edge_feats = torch.rand(3, 5)
    node_timestamps = torch.LongTensor([1, 5, 10])
    node_ids = torch.LongTensor([2, 4, 6])
    dynamic_node_feats = torch.rand([3, 5])
    static_node_feats = torch.rand(9, 11)
    return DGData.from_raw(
        edge_timestamps,
        edge_index,
        edge_feats,
        node_timestamps,
        node_ids,
        dynamic_node_feats,
        static_node_feats,
    )


def test_init_from_data(data):
    dg = DGraph(data)

    assert dg.time_delta.is_ordered

    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edges == 3
    assert dg.num_timestamps == 4
    assert dg.num_events == 6
    assert dg.nodes == {1, 2, 4, 6, 8}
    assert dg.static_node_feats_dim == 11
    assert dg.dynamic_node_feats_dim == 5
    assert dg.edge_feats_dim == 5
    assert dg.device == torch.device('cpu')

    expected_edges = (
        torch.tensor([2, 2, 1], dtype=torch.int64),
        torch.tensor([2, 4, 8], dtype=torch.int64),
        torch.tensor([1, 5, 20], dtype=torch.int64),
    )
    torch.testing.assert_close(dg.edges, expected_edges)

    exp_dynamic_node_feats = torch.zeros(dg.end_time + 1, dg.num_nodes, 5)
    exp_dynamic_node_feats[1, 2] = data.dynamic_node_feats[0]
    exp_dynamic_node_feats[5, 4] = data.dynamic_node_feats[1]
    exp_dynamic_node_feats[10, 6] = data.dynamic_node_feats[2]
    torch.testing.assert_close(dg.dynamic_node_feats.to_dense(), exp_dynamic_node_feats)

    exp_edge_feats = torch.zeros(dg.end_time + 1, dg.num_nodes, dg.num_nodes, 5)
    exp_edge_feats[1, 2, 2] = data.edge_feats[0]
    exp_edge_feats[5, 2, 4] = data.edge_feats[1]
    exp_edge_feats[20, 1, 8] = data.edge_feats[2]
    torch.testing.assert_close(dg.edge_feats.to_dense(), exp_edge_feats)


@pytest.mark.gpu
def test_init_gpu(data):
    dg = DGraph(data, device='cuda')

    assert dg.device == torch.device('cuda')

    expected_edges = (
        torch.tensor([2, 2, 1], dtype=torch.int64, device='cuda'),
        torch.tensor([2, 4, 8], dtype=torch.int64, device='cuda'),
        torch.tensor([1, 5, 20], dtype=torch.int64, device='cuda'),
    )
    torch.testing.assert_close(dg.edges, expected_edges)

    exp_dynamic_node_feats = torch.zeros(dg.end_time + 1, dg.num_nodes, 5)
    exp_dynamic_node_feats[1, 2] = data.dynamic_node_feats[0]
    exp_dynamic_node_feats[5, 4] = data.dynamic_node_feats[1]
    exp_dynamic_node_feats[10, 6] = data.dynamic_node_feats[2]
    exp_dynamic_node_feats = exp_dynamic_node_feats.cuda()
    torch.testing.assert_close(dg.dynamic_node_feats.to_dense(), exp_dynamic_node_feats)

    exp_edge_feats = torch.zeros(dg.end_time + 1, dg.num_nodes, dg.num_nodes, 5)
    exp_edge_feats[1, 2, 2] = data.edge_feats[0]
    exp_edge_feats[5, 2, 4] = data.edge_feats[1]
    exp_edge_feats[20, 1, 8] = data.edge_feats[2]
    exp_edge_feats = exp_edge_feats.cuda()
    torch.testing.assert_close(dg.edge_feats.to_dense(), exp_edge_feats)


def test_init_from_storage(data):
    dg_tmp = DGraph(data)
    dg = DGraph(dg_tmp._storage)
    assert id(dg_tmp._storage) == id(dg._storage)


def test_init_bad_args(data):
    with pytest.raises(ValueError):
        _ = DGraph(data, time_delta='r')


def test_init_construct_data():
    data = 'foo.csv'
    with patch.object(DGData, 'from_any') as mock:
        _ = DGraph(data)
        mock.assert_called_once_with(data)


def test_init_get_tgb_time_delta(data):
    _data = 'tgbl-foo'
    mock_td = {'tgbl-foo': 'bar'}
    with (
        patch.dict('tgm.timedelta.TGB_TIME_DELTAS', mock_td),
        patch.object(DGData, 'from_any', return_value=data),
    ):
        dg = DGraph(_data)
        assert dg.time_delta == 'bar'


def test_str(data):
    dg = DGraph(data)
    assert isinstance(dg.__str__(), str)


def test_materialize(data):
    dg = DGraph(data)
    exp_src = torch.tensor([2, 2, 1], dtype=torch.int64)
    exp_dst = torch.tensor([2, 4, 8], dtype=torch.int64)
    exp_t = torch.tensor([1, 5, 20], dtype=torch.int64)
    exp = DGBatch(
        exp_src,
        exp_dst,
        exp_t,
        dg.dynamic_node_feats._values(),
        dg.edge_feats._values(),
        dg.dynamic_node_feats._indices()[0],
        dg.dynamic_node_feats._indices()[1],
    )
    torch.testing.assert_close(asdict(dg.materialize()), asdict(exp))


def test_materialize_skip_feature_materialization(data):
    dg = DGraph(data)
    exp_src = torch.tensor([2, 2, 1], dtype=torch.int64)
    exp_dst = torch.tensor([2, 4, 8], dtype=torch.int64)
    exp_t = torch.tensor([1, 5, 20], dtype=torch.int64)
    exp = DGBatch(exp_src, exp_dst, exp_t, None, None)
    torch.testing.assert_close(
        asdict(dg.materialize(materialize_features=False)), asdict(exp)
    )


def test_slice_time_no_time_bounds(data):
    dg = DGraph(data)
    dg1 = dg.slice_time()
    assert id(dg1._storage) == id(dg._storage)
    assert dg1._slice == dg._slice


def test_slice_time_no_upper_bound(data):
    dg = DGraph(data)

    dg1 = dg.slice_time(5)
    assert id(dg1._storage) == id(dg._storage)

    assert len(dg1) == 3
    assert dg1.start_time == 5
    assert dg1.end_time == 20
    assert dg1.num_nodes == 9
    assert dg1.num_edges == 2
    assert dg1.num_timestamps == 3
    assert dg.nodes == {1, 2, 4, 6, 8}

    exp_edges = (
        torch.LongTensor([2, 1]),
        torch.LongTensor([4, 8]),
        torch.LongTensor([5, 20]),
    )
    torch.testing.assert_close(dg1.edges, exp_edges)

    exp_dynamic_node_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, 5)
    exp_dynamic_node_feats[5, 4] = data.dynamic_node_feats[1]
    exp_dynamic_node_feats[10, 6] = data.dynamic_node_feats[2]
    torch.testing.assert_close(
        dg1.dynamic_node_feats.to_dense(), exp_dynamic_node_feats
    )

    exp_edge_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, dg1.num_nodes, 5)
    exp_edge_feats[5, 2, 4] = data.edge_feats[1]
    exp_edge_feats[20, 1, 8] = data.edge_feats[2]
    torch.testing.assert_close(dg1.edge_feats.to_dense(), exp_edge_feats)
    torch.testing.assert_close(dg.static_node_feats, dg1.static_node_feats)


def test_slice_time_at_end_time(data):
    dg = DGraph(data)

    dg1 = dg.slice_time(1, 20)
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

    exp_dynamic_node_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, 5)
    exp_dynamic_node_feats[1, 2] = data.dynamic_node_feats[0]
    exp_dynamic_node_feats[5, 4] = data.dynamic_node_feats[1]
    exp_dynamic_node_feats[10, 6] = data.dynamic_node_feats[2]
    assert torch.equal(dg1.dynamic_node_feats.to_dense(), exp_dynamic_node_feats)

    exp_edge_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, dg1.num_nodes, 5)
    exp_edge_feats[1, 2, 2] = data.edge_feats[0]
    exp_edge_feats[5, 2, 4] = data.edge_feats[1]
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


def test_slice_time_to_empty(data):
    dg = DGraph(data)

    # Slice Number 1
    dg1 = dg.slice_time(1, 15)
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

    exp_dynamic_node_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, 5)
    exp_dynamic_node_feats[1, 2] = data.dynamic_node_feats[0]
    exp_dynamic_node_feats[5, 4] = data.dynamic_node_feats[1]
    exp_dynamic_node_feats[10, 6] = data.dynamic_node_feats[2]
    assert torch.equal(dg1.dynamic_node_feats.to_dense(), exp_dynamic_node_feats)

    exp_edge_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, dg1.num_nodes, 5)
    exp_edge_feats[1, 2, 2] = data.edge_feats[0]
    exp_edge_feats[5, 2, 4] = data.edge_feats[1]
    assert torch.equal(dg1.edge_feats.to_dense(), exp_edge_feats)

    # Slice Number 2
    dg2 = dg1.slice_time(5, 15)
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

    exp_dynamic_node_feats = torch.zeros(dg2.end_time + 1, dg2.num_nodes, 5)
    exp_dynamic_node_feats[5, 4] = data.dynamic_node_feats[1]
    exp_dynamic_node_feats[10, 6] = data.dynamic_node_feats[2]
    assert torch.equal(dg2.dynamic_node_feats.to_dense(), exp_dynamic_node_feats)

    exp_edge_feats = torch.zeros(dg2.end_time + 1, dg2.num_nodes, dg2.num_nodes, 5)
    exp_edge_feats[5, 2, 4] = data.edge_feats[1]
    assert torch.equal(dg2.edge_feats.to_dense(), exp_edge_feats)

    # Slice number 3
    dg3 = dg2.slice_time(7, 11)
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

    exp_dynamic_node_feats = torch.zeros(dg3.end_time + 1, dg3.num_nodes, 5)
    exp_dynamic_node_feats[10, 6] = data.dynamic_node_feats[2]
    assert torch.equal(dg3.dynamic_node_feats.to_dense(), exp_dynamic_node_feats)

    assert dg3.edge_feats is None

    # Slice number 4 (to empty)
    dg4 = dg3.slice_time(0, 8)
    assert id(dg4._storage) == id(dg._storage)

    assert len(dg4) == 0
    assert dg4.start_time == 7
    assert dg4.end_time == 7
    assert dg4.num_nodes == 0
    assert dg4.num_edges == 0
    assert dg4.num_timestamps == 0
    assert dg4.nodes == set()
    assert dg4.dynamic_node_feats is None
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


def test_slice_time_bad_args(data):
    dg = DGraph(data)
    with pytest.raises(ValueError):
        dg.slice_time(2, 1)


def test_slice_events(data):
    dg = DGraph(data)

    dg1 = dg.slice_events(2, 5)
    assert id(dg1._storage) == id(dg._storage)

    assert len(dg1) == 2
    assert dg1.start_time == 5
    assert dg1.end_time == 10
    assert dg1.num_nodes == 7
    assert dg1.num_edges == 1
    assert dg1.num_timestamps == 2
    assert dg1.nodes == {2, 4, 6}

    exp_edges = (
        torch.LongTensor([2]),
        torch.LongTensor([4]),
        torch.LongTensor([5]),
    )
    torch.testing.assert_close(dg1.edges, exp_edges)

    exp_dynamic_node_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, 5)
    exp_dynamic_node_feats[5, 4] = data.dynamic_node_feats[1]
    exp_dynamic_node_feats[10, 6] = data.dynamic_node_feats[2]
    assert torch.equal(dg1.dynamic_node_feats.to_dense(), exp_dynamic_node_feats)

    exp_edge_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, dg1.num_nodes, 5)
    exp_edge_feats[5, 2, 4] = data.edge_feats[1]
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


def test_slice_events_bad_args(data):
    dg = DGraph(data)
    with pytest.raises(ValueError):
        dg.slice_events(2, 1)


def test_slice_events_slice_time_combination(data):
    dg = DGraph(data)

    dg1 = dg.slice_events(2, 5).slice_time(5, 7)
    assert id(dg1._storage) == id(dg._storage)

    assert len(dg1) == 1
    assert dg1.start_time == 5
    assert dg1.end_time == 6
    assert dg1.num_nodes == 5
    assert dg1.num_edges == 1
    assert dg1.num_timestamps == 1
    assert dg1.nodes == {2, 4}

    exp_edges = (
        torch.LongTensor([2]),
        torch.LongTensor([4]),
        torch.LongTensor([5]),
    )
    torch.testing.assert_close(dg1.edges, exp_edges)

    exp_dynamic_node_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, 5)
    exp_dynamic_node_feats[5, 4] = data.dynamic_node_feats[1]
    assert torch.equal(dg1.dynamic_node_feats.to_dense(), exp_dynamic_node_feats)

    exp_edge_feats = torch.zeros(dg1.end_time + 1, dg1.num_nodes, dg1.num_nodes, 5)
    exp_edge_feats[5, 2, 4] = data.edge_feats[1]
    assert torch.equal(dg1.edge_feats.to_dense(), exp_edge_feats)
