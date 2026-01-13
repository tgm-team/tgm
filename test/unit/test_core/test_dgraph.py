from dataclasses import asdict

import pytest
import torch

from tgm import DGBatch, DGraph
from tgm.data import DGData


@pytest.fixture
def data():
    edge_index = torch.IntTensor([[2, 2], [2, 4], [1, 8]])
    edge_timestamps = torch.LongTensor([1, 5, 20])
    edge_x = torch.rand(3, 5)
    node_timestamps = torch.LongTensor([1, 5, 10])
    node_ids = torch.IntTensor([2, 4, 6])
    node_x = torch.rand([3, 5])
    static_node_x = torch.rand(9, 11)
    edge_type = torch.IntTensor([0, 1, 2])
    node_type = torch.arange(9, dtype=torch.int32)
    return DGData.from_raw(
        edge_timestamps,
        edge_index,
        edge_x,
        node_timestamps,
        node_ids,
        node_x,
        static_node_x,
        edge_type=edge_type,
        node_type=node_type,
    )


@pytest.fixture
def unorder_data():
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([5, 1])
    edge_x = torch.rand(2, 5)
    node_ids = torch.IntTensor([1, 2, 3])
    node_timestamps = torch.LongTensor([8, 7, 6])
    node_x = torch.rand(3, 7)
    static_node_x = torch.rand(21, 11)
    data = DGData.from_raw(
        edge_timestamps,
        edge_index,
        edge_x,
        node_timestamps,
        node_ids,
        node_x,
        static_node_x,
    )
    return data


def test_init_from_data(data):
    dg = DGraph(data)

    assert dg.time_delta.is_event_ordered

    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edge_events == 3
    assert dg.num_timestamps == 4
    assert dg.num_events == 6
    assert dg.static_node_x_dim == 11
    assert dg.node_x_dim == 5
    assert dg.edge_x_dim == 5
    assert dg.device == torch.device('cpu')

    expected_edges = (
        torch.tensor([2, 2, 1], dtype=torch.int32),
        torch.tensor([2, 4, 8], dtype=torch.int32),
        torch.tensor([1, 5, 20], dtype=torch.int64),
    )
    torch.testing.assert_close(dg.edge_events, expected_edges)

    exp_node_x = torch.zeros(dg.end_time + 1, dg.num_nodes, 5)
    exp_node_x[1, 2] = data.node_x[0]
    exp_node_x[5, 4] = data.node_x[1]
    exp_node_x[10, 6] = data.node_x[2]
    torch.testing.assert_close(dg.node_x.to_dense(), exp_node_x)
    torch.testing.assert_close(dg.edge_x, data.edge_x)
    torch.testing.assert_close(dg.edge_type, data.edge_type)
    torch.testing.assert_close(dg.node_type, data.node_type)


@pytest.mark.gpu
def test_init_gpu(data):
    dg = DGraph(data, device='cuda')

    assert dg.device == torch.device('cuda')

    expected_edges = (
        torch.tensor([2, 2, 1], dtype=torch.int32, device='cuda'),
        torch.tensor([2, 4, 8], dtype=torch.int32, device='cuda'),
        torch.tensor([1, 5, 20], dtype=torch.int64, device='cuda'),
    )
    torch.testing.assert_close(dg.edge_events, expected_edges)

    exp_node_x = torch.zeros(dg.end_time + 1, dg.num_nodes, 5)
    exp_node_x[1, 2] = data.node_x[0]
    exp_node_x[5, 4] = data.node_x[1]
    exp_node_x[10, 6] = data.node_x[2]
    exp_node_x = exp_node_x.cuda()
    torch.testing.assert_close(dg.node_x.to_dense(), exp_node_x)
    torch.testing.assert_close(dg.edge_x, data.edge_x.to('cuda'))
    torch.testing.assert_close(dg.edge_type, data.edge_type.to('cuda'))
    torch.testing.assert_close(dg.node_type, data.node_type.to('cuda'))


def test_to_cpu(data):
    dg = DGraph(data)
    dg = dg.to('cpu')
    assert dg.device == torch.device('cpu')


@pytest.mark.gpu
def test_to_gpu(data):
    dg = DGraph(data)
    dg = dg.to('cuda')
    assert dg.device == torch.device('cuda')


def test_init_from_storage(data):
    dg_tmp = DGraph(data)
    dg = DGraph._from_storage(
        dg_tmp._storage, dg_tmp.time_delta, dg_tmp.device, dg_tmp._slice
    )
    assert id(dg_tmp._storage) == id(dg._storage)


def test_str(data):
    dg = DGraph(data)
    assert isinstance(dg.__str__(), str)


def test_init_bad_data():
    with pytest.raises(TypeError):
        DGraph('foo')


def test_materialize(data):
    dg = DGraph(data)
    exp_src = torch.tensor([2, 2, 1], dtype=torch.int32)
    exp_dst = torch.tensor([2, 4, 8], dtype=torch.int32)
    exp_t = torch.tensor([1, 5, 20], dtype=torch.int64)
    edge_type = torch.IntTensor([0, 1, 2])
    exp = DGBatch(
        exp_src,
        exp_dst,
        exp_t,
        dg.node_x._values(),
        dg.edge_x,
        dg.node_x._indices()[0],
        dg.node_x._indices()[1].int(),
        edge_type=edge_type,
    )
    torch.testing.assert_close(asdict(dg.materialize()), asdict(exp))


def test_materialize_skip_feature_materialization(data):
    dg = DGraph(data)
    exp_src = torch.tensor([2, 2, 1], dtype=torch.int32)
    exp_dst = torch.tensor([2, 4, 8], dtype=torch.int32)
    exp_t = torch.tensor([1, 5, 20], dtype=torch.int64)
    exp_edge_type = torch.IntTensor([0, 1, 2])
    exp = DGBatch(exp_src, exp_dst, exp_t, None, None, edge_type=exp_edge_type)
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
    assert dg1.num_edge_events == 2
    assert dg1.num_timestamps == 3

    exp_edges = (
        torch.IntTensor([2, 1]),
        torch.IntTensor([4, 8]),
        torch.LongTensor([5, 20]),
    )
    torch.testing.assert_close(dg1.edge_events, exp_edges)

    exp_node_x = torch.zeros(dg1.end_time + 1, dg1.num_nodes, 5)
    exp_node_x[5, 4] = data.node_x[1]
    exp_node_x[10, 6] = data.node_x[2]
    torch.testing.assert_close(dg1.node_x.to_dense(), exp_node_x)
    torch.testing.assert_close(dg1.edge_x, data.edge_x[1:3])
    torch.testing.assert_close(dg.static_node_x, dg1.static_node_x)
    torch.testing.assert_close(dg.node_type, dg1.node_type)
    torch.testing.assert_close(dg1.edge_type, data.edge_type[1:3])


def test_slice_time_at_end_time(data):
    dg = DGraph(data)

    dg1 = dg.slice_time(1, 20)
    assert id(dg1._storage) == id(dg._storage)

    assert len(dg1) == 3
    assert dg1.start_time == 1
    assert dg1.end_time == 19  # Note: this is 19 despite no events in [10, 19)
    assert dg1.num_nodes == 7
    assert dg1.num_edge_events == 2
    assert dg1.num_timestamps == 3

    exp_edges = (
        torch.IntTensor([2, 2]),
        torch.IntTensor([2, 4]),
        torch.LongTensor([1, 5]),
    )
    torch.testing.assert_close(dg1.edge_events, exp_edges)

    exp_node_x = torch.zeros(dg1.end_time + 1, dg1.num_nodes, 5)
    exp_node_x[1, 2] = data.node_x[0]
    exp_node_x[5, 4] = data.node_x[1]
    exp_node_x[10, 6] = data.node_x[2]
    assert torch.equal(dg1.node_x.to_dense(), exp_node_x)
    assert torch.equal(dg1.edge_x, data.edge_x[0:2])
    assert torch.equal(dg1.edge_type, data.edge_type[0:2])

    # Check original graph cache is not updated
    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edge_events == 3
    assert dg.num_timestamps == 4
    assert dg.num_events == 6


def test_slice_time_to_empty(data):
    dg = DGraph(data)

    # Slice Number 1
    dg1 = dg.slice_time(1, 15)
    assert id(dg1._storage) == id(dg._storage)

    assert len(dg1) == 3
    assert dg1.start_time == 1
    assert dg1.end_time == 14
    assert dg1.num_nodes == 7
    assert dg1.num_edge_events == 2
    assert dg1.num_timestamps == 3

    exp_edges = (
        torch.IntTensor([2, 2]),
        torch.IntTensor([2, 4]),
        torch.LongTensor([1, 5]),
    )
    torch.testing.assert_close(dg1.edge_events, exp_edges)

    exp_node_x = torch.zeros(dg1.end_time + 1, dg1.num_nodes, 5)
    exp_node_x[1, 2] = data.node_x[0]
    exp_node_x[5, 4] = data.node_x[1]
    exp_node_x[10, 6] = data.node_x[2]
    assert torch.equal(dg1.node_x.to_dense(), exp_node_x)
    assert torch.equal(dg1.edge_x, data.edge_x[:2])
    assert torch.equal(dg1.edge_type, data.edge_type[:2])

    # Slice Number 2
    dg2 = dg1.slice_time(5, 15)
    assert id(dg2._storage) == id(dg._storage)

    assert len(dg2) == 2
    assert dg2.start_time == 5
    assert dg2.end_time == 14
    assert dg2.num_nodes == 7
    assert dg2.num_edge_events == 1
    assert dg2.num_timestamps == 2

    exp_edges = (
        torch.IntTensor([2]),
        torch.IntTensor([4]),
        torch.LongTensor([5]),
    )
    torch.testing.assert_close(dg2.edge_events, exp_edges)

    exp_node_x = torch.zeros(dg2.end_time + 1, dg2.num_nodes, 5)
    exp_node_x[5, 4] = data.node_x[1]
    exp_node_x[10, 6] = data.node_x[2]
    assert torch.equal(dg2.node_x.to_dense(), exp_node_x)
    assert torch.equal(dg2.edge_x, data.edge_x[1].unsqueeze(0))
    assert torch.equal(dg2.edge_type, data.edge_type[1].unsqueeze(0))

    # Slice number 3
    dg3 = dg2.slice_time(7, 11)
    assert id(dg3._storage) == id(dg._storage)

    assert len(dg3) == 1
    assert dg3.start_time == 7
    assert dg3.end_time == 10
    assert dg3.num_nodes == 7
    assert dg3.num_edge_events == 0
    assert dg3.num_timestamps == 1

    exp_edges = (torch.IntTensor([]), torch.IntTensor([]), torch.LongTensor([]))
    torch.testing.assert_close(dg3.edge_events, exp_edges)

    exp_node_x = torch.zeros(dg3.end_time + 1, dg3.num_nodes, 5)
    exp_node_x[10, 6] = data.node_x[2]
    assert torch.equal(dg3.node_x.to_dense(), exp_node_x)

    assert dg3.edge_x is None
    assert dg3.edge_type is None

    # Slice number 4 (to empty)
    dg4 = dg3.slice_time(0, 8)
    assert id(dg4._storage) == id(dg._storage)

    assert len(dg4) == 0
    assert dg4.start_time == 7
    assert dg4.end_time == 7
    assert dg4.num_nodes == 0
    assert dg4.num_edge_events == 0
    assert dg4.num_timestamps == 0
    assert dg4.node_x is None
    assert dg4.edge_x is None
    assert dg4.edge_type is None

    exp_edges = (torch.IntTensor([]), torch.IntTensor([]), torch.LongTensor([]))
    torch.testing.assert_close(dg4.edge_events, exp_edges)

    # Check original graph cache is not updated
    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edge_events == 3
    assert dg.num_timestamps == 4
    assert dg.num_events == 6


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
    assert dg1.num_edge_events == 1
    assert dg1.num_timestamps == 2

    exp_edges = (
        torch.IntTensor([2]),
        torch.IntTensor([4]),
        torch.LongTensor([5]),
    )
    torch.testing.assert_close(dg1.edge_events, exp_edges)

    exp_node_x = torch.zeros(dg1.end_time + 1, dg1.num_nodes, 5)
    exp_node_x[5, 4] = data.node_x[1]
    exp_node_x[10, 6] = data.node_x[2]
    assert torch.equal(dg1.node_x.to_dense(), exp_node_x)
    assert torch.equal(dg1.edge_x, data.edge_x[1].unsqueeze(0))
    assert torch.equal(dg1.edge_type, data.edge_type[1].unsqueeze(0))

    # Check original graph cache is not updated
    assert len(dg) == 4
    assert dg.start_time == 1
    assert dg.end_time == 20
    assert dg.num_nodes == 9
    assert dg.num_edge_events == 3
    assert dg.num_timestamps == 4
    assert dg.num_events == 6


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
    assert dg1.num_edge_events == 1
    assert dg1.num_timestamps == 1

    exp_edges = (
        torch.IntTensor([2]),
        torch.IntTensor([4]),
        torch.LongTensor([5]),
    )
    torch.testing.assert_close(dg1.edge_events, exp_edges)

    exp_node_x = torch.zeros(dg1.end_time + 1, dg1.num_nodes, 5)
    exp_node_x[5, 4] = data.node_x[1]
    assert torch.equal(dg1.node_x.to_dense(), exp_node_x)
    assert torch.equal(dg1.edge_x, data.edge_x[1].unsqueeze(0))
    assert torch.equal(dg1.edge_type, data.edge_type[1].unsqueeze(0))


def test_batch_stringify(data):
    dg = DGraph(data)
    batch = dg.materialize()
    batch.foo = ['a']  # Add an iterable to the batch
    assert isinstance(str(batch), str)


def test_unorder_data_init(unorder_data):
    dg = DGraph(unorder_data)
    src, dst, t = dg.edge_events
    expected_src = torch.IntTensor([10, 2])
    expected_dst = torch.IntTensor([20, 3])
    expected_edge_time = torch.tensor([1, 5], dtype=torch.int64)
    expected_node_id = torch.tensor([3, 2, 1], dtype=torch.int64)
    expected_node_times = torch.tensor([6, 7, 8], dtype=torch.int64)

    torch.testing.assert_close(src, expected_src)
    torch.testing.assert_close(dst, expected_dst)
    torch.testing.assert_close(t, expected_edge_time)

    node_times, node_id = dg.node_x._indices()
    torch.testing.assert_close(node_times, expected_node_times)
    torch.testing.assert_close(node_id, expected_node_id)
