import random

import pytest
import torch

from tgm._storage import (
    DGSliceTracker,
    DGStorageBackends,
    get_dg_storage_backend,
    set_dg_storage_backend,
)
from tgm._storage.backends import DGStorageArrayBackend
from tgm.data import DGData
from tgm.timedelta import TimeDeltaDG


@pytest.fixture(autouse=True)
def seed():
    random.seed(1337)


@pytest.fixture(params=DGStorageBackends.values())
def DGStorageImpl(request):
    return request.param


@pytest.fixture
def edge_only_data():
    edge_index = torch.LongTensor([[2, 2], [2, 4], [6, 8]])
    edge_timestamps = torch.LongTensor([1, 5, 10])
    return DGData.from_raw(edge_timestamps, edge_index)


@pytest.fixture
def edge_only_data_with_features():
    edge_index = torch.LongTensor([[2, 2], [2, 4], [6, 8]])
    edge_timestamps = torch.LongTensor([1, 5, 10])
    edge_feats = torch.rand(3, 5)
    return DGData.from_raw(edge_timestamps, edge_index, edge_feats)


@pytest.fixture
def data_with_features():
    edge_index = torch.LongTensor([[2, 2], [2, 4], [1, 8]])
    edge_timestamps = torch.LongTensor([1, 5, 20])
    edge_feats = torch.rand(3, 5)
    node_timestamps = torch.LongTensor([1, 5, 10])
    node_ids = torch.LongTensor([2, 4, 6])
    dynamic_node_feats = torch.rand(3, 5)
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


def test_discretize_reduce_op_first(DGStorageImpl):
    edge_index = torch.LongTensor([[1, 2], [1, 2], [2, 3], [1, 2], [4, 5]])
    edge_timestamps = torch.LongTensor([1, 2, 3, 63, 65])
    edge_feats = torch.rand(5, 5)
    static_node_feats = torch.rand(6, 11)
    data = DGData.from_raw(
        edge_timestamps,
        edge_index,
        edge_feats,
        static_node_feats=static_node_feats,
    )
    storage = DGStorageImpl(data)
    old_granularity = TimeDeltaDG('m')
    new_granularity = TimeDeltaDG('h')
    coarse_storage = storage.discretize(
        old_granularity, new_granularity, reduce_op='first'
    )
    full_slice = DGSliceTracker()
    assert coarse_storage.get_start_time(full_slice) == 0
    assert coarse_storage.get_end_time(full_slice) == 1
    assert coarse_storage.get_num_timestamps(full_slice) == 2
    assert coarse_storage.get_num_events(full_slice) == 4
    assert coarse_storage.get_nodes(full_slice) == storage.get_nodes(full_slice)
    torch.testing.assert_close(
        coarse_storage.get_static_node_feats(), storage.get_static_node_feats()
    )
    assert coarse_storage.get_dynamic_node_feats(full_slice) is None

    exp_src = torch.LongTensor([1, 2, 1, 4])
    exp_dst = torch.LongTensor([2, 3, 2, 5])
    exp_time = torch.LongTensor([0, 0, 1, 1])
    exp_edges = (exp_src, exp_dst, exp_time)
    edges = coarse_storage.get_edges(full_slice)
    torch.testing.assert_close(edges, exp_edges)

    exp_edge_feats = torch.zeros(2, 5 + 1, 5 + 1, 5)
    exp_edge_feats[0, 1, 2] = data.edge_feats[0]
    exp_edge_feats[0, 2, 3] = data.edge_feats[2]
    exp_edge_feats[1, 1, 2] = data.edge_feats[3]
    exp_edge_feats[1, 4, 5] = data.edge_feats[4]
    assert torch.equal(
        coarse_storage.get_edge_feats(full_slice).to_dense(), exp_edge_feats
    )


def test_discretize_with_node_events_reduce_op_first(DGStorageImpl):
    edge_index = torch.LongTensor([[1, 2], [1, 2], [2, 3], [1, 2], [4, 5]])
    edge_timestamps = torch.LongTensor([1, 2, 3, 63, 65])
    edge_feats = torch.rand(5, 5)

    node_ids = torch.LongTensor([6, 6, 7, 6, 6, 7])
    node_timestamps = torch.LongTensor([10, 20, 30, 70, 80, 90])
    dynamic_node_feats = torch.rand(6, 5)
    static_node_feats = torch.rand(8, 11)
    data = DGData.from_raw(
        edge_timestamps,
        edge_index,
        edge_feats,
        node_timestamps,
        node_ids,
        dynamic_node_feats,
        static_node_feats,
    )
    storage = DGStorageImpl(data)

    old_granularity = TimeDeltaDG('m')
    new_granularity = TimeDeltaDG('h')
    coarse_storage = storage.discretize(
        old_granularity, new_granularity, reduce_op='first'
    )
    full_slice = DGSliceTracker()
    assert coarse_storage.get_start_time(full_slice) == 0
    assert coarse_storage.get_end_time(full_slice) == 1
    assert coarse_storage.get_num_timestamps(full_slice) == 2
    assert coarse_storage.get_num_events(full_slice) == 8
    assert coarse_storage.get_nodes(full_slice) == storage.get_nodes(full_slice)
    torch.testing.assert_close(
        coarse_storage.get_static_node_feats(), storage.get_static_node_feats()
    )

    exp_node_feats = torch.zeros(2, 7 + 1, 5)
    exp_node_feats[0, 6] = data.dynamic_node_feats[0]
    exp_node_feats[0, 7] = data.dynamic_node_feats[2]
    exp_node_feats[1, 6] = data.dynamic_node_feats[3]
    exp_node_feats[1, 7] = data.dynamic_node_feats[5]
    assert torch.equal(
        coarse_storage.get_dynamic_node_feats(full_slice).to_dense(), exp_node_feats
    )

    exp_src = torch.LongTensor([1, 2, 1, 4])
    exp_dst = torch.LongTensor([2, 3, 2, 5])
    exp_time = torch.LongTensor([0, 0, 1, 1])
    exp_edges = (exp_src, exp_dst, exp_time)
    edges = coarse_storage.get_edges(full_slice)
    torch.testing.assert_close(edges, exp_edges)

    exp_edge_feats = torch.zeros(2, 7 + 1, 7 + 1, 5)
    exp_edge_feats[0, 1, 2] = data.edge_feats[0]
    exp_edge_feats[0, 2, 3] = data.edge_feats[2]
    exp_edge_feats[1, 1, 2] = data.edge_feats[3]
    exp_edge_feats[1, 4, 5] = data.edge_feats[4]
    assert torch.equal(
        coarse_storage.get_edge_feats(full_slice).to_dense(), exp_edge_feats
    )


@pytest.mark.parametrize('data', ['edge_only_data', 'edge_only_data_with_features'])
def test_get_start_time_edge_data(DGStorageImpl, data, request):
    data = request.getfixturevalue(data)
    storage = DGStorageImpl(data)

    assert storage.get_start_time(DGSliceTracker()) == data.timestamps[0]
    assert storage.get_start_time(DGSliceTracker(start_time=5)) == 5
    assert storage.get_start_time(DGSliceTracker(end_time=4)) == 1
    assert storage.get_start_time(DGSliceTracker(start_time=5, end_time=9)) == 5
    assert storage.get_start_time(DGSliceTracker(start_idx=2, end_idx=5)) == 10
    assert (
        storage.get_start_time(DGSliceTracker(start_idx=2, end_idx=5, end_time=6))
        is None
    )


@pytest.mark.parametrize('data', ['edge_only_data', 'edge_only_data_with_features'])
def test_get_end_time_edge_data(DGStorageImpl, data, request):
    data = request.getfixturevalue(data)
    storage = DGStorageImpl(data)

    assert storage.get_end_time(DGSliceTracker()) == data.timestamps[-1]
    assert storage.get_end_time(DGSliceTracker(start_time=5)) == 10
    assert storage.get_end_time(DGSliceTracker(end_time=4)) == 1
    assert storage.get_end_time(DGSliceTracker(start_time=5, end_time=9)) == 5
    assert storage.get_end_time(DGSliceTracker(start_idx=2, end_idx=5)) == 10
    assert (
        storage.get_end_time(DGSliceTracker(start_idx=2, end_idx=5, end_time=6)) is None
    )


@pytest.mark.parametrize('data', ['edge_only_data', 'edge_only_data_with_features'])
def test_get_nodes_edge_data(DGStorageImpl, data, request):
    data = request.getfixturevalue(data)
    storage = DGStorageImpl(data)
    assert storage.get_nodes(DGSliceTracker()) == set([2, 4, 6, 8])
    assert storage.get_nodes(DGSliceTracker(start_time=5)) == set([2, 4, 6, 8])
    assert storage.get_nodes(DGSliceTracker(end_time=4)) == set([2])
    assert storage.get_nodes(DGSliceTracker(start_time=5, end_time=9)) == set([2, 4])
    assert storage.get_nodes(DGSliceTracker(start_idx=2, end_idx=5)) == set([6, 8])
    assert (
        storage.get_nodes(DGSliceTracker(start_idx=2, end_idx=5, end_time=6)) == set()
    )


def test_get_nodes_data_with_multi_events_per_timestamp(
    DGStorageImpl, data_with_features
):
    data = data_with_features
    storage = DGStorageImpl(data)

    assert storage.get_nodes(DGSliceTracker()) == set([1, 2, 4, 6, 8])
    assert storage.get_nodes(DGSliceTracker(start_time=5)) == set([1, 2, 4, 6, 8])
    assert storage.get_nodes(DGSliceTracker(end_time=4)) == set([2])
    assert storage.get_nodes(DGSliceTracker(start_time=5, end_time=9)) == set([2, 4])
    assert storage.get_nodes(DGSliceTracker(start_idx=2, end_idx=5)) == set([2, 4, 6])
    assert storage.get_nodes(DGSliceTracker(start_idx=2, end_idx=5, end_time=6)) == set(
        [2, 4]
    )


@pytest.mark.parametrize('data', ['edge_only_data', 'edge_only_data_with_features'])
def test_get_edges(DGStorageImpl, data, request):
    data = request.getfixturevalue(data)
    storage = DGStorageImpl(data)

    expected = (
        torch.tensor([2, 2, 6], dtype=torch.int64),
        torch.tensor([2, 4, 8], dtype=torch.int64),
        torch.tensor([1, 5, 10], dtype=torch.int64),
    )
    torch.testing.assert_close(storage.get_edges(DGSliceTracker()), expected)

    expected = (
        torch.tensor([2, 6], dtype=torch.int64),
        torch.tensor([4, 8], dtype=torch.int64),
        torch.tensor([5, 10], dtype=torch.int64),
    )
    torch.testing.assert_close(
        storage.get_edges(DGSliceTracker(start_time=5)), expected
    )

    expected = (
        torch.tensor([2], dtype=torch.int64),
        torch.tensor([2], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int64),
    )
    torch.testing.assert_close(storage.get_edges(DGSliceTracker(end_time=4)), expected)

    expected = (
        torch.tensor([2], dtype=torch.int64),
        torch.tensor([4], dtype=torch.int64),
        torch.tensor([5], dtype=torch.int64),
    )
    torch.testing.assert_close(
        storage.get_edges(DGSliceTracker(start_time=5, end_time=9)), expected
    )

    expected = (
        torch.tensor([6], dtype=torch.int64),
        torch.tensor([8], dtype=torch.int64),
        torch.tensor([10], dtype=torch.int64),
    )

    torch.testing.assert_close(
        storage.get_edges(DGSliceTracker(start_idx=2, end_idx=5)), expected
    )

    expected = (
        torch.tensor([], dtype=torch.int64),
        torch.tensor([], dtype=torch.int64),
        torch.tensor([], dtype=torch.int64),
    )
    torch.testing.assert_close(
        storage.get_edges(DGSliceTracker(start_idx=2, end_idx=5, end_time=6)), expected
    )


@pytest.mark.parametrize('data', ['edge_only_data', 'edge_only_data_with_features'])
def test_get_num_timestamps_edge_data(DGStorageImpl, data, request):
    data = request.getfixturevalue(data)
    storage = DGStorageImpl(data)

    assert storage.get_num_timestamps(DGSliceTracker()) == 3
    assert storage.get_num_timestamps(DGSliceTracker(start_time=5)) == 2
    assert storage.get_num_timestamps(DGSliceTracker(end_time=4)) == 1
    assert storage.get_num_timestamps(DGSliceTracker(start_time=5, end_time=9)) == 1
    assert storage.get_num_timestamps(DGSliceTracker(start_idx=2, end_idx=5)) == 1
    assert (
        storage.get_num_timestamps(DGSliceTracker(start_idx=2, end_idx=5, end_time=6))
        == 0
    )


def test_get_num_events_data_with_multi_events_per_timestamp(
    DGStorageImpl, data_with_features
):
    data = data_with_features
    storage = DGStorageImpl(data)

    assert storage.get_num_events(DGSliceTracker()) == 6
    assert storage.get_num_events(DGSliceTracker(start_time=5)) == 4
    assert storage.get_num_events(DGSliceTracker(end_time=4)) == 2
    assert storage.get_num_events(DGSliceTracker(start_time=5, end_time=9)) == 2
    assert storage.get_num_events(DGSliceTracker(start_idx=2, end_idx=5)) == 3
    assert (
        storage.get_num_events(DGSliceTracker(start_idx=2, end_idx=5, end_time=6)) == 2
    )


def test_get_edge_feats_no_edge_feats(DGStorageImpl, edge_only_data):
    data = edge_only_data
    storage = DGStorageImpl(data)

    assert storage.get_edge_feats(DGSliceTracker()) is None
    assert storage.get_edge_feats_dim() is None


def test_get_edge_feats(DGStorageImpl, edge_only_data_with_features):
    data = edge_only_data_with_features
    storage = DGStorageImpl(data)

    exp_edge_feats = torch.zeros(11, 8 + 1, 8 + 1, 5)
    exp_edge_feats[1, 2, 2] = data.edge_feats[0]
    exp_edge_feats[5, 2, 4] = data.edge_feats[1]
    exp_edge_feats[10, 6, 8] = data.edge_feats[2]
    assert torch.equal(
        storage.get_edge_feats(DGSliceTracker()).to_dense(), exp_edge_feats
    )

    exp_edge_feats = torch.zeros(11, 8 + 1, 8 + 1, 5)
    exp_edge_feats[5, 2, 4] = data.edge_feats[1]
    exp_edge_feats[10, 6, 8] = data.edge_feats[2]
    assert torch.equal(
        storage.get_edge_feats(DGSliceTracker(start_time=5)).to_dense(), exp_edge_feats
    )

    exp_edge_feats = torch.zeros(5, 2 + 1, 2 + 1, 5)
    exp_edge_feats[1, 2, 2] = data.edge_feats[0]
    assert torch.equal(
        storage.get_edge_feats(DGSliceTracker(end_time=4)).to_dense(), exp_edge_feats
    )

    exp_edge_feats = torch.zeros(10, 4 + 1, 4 + 1, 5)
    exp_edge_feats[5, 2, 4] = data.edge_feats[1]
    assert torch.equal(
        storage.get_edge_feats(DGSliceTracker(start_time=5, end_time=9)).to_dense(),
        exp_edge_feats,
    )

    exp_edge_feats = torch.zeros(11, 8 + 1, 8 + 1, 5)
    exp_edge_feats[10, 6, 8] = data.edge_feats[2]
    assert torch.equal(
        storage.get_edge_feats(DGSliceTracker(start_idx=2, end_idx=5)).to_dense(),
        exp_edge_feats,
    )

    assert (
        storage.get_edge_feats(DGSliceTracker(start_idx=2, end_idx=5, end_time=6))
        is None
    )


@pytest.mark.parametrize('data', ['edge_only_data', 'edge_only_data_with_features'])
def test_get_dynamic_node_feats_no_node_feats(DGStorageImpl, data, request):
    data = request.getfixturevalue(data)
    storage = DGStorageImpl(data)

    assert storage.get_dynamic_node_feats(DGSliceTracker()) is None
    assert storage.get_dynamic_node_feats_dim() is None


def test_get_dynamic_node_feats(DGStorageImpl, data_with_features):
    data = data_with_features
    storage = DGStorageImpl(data)

    exp_node_feats = torch.zeros(21, 8 + 1, 5)
    exp_node_feats[1, 2] = data.dynamic_node_feats[0]
    exp_node_feats[5, 4] = data.dynamic_node_feats[1]
    exp_node_feats[10, 6] = data.dynamic_node_feats[2]
    assert torch.equal(
        storage.get_dynamic_node_feats(DGSliceTracker()).to_dense(), exp_node_feats
    )

    exp_node_feats = torch.zeros(21, 8 + 1, 5)
    exp_node_feats[5, 4] = data.dynamic_node_feats[1]
    exp_node_feats[10, 6] = data.dynamic_node_feats[2]
    assert torch.equal(
        storage.get_dynamic_node_feats(DGSliceTracker(start_time=5)).to_dense(),
        exp_node_feats,
    )

    exp_node_feats = torch.zeros(5, 2 + 1, 5)
    exp_node_feats[1, 2] = data.dynamic_node_feats[0]
    assert torch.equal(
        storage.get_dynamic_node_feats(DGSliceTracker(end_time=4)).to_dense(),
        exp_node_feats,
    )

    exp_node_feats = torch.zeros(10, 4 + 1, 5)
    exp_node_feats[5, 4] = data.dynamic_node_feats[1]
    assert torch.equal(
        storage.get_dynamic_node_feats(
            DGSliceTracker(start_time=5, end_time=9)
        ).to_dense(),
        exp_node_feats,
    )

    exp_node_feats = torch.zeros(11, 6 + 1, 5)
    exp_node_feats[5, 4] = data.dynamic_node_feats[1]
    exp_node_feats[10, 6] = data.dynamic_node_feats[2]
    assert torch.equal(
        storage.get_dynamic_node_feats(
            DGSliceTracker(start_idx=2, end_idx=5)
        ).to_dense(),
        exp_node_feats,
    )

    exp_node_feats = torch.zeros(7, 4 + 1, 5)
    exp_node_feats[5, 4] = data.dynamic_node_feats[1]
    assert torch.equal(
        storage.get_dynamic_node_feats(
            DGSliceTracker(start_idx=2, end_idx=5, end_time=6)
        ).to_dense(),
        exp_node_feats,
    )


@pytest.mark.parametrize('data', ['edge_only_data', 'edge_only_data_with_features'])
def test_get_static_node_feats_no_node_feats(DGStorageImpl, data, request):
    data = request.getfixturevalue(data)
    storage = DGStorageImpl(data)
    assert storage.get_static_node_feats() is None
    assert storage.get_static_node_feats_dim() is None


def test_get_static_node_feats(DGStorageImpl, data_with_features):
    data = data_with_features
    storage = DGStorageImpl(data)
    assert storage.get_static_node_feats().shape == (9, 11)
    assert storage.get_static_node_feats_dim() == 11


@pytest.mark.skip('TODO: Add get_nbr')
def test_get_nbrs_single_hop(DGStorageImpl):
    edge_index = torch.LongTensor([[2, 2], [2, 4], [1, 8]])
    edge_timestamps = torch.LongTensor([1, 5, 20])
    data = DGData.from_raw(edge_timestamps, edge_index)
    storage = DGStorageImpl(data)

    nbrs = storage.get_nbrs(
        seed_nodes=[1, 2, 3, 4, 5, 6, 7, 8], num_nbrs=[-1], slice=DGSliceTracker()
    )
    exp_nbrs = {
        1: [[(8, 20)]],
        2: [[(2, 1), (4, 5)]],
        3: [[]],
        4: [[(2, 5)]],
        5: [[]],
        6: [[]],
        7: [[]],
        8: [[(1, 20)]],
    }
    assert nbrs.keys() == exp_nbrs.keys()
    for k, v in nbrs.items():
        for hop_num, nbrs in enumerate(v):
            assert sorted(nbrs) == sorted(exp_nbrs[k][hop_num])

    nbrs = storage.get_nbrs(
        seed_nodes=[1, 2, 3, 4, 5, 6, 7, 8],
        num_nbrs=[-1],
        slice=DGSliceTracker(start_time=5),
    )
    exp_nbrs = {
        1: [[(8, 20)]],
        2: [[(4, 5)]],
        3: [[]],
        4: [[(2, 5)]],
        5: [[]],
        6: [[]],
        7: [[]],
        8: [[(1, 20)]],
    }
    assert nbrs.keys() == exp_nbrs.keys()
    for k, v in nbrs.items():
        for hop_num, nbrs in enumerate(v):
            assert sorted(nbrs) == sorted(exp_nbrs[k][hop_num])

    nbrs = storage.get_nbrs(
        seed_nodes=[1, 2, 3, 4, 5, 6, 7, 8],
        num_nbrs=[-1],
        slice=DGSliceTracker(start_time=5, end_time=9),
    )
    exp_nbrs = {
        1: [[]],
        2: [[(4, 5)]],
        3: [[]],
        4: [[(2, 5)]],
        5: [[]],
        6: [[]],
        7: [[]],
        8: [[]],
    }
    assert nbrs.keys() == exp_nbrs.keys()
    for k, v in nbrs.items():
        for hop_num, nbrs in enumerate(v):
            assert sorted(nbrs) == sorted(exp_nbrs[k][hop_num])


@pytest.mark.skip('TODO: Add get_nbr tests')
def test_get_nbrs_single_hop_sampling_required(DGStorageImpl):
    edge_index = torch.LongTensor([[2, 2], [2, 4], [1, 8]])
    edge_timestamps = torch.LongTensor([1, 5, 20])
    data = DGData.from_raw(edge_timestamps, edge_index)
    storage = DGStorageImpl(data)

    nbrs = storage.get_nbrs(seed_nodes=[2], num_nbrs=[1], slice=DGSliceTracker())
    exp_nbrs = {
        2: [[(2, 1)]],
    }
    # TODO: Either return a set or make this easier to check
    assert nbrs.keys() == exp_nbrs.keys()
    for k, v in nbrs.items():
        for hop_num, nbrs in enumerate(v):
            assert sorted(nbrs) == sorted(exp_nbrs[k][hop_num])

    nbrs = storage.get_nbrs(
        seed_nodes=[2], num_nbrs=[1], slice=DGSliceTracker(end_time=4)
    )
    exp_nbrs = {
        2: [[(2, 1)]],
    }
    # TODO: Either return a set or make this easier to check
    assert nbrs.keys() == exp_nbrs.keys()
    for k, v in nbrs.items():
        for hop_num, nbrs in enumerate(v):
            assert sorted(nbrs) == sorted(exp_nbrs[k][hop_num])


@pytest.mark.skip(reason='Multi-hop get_nbrs not implemented')
def test_get_nbrs_multiple_hops(DGStorageImpl):
    pass


def test_get_dg_storage_backend():
    assert get_dg_storage_backend() == DGStorageArrayBackend


def test_set_dg_storage_backend_with_class():
    set_dg_storage_backend(DGStorageArrayBackend)
    assert get_dg_storage_backend() == DGStorageArrayBackend


def test_set_dg_storage_backend_with_str():
    set_dg_storage_backend('ArrayBackend')
    assert get_dg_storage_backend() == DGStorageArrayBackend


def test_set_dg_storage_backend_with_bad_str():
    with pytest.raises(ValueError):
        set_dg_storage_backend('foo')
