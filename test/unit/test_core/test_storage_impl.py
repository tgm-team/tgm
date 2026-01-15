import random

import pytest
import torch

from tgm import DGraph
from tgm.constants import PADDED_NODE_ID
from tgm.core._storage import (
    DGSliceTracker,
    DGStorageBackends,
    get_dg_storage_backend,
    set_dg_storage_backend,
)
from tgm.core._storage.backends import DGStorageArrayBackend
from tgm.data import DGData


@pytest.fixture(autouse=True)
def seed():
    random.seed(1337)


@pytest.fixture(params=DGStorageBackends.values())
def DGStorageImpl(request):
    return request.param


@pytest.fixture
def edge_only_data():
    edge_index = torch.IntTensor([[2, 2], [2, 4], [6, 8]])
    edge_timestamps = torch.LongTensor([1, 5, 10])
    return DGData.from_raw(edge_timestamps, edge_index)


@pytest.fixture
def edge_only_data_with_features():
    edge_index = torch.IntTensor([[2, 2], [2, 4], [6, 8]])
    edge_timestamps = torch.LongTensor([1, 5, 10])
    edge_x = torch.rand(3, 5)
    return DGData.from_raw(edge_timestamps, edge_index, edge_x)


@pytest.fixture
def edge_only_data_with_edge_type():
    edge_index = torch.IntTensor([[2, 2], [2, 4], [6, 8]])
    edge_timestamps = torch.LongTensor([1, 5, 10])
    edge_type = torch.IntTensor([0, 1, 2])
    return DGData.from_raw(edge_timestamps, edge_index, edge_type=edge_type)


@pytest.fixture
def edge_only_data_with_node_type():
    edge_index = torch.IntTensor([[2, 2], [2, 4], [6, 8]])
    edge_timestamps = torch.LongTensor([1, 5, 10])
    node_type = torch.arange(9, dtype=torch.int32)
    return DGData.from_raw(edge_timestamps, edge_index, node_type=node_type)


@pytest.fixture
def data_with_features():
    edge_index = torch.IntTensor([[2, 2], [2, 4], [1, 8]])
    edge_timestamps = torch.LongTensor([1, 5, 20])
    edge_x = torch.rand(3, 5)
    node_timestamps = torch.LongTensor([1, 5, 10])
    node_ids = torch.IntTensor([2, 4, 6])
    node_x = torch.rand(3, 5)
    static_node_x = torch.rand(9, 11)
    node_type = torch.arange(9, dtype=torch.int32)
    edge_type = torch.IntTensor([0, 1, 2])
    return DGData.from_raw(
        edge_timestamps,
        edge_index,
        edge_x,
        node_timestamps,
        node_ids,
        node_x,
        static_node_x=static_node_x,
        edge_type=edge_type,
        node_type=node_type,
    )


@pytest.mark.parametrize('data', ['edge_only_data', 'edge_only_data_with_features'])
def test_get_start_time_edge_data(DGStorageImpl, data, request):
    data = request.getfixturevalue(data)
    storage = DGStorageImpl(data)

    assert storage.get_start_time(DGSliceTracker()) == data.time[0]
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

    assert storage.get_end_time(DGSliceTracker()) == data.time[-1]
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
        torch.tensor([2, 2, 6], dtype=torch.int32),
        torch.tensor([2, 4, 8], dtype=torch.int32),
        torch.tensor([1, 5, 10], dtype=torch.int64),
    )
    torch.testing.assert_close(storage.get_edges(DGSliceTracker()), expected)

    expected = (
        torch.tensor([2, 6], dtype=torch.int32),
        torch.tensor([4, 8], dtype=torch.int32),
        torch.tensor([5, 10], dtype=torch.int64),
    )
    torch.testing.assert_close(
        storage.get_edges(DGSliceTracker(start_time=5)), expected
    )

    expected = (
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int64),
    )
    torch.testing.assert_close(storage.get_edges(DGSliceTracker(end_time=4)), expected)

    expected = (
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
        torch.tensor([5], dtype=torch.int64),
    )
    torch.testing.assert_close(
        storage.get_edges(DGSliceTracker(start_time=5, end_time=9)), expected
    )

    expected = (
        torch.tensor([6], dtype=torch.int32),
        torch.tensor([8], dtype=torch.int32),
        torch.tensor([10], dtype=torch.int64),
    )

    torch.testing.assert_close(
        storage.get_edges(DGSliceTracker(start_idx=2, end_idx=5)), expected
    )

    expected = (
        torch.tensor([], dtype=torch.int32),
        torch.tensor([], dtype=torch.int32),
        torch.tensor([], dtype=torch.int64),
    )
    torch.testing.assert_close(
        storage.get_edges(DGSliceTracker(start_idx=2, end_idx=5, end_time=6)), expected
    )


@pytest.mark.parametrize('data', ['data_with_features'])
def test_get_node_events(DGStorageImpl, data, request):
    data = request.getfixturevalue(data)
    storage = DGStorageImpl(data)

    expected = (
        torch.tensor([2, 4, 6], dtype=torch.int32),
        torch.tensor([1, 5, 10], dtype=torch.int64),
    )
    torch.testing.assert_close(storage.get_node_events(DGSliceTracker()), expected)

    expected = (
        torch.tensor([4, 6], dtype=torch.int32),
        torch.tensor([5, 10], dtype=torch.int64),
    )
    torch.testing.assert_close(
        storage.get_node_events(DGSliceTracker(start_time=5)), expected
    )

    expected = (
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int64),
    )
    torch.testing.assert_close(
        storage.get_node_events(DGSliceTracker(end_time=4)), expected
    )

    expected = (
        torch.tensor([6], dtype=torch.int32),
        torch.tensor([10], dtype=torch.int64),
    )
    torch.testing.assert_close(
        storage.get_node_events(DGSliceTracker(start_idx=4, end_idx=5)), expected
    )

    expected = (
        torch.tensor([], dtype=torch.int32),
        torch.tensor([], dtype=torch.int64),
    )
    torch.testing.assert_close(
        storage.get_node_events(DGSliceTracker(start_idx=4, end_idx=5, end_time=6)),
        expected,
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

    assert storage.get_edge_x(DGSliceTracker()) is None
    assert storage.get_edge_x_dim() is None


def test_get_edge_feats(DGStorageImpl, edge_only_data_with_features):
    data = edge_only_data_with_features
    storage = DGStorageImpl(data)

    assert torch.equal(
        storage.get_edge_x(DGSliceTracker()),
        edge_only_data_with_features.edge_x,
    )
    assert torch.equal(
        storage.get_edge_x(DGSliceTracker(start_time=5)),
        edge_only_data_with_features.edge_x[1:],
    )
    assert torch.equal(
        storage.get_edge_x(DGSliceTracker(end_time=4)),
        edge_only_data_with_features.edge_x[:1],
    )
    assert torch.equal(
        storage.get_edge_x(DGSliceTracker(start_time=5)),
        edge_only_data_with_features.edge_x[1:],
    )
    assert torch.equal(
        storage.get_edge_x(DGSliceTracker(start_idx=2, end_idx=5)),
        edge_only_data_with_features.edge_x[2].unsqueeze(0),
    )
    assert (
        storage.get_edge_x(DGSliceTracker(start_idx=2, end_idx=5, end_time=6)) is None
    )


@pytest.mark.parametrize('data', ['edge_only_data', 'edge_only_data_with_features'])
def test_get_dynamic_node_feats_no_node_feats(DGStorageImpl, data, request):
    data = request.getfixturevalue(data)
    storage = DGStorageImpl(data)

    assert storage.get_node_x(DGSliceTracker()) is None
    assert storage.get_node_x_dim() is None


def test_get_dynamic_node_feats(DGStorageImpl, data_with_features):
    data = data_with_features
    storage = DGStorageImpl(data)

    exp_node_feats = torch.zeros(21, 8 + 1, 5)
    exp_node_feats[1, 2] = data.node_x[0]
    exp_node_feats[5, 4] = data.node_x[1]
    exp_node_feats[10, 6] = data.node_x[2]
    assert torch.equal(storage.get_node_x(DGSliceTracker()).to_dense(), exp_node_feats)

    exp_node_feats = torch.zeros(21, 8 + 1, 5)
    exp_node_feats[5, 4] = data.node_x[1]
    exp_node_feats[10, 6] = data.node_x[2]
    assert torch.equal(
        storage.get_node_x(DGSliceTracker(start_time=5)).to_dense(),
        exp_node_feats,
    )

    exp_node_feats = torch.zeros(5, 2 + 1, 5)
    exp_node_feats[1, 2] = data.node_x[0]
    assert torch.equal(
        storage.get_node_x(DGSliceTracker(end_time=4)).to_dense(),
        exp_node_feats,
    )

    exp_node_feats = torch.zeros(10, 4 + 1, 5)
    exp_node_feats[5, 4] = data.node_x[1]
    assert torch.equal(
        storage.get_node_x(DGSliceTracker(start_time=5, end_time=9)).to_dense(),
        exp_node_feats,
    )

    exp_node_feats = torch.zeros(11, 6 + 1, 5)
    exp_node_feats[5, 4] = data.node_x[1]
    exp_node_feats[10, 6] = data.node_x[2]
    assert torch.equal(
        storage.get_node_x(DGSliceTracker(start_idx=2, end_idx=5)).to_dense(),
        exp_node_feats,
    )

    exp_node_feats = torch.zeros(7, 4 + 1, 5)
    exp_node_feats[5, 4] = data.node_x[1]
    assert torch.equal(
        storage.get_node_x(
            DGSliceTracker(start_idx=2, end_idx=5, end_time=6)
        ).to_dense(),
        exp_node_feats,
    )


@pytest.mark.parametrize('data', ['edge_only_data', 'edge_only_data_with_features'])
def test_get_static_node_feats_no_node_feats(DGStorageImpl, data, request):
    data = request.getfixturevalue(data)
    storage = DGStorageImpl(data)
    assert storage.get_static_node_x() is None
    assert storage.get_static_node_x_dim() is None


def test_get_static_node_feats(DGStorageImpl, data_with_features):
    data = data_with_features
    storage = DGStorageImpl(data)
    assert storage.get_static_node_x().shape == (9, 11)
    assert storage.get_static_node_x_dim() == 11


@pytest.mark.parametrize('data', ['edge_only_data'])
def test_get_node_type_no_node_type(DGStorageImpl, data, request):
    data = request.getfixturevalue(data)
    storage = DGStorageImpl(data)
    assert storage.get_node_type() is None


def test_get_node_type(DGStorageImpl, edge_only_data_with_node_type):
    data = edge_only_data_with_node_type
    storage = DGStorageImpl(data)
    assert storage.get_node_type().shape == (9,)


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


@pytest.fixture
def basic_sample_graph():
    """Initializes the following graph.

    #############                    ###########
    # Alice (0) # ->    t = 1     -> # Bob (1) #
    #############                    ###########
         |
         v
       t = 2
         |
         v
    #############                    ############
    # Carol (2) # ->   t = 3      -> # Dave (3) #
    #############                    ############
         |
         v
       t = 4
         |
         v
    #############
    # Alice (0) #
    #############
    """
    edge_index = torch.IntTensor([[0, 1], [0, 2], [2, 3], [2, 0]])
    edge_timestamps = torch.LongTensor([1, 2, 3, 4])
    edge_x = torch.Tensor(
        [[1], [2], [5], [2]]
    )  # edge feat is simply summing the node IDs at two end points
    data = DGData.from_raw(edge_timestamps, edge_index, edge_x)
    return data


def test_dg_storage_get_nbrs(DGStorageImpl, basic_sample_graph):
    null_value = 0

    storage = DGStorageImpl(basic_sample_graph)
    dg = DGraph(basic_sample_graph)
    n_nbrs = 3  # 3 neighbor for each node
    end_time = 1
    nbr_nids, nbr_times, nbr_feats = storage.get_nbrs(
        torch.tensor([0, 1]),
        num_nbrs=n_nbrs,
        slice=DGSliceTracker(end_time=(end_time - 1)),  # type: ignore
        directed=False,
    )
    assert nbr_nids.shape == (2, 3)
    assert nbr_nids[0][0] == PADDED_NODE_ID
    assert nbr_nids[0][1] == PADDED_NODE_ID
    assert nbr_nids[0][2] == PADDED_NODE_ID
    assert nbr_nids[1][0] == PADDED_NODE_ID
    assert nbr_nids[1][1] == PADDED_NODE_ID
    assert nbr_nids[1][2] == PADDED_NODE_ID
    assert nbr_times.shape == (2, 3)
    assert nbr_times[0][0] == null_value
    assert nbr_times[0][1] == null_value
    assert nbr_times[0][2] == null_value
    assert nbr_times[1][0] == null_value
    assert nbr_times[1][1] == null_value
    assert nbr_times[1][2] == null_value
    assert nbr_feats.shape == (2, 3, 1)  # 1 feature per edge
    assert nbr_feats[0][0][0] == null_value
    assert nbr_feats[0][1][0] == null_value
    assert nbr_feats[0][2][0] == null_value
    assert nbr_feats[1][0][0] == null_value
    assert nbr_feats[1][1][0] == null_value
    assert nbr_feats[1][2][0] == null_value

    end_time = 2
    nbr_nids, nbr_times, nbr_feats = dg._storage.get_nbrs(
        torch.tensor([0, 2]),
        num_nbrs=n_nbrs,
        slice=DGSliceTracker(end_time=(end_time - 1)),  # type: ignore
        directed=False,
    )
    assert nbr_nids.shape == (2, 3)
    assert nbr_nids[0][0] == 1
    assert nbr_nids[0][1] == PADDED_NODE_ID
    assert nbr_nids[0][2] == PADDED_NODE_ID
    assert nbr_nids[1][0] == PADDED_NODE_ID
    assert nbr_nids[1][1] == PADDED_NODE_ID
    assert nbr_nids[1][2] == PADDED_NODE_ID
    assert nbr_times.shape == (2, 3)
    assert nbr_times[0][0] == 1
    assert nbr_times[0][1] == null_value
    assert nbr_times[0][2] == null_value
    assert nbr_times[1][0] == null_value
    assert nbr_times[1][1] == null_value
    assert nbr_times[1][2] == null_value
    assert nbr_feats.shape == (2, 3, 1)  # 1 feature per edge
    assert nbr_feats[0][0][0] == 1.0
    assert nbr_feats[0][1][0] == null_value
    assert nbr_feats[0][2][0] == null_value
    assert nbr_feats[1][0][0] == null_value
    assert nbr_feats[1][1][0] == null_value
    assert nbr_feats[1][2][0] == null_value

    end_time = 3
    nbr_nids, nbr_times, nbr_feats = dg._storage.get_nbrs(
        torch.tensor([2, 3]),
        num_nbrs=n_nbrs,
        slice=DGSliceTracker(end_time=(end_time - 1)),  # type: ignore
        directed=False,
    )
    assert nbr_nids.shape == (2, 3)
    assert nbr_nids[0][0] == 0
    assert nbr_nids[0][1] == PADDED_NODE_ID
    assert nbr_nids[0][2] == PADDED_NODE_ID
    assert nbr_nids[1][0] == PADDED_NODE_ID
    assert nbr_nids[1][1] == PADDED_NODE_ID
    assert nbr_nids[1][2] == PADDED_NODE_ID
    assert nbr_times.shape == (2, 3)
    assert nbr_times[0][0] == 2
    assert nbr_times[0][1] == null_value
    assert nbr_times[0][2] == null_value
    assert nbr_times[1][0] == null_value
    assert nbr_times[1][1] == null_value
    assert nbr_times[1][2] == null_value
    assert nbr_feats.shape == (2, 3, 1)  # 1 feature per edge
    assert nbr_feats[0][0][0] == 2.0
    assert nbr_feats[0][1][0] == null_value
    assert nbr_feats[0][2][0] == null_value
    assert nbr_feats[1][0][0] == null_value
    assert nbr_feats[1][1][0] == null_value
    assert nbr_feats[1][2][0] == null_value

    end_time = 4
    nbr_nids, nbr_times, nbr_feats = dg._storage.get_nbrs(
        torch.tensor([2, 0]),
        num_nbrs=n_nbrs,
        slice=DGSliceTracker(end_time=(end_time - 1)),  # type: ignore
        directed=False,
    )
    assert nbr_nids.shape == (2, 3)
    assert nbr_nids[0][0] == 0
    assert nbr_nids[0][1] == 3
    assert nbr_nids[0][2] == PADDED_NODE_ID
    assert nbr_nids[1][0] == 1
    assert nbr_nids[1][1] == 2
    assert nbr_nids[1][2] == PADDED_NODE_ID
    assert nbr_times.shape == (2, 3)
    assert nbr_times[0][0] == 2
    assert nbr_times[0][1] == 3
    assert nbr_times[0][2] == null_value
    assert nbr_times[1][0] == 1
    assert nbr_times[1][1] == 2
    assert nbr_times[1][2] == null_value
    assert nbr_feats.shape == (2, 3, 1)  # 1 feature per edge
    assert nbr_feats[0][0][0] == 2.0
    assert nbr_feats[0][1][0] == 5.0
    assert nbr_feats[0][2][0] == null_value
    assert nbr_feats[1][0][0] == 1.0
    assert nbr_feats[1][1][0] == 2.0
    assert nbr_feats[1][2][0] == null_value


def test_dg_storage_get_nbrs_directed(DGStorageImpl, basic_sample_graph):
    null_value = 0

    storage = DGStorageImpl(basic_sample_graph)
    dg = DGraph(basic_sample_graph)
    n_nbrs = 3  # 3 neighbor for each node
    end_time = 1
    nbr_nids, nbr_times, nbr_feats = storage.get_nbrs(
        torch.tensor([0, 1]),
        num_nbrs=n_nbrs,
        slice=DGSliceTracker(end_time=(end_time - 1)),  # type: ignore
        directed=True,
    )
    assert nbr_nids.shape == (2, 3)
    assert nbr_nids[0][0] == PADDED_NODE_ID
    assert nbr_nids[0][1] == PADDED_NODE_ID
    assert nbr_nids[0][2] == PADDED_NODE_ID
    assert nbr_nids[1][0] == PADDED_NODE_ID
    assert nbr_nids[1][1] == PADDED_NODE_ID
    assert nbr_nids[1][2] == PADDED_NODE_ID
    assert nbr_times.shape == (2, 3)
    assert nbr_times[0][0] == null_value
    assert nbr_times[0][1] == null_value
    assert nbr_times[0][2] == null_value
    assert nbr_times[1][0] == null_value
    assert nbr_times[1][1] == null_value
    assert nbr_times[1][2] == null_value
    assert nbr_feats.shape == (2, 3, 1)  # 1 feature per edge
    assert nbr_feats[0][0][0] == null_value
    assert nbr_feats[0][1][0] == null_value
    assert nbr_feats[0][2][0] == null_value
    assert nbr_feats[1][0][0] == null_value
    assert nbr_feats[1][1][0] == null_value
    assert nbr_feats[1][2][0] == null_value

    end_time = 2
    nbr_nids, nbr_times, nbr_feats = dg._storage.get_nbrs(
        torch.tensor([0, 2]),
        num_nbrs=n_nbrs,
        slice=DGSliceTracker(end_time=(end_time - 1)),  # type: ignore
        directed=True,
    )
    assert nbr_nids.shape == (2, 3)
    assert nbr_nids[0][0] == 1
    assert nbr_nids[0][1] == PADDED_NODE_ID
    assert nbr_nids[0][2] == PADDED_NODE_ID
    assert nbr_nids[1][0] == PADDED_NODE_ID
    assert nbr_nids[1][1] == PADDED_NODE_ID
    assert nbr_nids[1][2] == PADDED_NODE_ID
    assert nbr_times.shape == (2, 3)
    assert nbr_times[0][0] == 1
    assert nbr_times[0][1] == null_value
    assert nbr_times[0][2] == null_value
    assert nbr_times[1][0] == null_value
    assert nbr_times[1][1] == null_value
    assert nbr_times[1][2] == null_value
    assert nbr_feats.shape == (2, 3, 1)  # 1 feature per edge
    assert nbr_feats[0][0][0] == 1.0
    assert nbr_feats[0][1][0] == null_value
    assert nbr_feats[0][2][0] == null_value
    assert nbr_feats[1][0][0] == null_value
    assert nbr_feats[1][1][0] == null_value
    assert nbr_feats[1][2][0] == null_value

    end_time = 3
    nbr_nids, nbr_times, nbr_feats = dg._storage.get_nbrs(
        torch.tensor([2, 3]),
        num_nbrs=n_nbrs,
        slice=DGSliceTracker(end_time=(end_time - 1)),  # type: ignore
        directed=True,
    )
    assert nbr_nids.shape == (2, 3)
    assert nbr_nids[0][0] == PADDED_NODE_ID
    assert nbr_nids[0][1] == PADDED_NODE_ID
    assert nbr_nids[0][2] == PADDED_NODE_ID
    assert nbr_nids[1][0] == PADDED_NODE_ID
    assert nbr_nids[1][1] == PADDED_NODE_ID
    assert nbr_nids[1][2] == PADDED_NODE_ID
    assert nbr_times.shape == (2, 3)
    assert nbr_times[0][0] == null_value
    assert nbr_times[0][1] == null_value
    assert nbr_times[0][2] == null_value
    assert nbr_times[1][0] == null_value
    assert nbr_times[1][1] == null_value
    assert nbr_times[1][2] == null_value
    assert nbr_feats.shape == (2, 3, 1)  # 1 feature per edge
    assert nbr_feats[0][0][0] == null_value
    assert nbr_feats[0][1][0] == null_value
    assert nbr_feats[0][2][0] == null_value
    assert nbr_feats[1][0][0] == null_value
    assert nbr_feats[1][1][0] == null_value
    assert nbr_feats[1][2][0] == null_value

    end_time = 4
    nbr_nids, nbr_times, nbr_feats = dg._storage.get_nbrs(
        torch.tensor([2, 0]),
        num_nbrs=n_nbrs,
        slice=DGSliceTracker(end_time=(end_time - 1)),  # type: ignore
        directed=True,
    )
    assert nbr_nids.shape == (2, 3)
    assert nbr_nids[0][0] == 3
    assert nbr_nids[0][1] == PADDED_NODE_ID
    assert nbr_nids[0][2] == PADDED_NODE_ID
    assert nbr_nids[1][0] == 1
    assert nbr_nids[1][1] == 2
    assert nbr_nids[1][2] == PADDED_NODE_ID
    assert nbr_times.shape == (2, 3)
    assert nbr_times[0][0] == 3
    assert nbr_times[0][1] == null_value
    assert nbr_times[0][2] == null_value
    assert nbr_times[1][0] == 1
    assert nbr_times[1][1] == 2
    assert nbr_times[1][2] == null_value
    assert nbr_feats.shape == (2, 3, 1)  # 1 feature per edge
    assert nbr_feats[0][0][0] == 5.0
    assert nbr_feats[0][1][0] == null_value
    assert nbr_feats[0][2][0] == null_value
    assert nbr_feats[1][0][0] == 1.0
    assert nbr_feats[1][1][0] == 2.0
    assert nbr_feats[1][2][0] == null_value


def test_dg_storage_get_nbrs_sample_required(DGStorageImpl):
    edge_index = torch.IntTensor([[0, 1], [0, 2], [0, 3], [0, 4]])
    edge_timestamps = torch.LongTensor([1, 2, 3, 4])
    edge_x = torch.Tensor([[1], [2], [3], [4]])
    data = DGData.from_raw(edge_timestamps, edge_index, edge_x)
    storage = DGStorageImpl(data)
    nbr_nids, nbr_times, nbr_feats = storage.get_nbrs(
        torch.tensor([0]), num_nbrs=1, slice=DGSliceTracker(), directed=False
    )
    torch.testing.assert_close(nbr_nids, torch.IntTensor([[3]]))
    torch.testing.assert_close(nbr_times, torch.LongTensor([[3]]))
    torch.testing.assert_close(nbr_feats, torch.FloatTensor([[[3]]]))


def test_get_edge_type_no_edge_type(DGStorageImpl, edge_only_data):
    data = edge_only_data
    storage = DGStorageImpl(data)

    assert storage.get_edge_type(DGSliceTracker()) is None


def test_get_edge_type(DGStorageImpl, edge_only_data_with_edge_type):
    data = edge_only_data_with_edge_type
    storage = DGStorageImpl(data)

    assert torch.equal(
        storage.get_edge_type(DGSliceTracker()),
        edge_only_data_with_edge_type.edge_type,
    )
    assert torch.equal(
        storage.get_edge_type(DGSliceTracker(start_time=5)),
        edge_only_data_with_edge_type.edge_type[1:],
    )
    assert torch.equal(
        storage.get_edge_type(DGSliceTracker(end_time=4)),
        edge_only_data_with_edge_type.edge_type[:1],
    )
    assert torch.equal(
        storage.get_edge_type(DGSliceTracker(start_time=5)),
        edge_only_data_with_edge_type.edge_type[1:],
    )
    assert torch.equal(
        storage.get_edge_type(DGSliceTracker(start_idx=2, end_idx=5)),
        edge_only_data_with_edge_type.edge_type[2].unsqueeze(0),
    )
    assert (
        storage.get_edge_type(DGSliceTracker(start_idx=2, end_idx=5, end_time=6))
        is None
    )
