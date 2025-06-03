import pytest
import torch

from opendg.data import DGData


def test_init_dg_data_no_node_events():
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    data = DGData.from_raw(edge_timestamps, edge_index)
    torch.testing.assert_close(data.edge_index, edge_index)
    torch.testing.assert_close(data.timestamps, edge_timestamps)
    torch.testing.assert_close(data.edge_event_idx, torch.LongTensor([0, 1]))
    assert data.edge_feats is None
    assert data.node_event_idx is None
    assert data.node_ids is None
    assert data.dynamic_node_feats is None
    assert data.static_node_feats is None


def test_init_dg_data_no_node_events_with_edge_features():
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    edge_feats = torch.rand(2, 5)
    data = DGData.from_raw(edge_timestamps, edge_index, edge_feats)
    torch.testing.assert_close(data.edge_index, edge_index)
    torch.testing.assert_close(data.timestamps, edge_timestamps)
    torch.testing.assert_close(data.edge_feats, edge_feats)
    torch.testing.assert_close(data.edge_event_idx, torch.LongTensor([0, 1]))
    assert data.node_event_idx is None
    assert data.node_ids is None
    assert data.dynamic_node_feats is None
    assert data.static_node_feats is None


def test_init_dg_data_node_events():
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    edge_feats = torch.rand(2, 5)
    node_ids = torch.LongTensor([1, 2, 3])
    node_timestamps = torch.LongTensor([6, 7, 8])
    data = DGData.from_raw(
        edge_timestamps, edge_index, edge_feats, node_timestamps, node_ids
    )
    torch.testing.assert_close(data.edge_index, edge_index)
    torch.testing.assert_close(data.timestamps, torch.LongTensor([1, 5, 6, 7, 8]))
    torch.testing.assert_close(data.edge_feats, edge_feats)
    torch.testing.assert_close(data.edge_event_idx, torch.LongTensor([0, 1]))
    torch.testing.assert_close(data.node_event_idx, torch.LongTensor([2, 3, 4]))
    torch.testing.assert_close(data.node_ids, node_ids)
    assert data.dynamic_node_feats is None
    assert data.static_node_feats is None


def test_init_dg_data_node_events_and_node_features():
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    edge_feats = torch.rand(2, 5)
    node_ids = torch.LongTensor([1, 2, 3])
    node_timestamps = torch.LongTensor([6, 7, 8])
    dynamic_node_feats = torch.rand(3, 7)
    static_node_feats = torch.rand(21, 11)
    data = DGData.from_raw(
        edge_timestamps,
        edge_index,
        edge_feats,
        node_timestamps,
        node_ids,
        dynamic_node_feats,
        static_node_feats,
    )
    torch.testing.assert_close(data.edge_index, edge_index)
    torch.testing.assert_close(data.timestamps, torch.LongTensor([1, 5, 6, 7, 8]))
    torch.testing.assert_close(data.edge_feats, edge_feats)
    torch.testing.assert_close(data.edge_event_idx, torch.LongTensor([0, 1]))
    torch.testing.assert_close(data.node_event_idx, torch.LongTensor([2, 3, 4]))
    torch.testing.assert_close(data.node_ids, node_ids)
    torch.testing.assert_close(data.dynamic_node_feats, dynamic_node_feats)
    torch.testing.assert_close(data.static_node_feats, static_node_feats)


def test_init_dg_data_sort_required():
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([5, 1])
    edge_feats = torch.rand(2, 5)
    node_ids = torch.LongTensor([1, 2, 3])
    node_timestamps = torch.LongTensor([8, 7, 6])
    dynamic_node_feats = torch.rand(3, 7)
    static_node_feats = torch.rand(21, 11)
    data = DGData.from_raw(
        edge_timestamps,
        edge_index,
        edge_feats,
        node_timestamps,
        node_ids,
        dynamic_node_feats,
        static_node_feats,
    )

    exp_edge_index = torch.LongTensor([[10, 20], [2, 3]])
    exp_node_ids = torch.LongTensor([3, 2, 1])
    exp_edge_feats = torch.Tensor([edge_feats[1].tolist(), edge_feats[0].tolist()])
    exp_dynamic_node_feats = torch.Tensor(
        [
            dynamic_node_feats[2].tolist(),
            dynamic_node_feats[1].tolist(),
            dynamic_node_feats[0].tolist(),
        ]
    )
    torch.testing.assert_close(data.edge_index, exp_edge_index)
    torch.testing.assert_close(data.timestamps, torch.LongTensor([1, 5, 6, 7, 8]))
    torch.testing.assert_close(data.edge_feats, exp_edge_feats)
    torch.testing.assert_close(data.edge_event_idx, torch.LongTensor([0, 1]))
    torch.testing.assert_close(data.node_event_idx, torch.LongTensor([2, 3, 4]))
    torch.testing.assert_close(data.node_ids, exp_node_ids)
    torch.testing.assert_close(data.dynamic_node_feats, exp_dynamic_node_feats)
    torch.testing.assert_close(data.static_node_feats, static_node_feats)


def test_init_dg_data_bad_args_empty_graph():
    # Empty graph not supported
    with pytest.raises(ValueError):
        _ = DGData.from_raw(torch.empty((0, 2)), torch.empty(0))


def test_init_dg_data_bad_args_bad_types():
    # Num edges = 2, Num nodes = 21, D_edge = 5, Num node events = 3, D_node_dynamic = 7
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    edge_feats = torch.rand(2, 5)
    node_ids = torch.LongTensor([1, 2, 3])
    node_timestamps = torch.LongTensor([6, 7, 8])
    dynamic_node_feats = torch.rand(3, 7)

    with pytest.raises(TypeError):
        _ = DGData.from_raw(edge_timestamps, 'foo')
    with pytest.raises(TypeError):
        _ = DGData.from_raw(edge_timestamps, edge_index, 'foo')
    with pytest.raises(TypeError):
        _ = DGData.from_raw(edge_timestamps, edge_index, edge_feats, 'foo')
    with pytest.raises(TypeError):
        _ = DGData.from_raw(edge_timestamps, edge_index, edge_feats, node_timestamps)
    with pytest.raises(TypeError):
        _ = DGData.from_raw(
            edge_timestamps, edge_index, edge_feats, node_timestamps, node_ids, 'foo'
        )
    with pytest.raises(TypeError):
        _ = DGData.from_raw(
            edge_timestamps,
            edge_index,
            edge_feats,
            node_timestamps,
            node_ids,
            dynamic_node_feats,
            'foo',
        )


def test_init_dg_data_bad_static_node_feats_shape():
    # Num nodes = 21
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    node_ids = torch.LongTensor([1, 2, 3])
    node_timestamps = torch.LongTensor([6, 7, 8])

    with pytest.raises(ValueError):
        _ = DGData.from_raw(
            edge_timestamps,
            edge_index,
            None,
            node_timestamps,
            node_ids,
            None,
            torch.rand(20, 11),  # should be [21, ...]
        )

    # Num nodes = 21
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])

    with pytest.raises(ValueError):
        _ = DGData.from_raw(
            edge_timestamps,
            edge_index,
            None,
            None,
            None,
            None,
            torch.rand(20, 11),  # should be [21, ...]
        )

    # Num nodes = 101
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    node_ids = torch.LongTensor([1, 2, 100])
    node_timestamps = torch.LongTensor([6, 7, 8])

    with pytest.raises(ValueError):
        _ = DGData.from_raw(
            edge_timestamps,
            edge_index,
            None,
            None,
            None,
            None,
            torch.rand(20, 11),  # should be [101, ...]
        )
