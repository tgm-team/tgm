import pytest
import torch

from opendg.data import DGData


def test_init_dg_data_no_feats():
    edge_index = torch.Tensor([[2, 3], [10, 20]])
    timestamps = torch.Tensor([1, 5])
    data = DGData(edge_index, timestamps)

    torch.testing.assert_close(data.edge_index, edge_index)
    torch.testing.assert_close(data.timestamps, timestamps)
    assert data.edge_feats is None
    assert data.node_feats is None


def test_init_dg_data_with_feats():
    edge_index = torch.Tensor([[2, 3], [10, 20]])
    timestamps = torch.Tensor([1, 5])
    edge_feats = torch.rand(2, 5)
    node_feats = torch.rand(100, 10)
    data = DGData(edge_index, timestamps, edge_feats, node_feats)

    torch.testing.assert_close(data.edge_index, edge_index)
    torch.testing.assert_close(data.timestamps, timestamps)
    torch.testing.assert_close(data.edge_feats, edge_feats)
    torch.testing.assert_close(data.node_feats, node_feats)


def test_init_dg_data_sort_required():
    edge_index = torch.Tensor([[2, 3], [10, 20]])
    timestamps = torch.Tensor([5, 1])
    edge_feats = torch.rand(2, 5)
    node_feats = torch.rand(100, 10)
    data = DGData(edge_index, timestamps, edge_feats, node_feats)

    # Expected sort
    exp_edge_index = torch.Tensor([[10, 20], [2, 3]])
    exp_timestamps = torch.Tensor([1, 5])
    exp_edge_feats = torch.Tensor([edge_feats[1].tolist(), edge_feats[0].tolist()])

    torch.testing.assert_close(data.edge_index, exp_edge_index)
    torch.testing.assert_close(data.timestamps, exp_timestamps)
    torch.testing.assert_close(data.edge_feats, exp_edge_feats)
    torch.testing.assert_close(data.node_feats, node_feats)


def test_init_dg_data_bad_args():
    edge_index = torch.Tensor([[2, 3], [10, 20]])
    timestamps = torch.Tensor([5, 1])
    edge_feats = torch.rand(2, 5)

    # Empty graph not supported
    with pytest.raises(ValueError):
        _ = DGData(torch.empty((0, 2)), torch.empty(0))

    # Bad types
    with pytest.raises(TypeError):
        _ = DGData('foo', timestamps)
    with pytest.raises(TypeError):
        _ = DGData(edge_index, 'foo')
    with pytest.raises(TypeError):
        _ = DGData(edge_index, timestamps, 'foo')
    with pytest.raises(TypeError):
        _ = DGData(edge_index, timestamps, edge_feats, 'foo')

    # Bad shapes
    with pytest.raises(ValueError):
        _ = DGData(torch.Tensor([1, 2]), timestamps)
    with pytest.raises(ValueError):
        _ = DGData(torch.Tensor([[1, 2, 3], [10, 20, 30]]), timestamps)
    with pytest.raises(ValueError):
        _ = DGData(edge_index, torch.Tensor([1]))
    with pytest.raises(ValueError):
        _ = DGData(edge_index, timestamps, torch.rand(1, 5))
    with pytest.raises(ValueError):
        _ = DGData(edge_index, timestamps, torch.rand(1, 5, 5))
    with pytest.raises(ValueError):
        _ = DGData(edge_index, timestamps, torch.rand(2))
    with pytest.raises(ValueError):
        _ = DGData(edge_index, timestamps, edge_feats, torch.rand(2))
    with pytest.raises(ValueError):
        _ = DGData(edge_index, timestamps, edge_feats, torch.rand(2, 5, 5))
