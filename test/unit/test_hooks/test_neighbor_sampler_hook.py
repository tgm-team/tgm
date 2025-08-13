import pytest
import torch

from tgm import DGBatch, DGraph
from tgm._storage import DGStorage
from tgm.data import DGData
from tgm.hooks import NeighborSamplerHook


@pytest.fixture
def data():
    edge_index = torch.LongTensor([[2, 2], [2, 4], [1, 8]])
    edge_timestamps = torch.LongTensor([1, 5, 20])
    edge_feats = torch.rand(3, 5)
    return DGData.from_raw(edge_timestamps, edge_index, edge_feats)


def test_hook_dependancies():
    assert NeighborSamplerHook.requires == set()
    assert NeighborSamplerHook.produces == {
        'nids',
        'nbr_nids',
        'nbr_times',
        'nbr_feats',
        'nbr_mask',
    }


def test_bad_neighbor_sampler_init():
    with pytest.raises(ValueError):
        NeighborSamplerHook(num_nbrs=[])
    with pytest.raises(ValueError):
        NeighborSamplerHook(num_nbrs=[1, 0])


def test_neighbor_sampler_hook_link_pred(data):
    data = DGStorage(data)
    dg = DGraph(data, discretize_time_delta='r')
    hook = NeighborSamplerHook(num_nbrs=[2])
    batch = dg.materialize()

    # Link Prediction will add negative edges to seed nodes for sampling
    batch.neg = torch.LongTensor([0] * len(batch.dst))
    batch = hook(dg, batch)
    assert isinstance(batch, DGBatch)
    assert hasattr(batch, 'nids')
    assert hasattr(batch, 'nbr_nids')
    assert hasattr(batch, 'nbr_times')
    assert hasattr(batch, 'nbr_feats')
    assert hasattr(batch, 'nbr_mask')


def test_neighbor_sampler_hook_node_pred(data):
    data = DGStorage(data)
    dg = DGraph(data, discretize_time_delta='r')
    hook = NeighborSamplerHook(num_nbrs=[2])
    batch = hook(dg, dg.materialize())
    assert isinstance(batch, DGBatch)
    assert hasattr(batch, 'nids')
    assert hasattr(batch, 'nbr_nids')
    assert hasattr(batch, 'nbr_times')
    assert hasattr(batch, 'nbr_feats')
    assert hasattr(batch, 'nbr_mask')
