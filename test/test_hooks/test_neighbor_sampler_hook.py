import pytest
import torch

from opendg.data import DGData
from opendg.graph import DGBatch, DGraph
from opendg.hooks import NeighborSamplerHook


@pytest.fixture
def data():
    edge_index = torch.LongTensor([[2, 2], [2, 4], [1, 8]])
    edge_timestamps = torch.LongTensor([1, 5, 20])
    return DGData.from_raw(edge_timestamps, edge_index)


def test_hook_dependancies():
    assert NeighborSamplerHook.requires == {'neg'}
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


@pytest.mark.skip('TODO: Add neighbor sampling tests')
def test_neighbor_sampler_hook(data):
    dg = DGraph(data)
    hook = NeighborSamplerHook(num_nbrs=[2])
    batch = hook(dg)
    assert isinstance(batch, DGBatch)

    # TODO: Add logic for testing


@pytest.mark.skip('TODO: Add neighbor sampling tests')
def test_neighbor_sampler_hook_full_neighborhood(data):
    dg = DGraph(data)
    hook = NeighborSamplerHook(num_nbrs=[-1])
    batch = hook(dg)
    assert isinstance(batch, DGBatch)

    # TODO: Add logic for testing
