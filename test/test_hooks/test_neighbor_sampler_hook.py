import pytest
import torch

from opendg.data import DGData
from opendg.graph import DGBatch, DGraph
from opendg.hooks import NeighborSamplerHook


@pytest.fixture
def data():
    # TODO: Missing node events
    edge_index = torch.Tensor([[2, 2], [2, 4], [1, 8]])
    timestamps = torch.Tensor([1, 5, 20])
    return DGData(edge_index, timestamps)
    # return [
    #    NodeEvent(t=1, src=2),
    #    EdgeEvent(t=1, src=2, dst=2),
    #    NodeEvent(t=5, src=4),
    #    EdgeEvent(t=5, src=2, dst=4),
    #    NodeEvent(t=10, src=6),
    #    EdgeEvent(t=20, src=1, dst=8),
    # ]


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
