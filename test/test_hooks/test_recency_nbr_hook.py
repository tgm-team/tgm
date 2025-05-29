import pytest
import torch

from opendg.data import DGData
from opendg.graph import DGBatch, DGraph
from opendg.hooks import RecencyNeighborHook


@pytest.fixture
def data():
    edge_index = torch.Tensor([[1, 10], [1, 11], [1, 12], [1, 13]])
    timestamps = torch.Tensor([1, 1, 2, 2])
    return DGData(edge_index, timestamps)


def test_bad_neighbor_sampler_init():
    with pytest.raises(ValueError):
        RecencyNeighborHook(num_nbrs=[0], num_nodes=2)
    with pytest.raises(ValueError):
        RecencyNeighborHook(num_nbrs=[-1], num_nodes=2)
    with pytest.raises(ValueError):
        RecencyNeighborHook(num_nbrs=[1, 2], num_nodes=2)


@pytest.mark.skip('TODO: Add neighbor sampling tests')
def test_neighbor_sampler_hook(data):
    dg = DGraph(data)
    hook = RecencyNeighborHook(num_nbrs=[2], num_nodes=dg.num_nodes)
    batch = hook(dg)
    # TODO: Add logic for testing
    assert isinstance(batch, DGBatch)
