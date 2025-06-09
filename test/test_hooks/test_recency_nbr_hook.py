import pytest
import torch

from opendg.data import DGData
from opendg.graph import DGraph
from opendg.hooks import RecencyNeighborHook


@pytest.fixture
def data():
    edge_index = torch.LongTensor([[1, 10], [1, 11], [1, 12], [1, 13]])
    edge_timestamps = torch.LongTensor([1, 1, 2, 2])
    return DGData.from_raw(edge_timestamps, edge_index)


def test_hook_dependancies():
    assert RecencyNeighborHook.requires == set()
    assert RecencyNeighborHook.produces == {
        'nids',
        'nbr_nids',
        'nbr_times',
        'nbr_feats',
        'nbr_mask',
    }


def test_bad_neighbor_sampler_init():
    with pytest.raises(ValueError):
        RecencyNeighborHook(num_nbrs=[0], num_nodes=2)
    with pytest.raises(ValueError):
        RecencyNeighborHook(num_nbrs=[-1], num_nodes=2)
    with pytest.raises(ValueError):
        RecencyNeighborHook(num_nbrs=[1, 2], num_nodes=2)


def test_neighbor_sampler_hook_init(data):
    dg = DGraph(data)
    hook = RecencyNeighborHook(num_nbrs=[2], num_nodes=dg.num_nodes)
    assert hook.num_nbrs == [2]


@pytest.mark.skip('TODO: Add neighbor sampling tests')
def test_neighbor_sampler_hook_link_pred(data):
    pass


@pytest.mark.skip('TODO: Add neighbor sampling tests')
def test_neighbor_sampler_hook_node_pred(data):
    pass
