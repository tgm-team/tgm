from typing import List

import numpy as np
import pytest
import torch

from tgm import DGBatch, DGraph
from tgm.data import DGData
from tgm.hooks import NeighborSamplerHook
from tgm.loader import DGDataLoader


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


def test_hook_reset_state():
    assert NeighborSamplerHook.has_state == False


def test_bad_neighbor_sampler_init():
    with pytest.raises(ValueError):
        NeighborSamplerHook(num_nbrs=[])
    with pytest.raises(ValueError):
        NeighborSamplerHook(num_nbrs=[1, 0])


def test_neighbor_sampler_hook_link_pred(data):
    dg = DGraph(data)
    hook = NeighborSamplerHook(num_nbrs=[2])
    batch = dg.materialize()

    # Link Prediction will add negative edges to seed nodes for sampling
    batch.neg = torch.LongTensor([0] * len(batch.dst))
    batch.neg_time = torch.LongTensor([0] * len(batch.dst))
    batch = hook(dg, batch)
    assert isinstance(batch, DGBatch)
    assert hasattr(batch, 'nids')
    assert hasattr(batch, 'nbr_nids')
    assert hasattr(batch, 'nbr_times')
    assert hasattr(batch, 'nbr_feats')
    assert hasattr(batch, 'nbr_mask')


def test_neighbor_sampler_hook_node_pred(data):
    dg = DGraph(data)
    hook = NeighborSamplerHook(num_nbrs=[2])
    batch = hook(dg, dg.materialize())
    assert isinstance(batch, DGBatch)
    assert hasattr(batch, 'nids')
    assert hasattr(batch, 'nbr_nids')
    assert hasattr(batch, 'nbr_times')
    assert hasattr(batch, 'nbr_feats')
    assert hasattr(batch, 'nbr_mask')


EMPTY = -1  # use to indicate uninitialized vectors


def _nbrs_2_np(batch: DGBatch) -> List[np.ndarray]:
    """Convert neighbors in batch to numpy arrays."""
    assert isinstance(batch, DGBatch)
    assert hasattr(batch, 'nids')
    assert hasattr(batch, 'nbr_nids')
    assert hasattr(batch, 'nbr_times')
    assert hasattr(batch, 'nbr_feats')
    assert hasattr(batch, 'nbr_mask')

    nids = np.array(batch.nids)
    nbr_nids = np.array(batch.nbr_nids)
    nbr_times = np.array(batch.nbr_times)
    nbr_feats = np.array(batch.nbr_feats)
    nbr_mask = np.array(batch.nbr_mask)
    return [nids, nbr_nids, nbr_times, nbr_feats, nbr_mask]


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
    edge_index = torch.LongTensor([[0, 1], [0, 2], [2, 3], [2, 0]])
    edge_timestamps = torch.LongTensor([1, 2, 3, 4])
    edge_feats = torch.LongTensor(
        [[1], [2], [5], [2]]
    )  # edge feat is simply summing the node IDs at two end points
    data = DGData.from_raw(edge_timestamps, edge_index, edge_feats)
    return data


@pytest.mark.skip('TODO: unskip')
def test_init_basic_sampled_graph_1_hop(basic_sample_graph):
    """The goal of this test is to provide a simple TG with 1-hop neighbors
    and test the basic functionality of the neighbor sampler.
    also make sure recency and uniform samplers return the same output.
    """
    dg = DGraph(basic_sample_graph)
    n_nbrs = [1]  # 1 neighbor for each node
    uniform_hook = NeighborSamplerHook(num_nbrs=n_nbrs)
    loader = DGDataLoader(dg, hook=[uniform_hook], batch_size=1)
    assert loader._batch_size == 1

    batch_iter = iter(loader)
    batch_1 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats, nbr_mask = _nbrs_2_np(batch_1)
    assert nids.shape == (1, 2)
    assert nids[0][0] == 0
    assert nids[0][1] == 1
    assert nbr_nids.shape == (1, 2, 1)
    assert nbr_nids[0][0][0] == EMPTY
    assert nbr_nids[0][1][0] == EMPTY
    assert nbr_times.shape == (1, 2, 1)
    assert nbr_times[0][0][0] == EMPTY
    assert nbr_times[0][1][0] == EMPTY
    assert nbr_feats.shape == (1, 2, 1, 1)  # 1 feature per edge
    assert nbr_feats[0][1][0][0] == nbr_feats[0][0][0][0] == EMPTY
    assert nbr_mask.shape == (1, 2, 1)

    batch_2 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats, nbr_mask = _nbrs_2_np(batch_2)
    assert nids.shape == (1, 2)
    assert nids[0][0] == 0
    assert nids[0][1] == 2
    assert nbr_nids.shape == (1, 2, 1)
    assert nbr_nids[0][0][0] == 1
    assert nbr_nids[0][1][0] == EMPTY
    assert nbr_times.shape == (1, 2, 1)
    assert nbr_times[0][0][0] == 1
    assert nbr_times[0][1][0] == EMPTY
    assert nbr_feats.shape == (1, 2, 1, 1)  # 1 feature per edge
    assert nbr_feats[0][0][0][0] == 1.0
    assert nbr_feats[0][1][0][0] == EMPTY
    assert nbr_mask.shape == (1, 2, 1)

    batch_3 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats, nbr_mask = _nbrs_2_np(batch_3)
    assert nids.shape == (1, 2)
    assert nids[0][0] == 2
    assert nids[0][1] == 3
    assert nbr_nids.shape == (1, 2, 1)
    assert nbr_nids[0][0][0] == 0
    assert nbr_nids[0][1][0] == EMPTY
    assert nbr_times.shape == (1, 2, 1)
    assert nbr_times[0][0][0] == 2
    assert nbr_times[0][1][0] == EMPTY
    assert nbr_feats.shape == (1, 2, 1, 1)  # 1 feature per edge
    assert nbr_feats[0][0][0][0] == 2.0
    assert nbr_feats[0][1][0][0] == EMPTY
    assert nbr_mask.shape == (1, 2, 1)

    batch_4 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats, nbr_mask = _nbrs_2_np(batch_4)
    assert nids.shape == (1, 2)
    assert nids[0][0] == 2
    assert nids[0][1] == 0
    assert nbr_nids.shape == (1, 2, 1)
    assert nbr_nids[0][0][0] == 3
    assert nbr_nids[0][1][0] == 2
    assert nbr_times.shape == (1, 2, 1)
    assert nbr_times[0][0][0] == 3
    assert nbr_times[0][1][0] == 2
    assert nbr_feats.shape == (1, 2, 1, 1)  # 1 feature per edge
    assert nbr_feats[0][0][0][0] == 5.0
    assert nbr_feats[0][1][0][0] == 2.0
    assert nbr_mask.shape == (1, 2, 1)
