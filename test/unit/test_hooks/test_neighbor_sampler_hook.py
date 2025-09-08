from typing import List

import numpy as np
import pytest
import torch

from tgm import DGBatch, DGraph
from tgm.constants import PADDED_NODE_ID
from tgm.data import DGData
from tgm.hooks import HookManager, NeighborSamplerHook
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


def test_neighbor_sampler_hook_node_pred(data):
    dg = DGraph(data)
    hook = NeighborSamplerHook(num_nbrs=[2])
    batch = hook(dg, dg.materialize())
    assert isinstance(batch, DGBatch)
    assert hasattr(batch, 'nids')
    assert hasattr(batch, 'nbr_nids')
    assert hasattr(batch, 'nbr_times')
    assert hasattr(batch, 'nbr_feats')


def _nbrs_2_np(batch: DGBatch) -> List[np.ndarray]:
    assert isinstance(batch, DGBatch)
    assert hasattr(batch, 'nids')
    assert hasattr(batch, 'nbr_nids')
    assert hasattr(batch, 'nbr_times')
    assert hasattr(batch, 'nbr_feats')

    nids = np.array(batch.nids)
    nbr_nids = np.array(batch.nbr_nids)
    nbr_times = np.array(batch.nbr_times)
    nbr_feats = np.array(batch.nbr_feats)
    return [nids, nbr_nids, nbr_times, nbr_feats]


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


def test_init_basic_sampled_graph_1_hop(basic_sample_graph):
    dg = DGraph(basic_sample_graph)
    n_nbrs = [3]  # 3 neighbor for each node
    uniform_hook = NeighborSamplerHook(num_nbrs=n_nbrs)
    hm = HookManager(keys=['unit'])
    hm.register_shared(uniform_hook)
    loader = DGDataLoader(dg, batch_size=1, hook_manager=hm)
    assert loader._batch_size == 1
    with hm.activate('unit'):
        batch_iter = iter(loader)
        batch_1 = next(batch_iter)
        nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_1)
        assert nids.shape == (1, 2)
        assert nids[0][0] == 0
        assert nids[0][1] == 1
        assert nbr_nids.shape == (1, 2, 3)
        assert nbr_nids[0][0][0] == PADDED_NODE_ID
        assert nbr_nids[0][0][1] == PADDED_NODE_ID
        assert nbr_nids[0][0][2] == PADDED_NODE_ID
        assert nbr_nids[0][1][0] == PADDED_NODE_ID
        assert nbr_nids[0][1][1] == PADDED_NODE_ID
        assert nbr_nids[0][1][2] == PADDED_NODE_ID
        assert nbr_times.shape == (1, 2, 3)
        assert nbr_feats.shape == (1, 2, 3, 1)  # 1 feature per edge

        batch_2 = next(batch_iter)
        nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_2)
        assert nids.shape == (1, 2)
        assert nids[0][0] == 0
        assert nids[0][1] == 2
        assert nbr_nids.shape == (1, 2, 3)
        assert nbr_nids[0][0][0] == 1
        assert nbr_nids[0][0][1] == PADDED_NODE_ID
        assert nbr_nids[0][0][2] == PADDED_NODE_ID
        assert nbr_nids[0][1][0] == PADDED_NODE_ID
        assert nbr_nids[0][1][1] == PADDED_NODE_ID
        assert nbr_nids[0][1][2] == PADDED_NODE_ID
        assert nbr_times.shape == (1, 2, 3)
        assert nbr_feats.shape == (1, 2, 3, 1)  # 1 feature per edge
        assert nbr_feats[0][0][0][0] == 1.0

        batch_3 = next(batch_iter)
        nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_3)
        assert nids.shape == (1, 2)
        assert nids[0][0] == 2
        assert nids[0][1] == 3
        assert nbr_nids.shape == (1, 2, 3)
        assert nbr_nids[0][0][0] == 0
        assert nbr_nids[0][0][1] == PADDED_NODE_ID
        assert nbr_nids[0][0][2] == PADDED_NODE_ID
        assert nbr_nids[0][1][0] == PADDED_NODE_ID
        assert nbr_nids[0][1][1] == PADDED_NODE_ID
        assert nbr_nids[0][1][2] == PADDED_NODE_ID
        assert nbr_times.shape == (1, 2, 3)
        assert nbr_times[0][0][0] == 2
        assert nbr_feats.shape == (1, 2, 3, 1)  # 1 feature per edge
        assert nbr_feats[0][0][0][0] == 2.0

        batch_4 = next(batch_iter)
        nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_4)
        assert nids.shape == (1, 2)
        assert nids[0][0] == 2
        assert nids[0][1] == 0
        assert nbr_nids.shape == (1, 2, 3)
        assert nbr_nids[0][0][0] == 0
        assert nbr_nids[0][0][1] == 3
        assert nbr_nids[0][1][0] == 1
        assert nbr_nids[0][1][1] == 2
        assert nbr_times.shape == (1, 2, 3)
        assert nbr_times[0][0][0] == 2
        assert nbr_times[0][0][1] == 3
        assert nbr_times[0][1][0] == 1
        assert nbr_times[0][1][1] == 2
        assert nbr_feats.shape == (1, 2, 3, 1)  # 1 feature per edge
        assert nbr_feats[0][0][0][0] == 2.0
        assert nbr_feats[0][0][1][0] == 5.0
        assert nbr_feats[0][1][0][0] == 1.0
        assert nbr_feats[0][1][1][0] == 2.0


def test_init_basic_sampled_graph_directed_1_hop(basic_sample_graph):
    dg = DGraph(basic_sample_graph)
    n_nbrs = [3]  # 3 neighbor for each node
    uniform_hook = NeighborSamplerHook(num_nbrs=n_nbrs, directed=True)
    hm = HookManager(keys=['unit'])
    hm.register_shared(uniform_hook)
    loader = DGDataLoader(dg, batch_size=1, hook_manager=hm)
    assert loader._batch_size == 1
    with hm.activate('unit'):
        batch_iter = iter(loader)
        batch_1 = next(batch_iter)
        nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_1)
        assert nids.shape == (1, 2)
        assert nids[0][0] == 0
        assert nids[0][1] == 1
        assert nbr_nids.shape == (1, 2, 3)
        assert nbr_nids[0][0][0] == PADDED_NODE_ID
        assert nbr_nids[0][0][1] == PADDED_NODE_ID
        assert nbr_nids[0][0][2] == PADDED_NODE_ID
        assert nbr_nids[0][1][0] == PADDED_NODE_ID
        assert nbr_nids[0][1][1] == PADDED_NODE_ID
        assert nbr_nids[0][1][2] == PADDED_NODE_ID
        assert nbr_times.shape == (1, 2, 3)
        assert nbr_feats.shape == (1, 2, 3, 1)  # 1 feature per edge

        batch_2 = next(batch_iter)
        nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_2)
        assert nids.shape == (1, 2)
        assert nids[0][0] == 0
        assert nids[0][1] == 2
        assert nbr_nids.shape == (1, 2, 3)
        assert nbr_nids[0][0][0] == 1
        assert nbr_nids[0][0][1] == PADDED_NODE_ID
        assert nbr_nids[0][0][2] == PADDED_NODE_ID
        assert nbr_nids[0][1][0] == PADDED_NODE_ID
        assert nbr_nids[0][1][1] == PADDED_NODE_ID
        assert nbr_nids[0][1][2] == PADDED_NODE_ID
        assert nbr_times.shape == (1, 2, 3)
        assert nbr_feats.shape == (1, 2, 3, 1)  # 1 feature per edge
        assert nbr_feats[0][0][0][0] == 1.0

        batch_3 = next(batch_iter)
        nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_3)
        assert nids.shape == (1, 2)
        assert nids[0][0] == 2
        assert nids[0][1] == 3
        assert nbr_nids.shape == (1, 2, 3)
        assert nbr_nids[0][0][0] == PADDED_NODE_ID
        assert nbr_nids[0][0][1] == PADDED_NODE_ID
        assert nbr_nids[0][0][2] == PADDED_NODE_ID
        assert nbr_nids[0][1][0] == PADDED_NODE_ID
        assert nbr_nids[0][1][1] == PADDED_NODE_ID
        assert nbr_nids[0][1][2] == PADDED_NODE_ID
        assert nbr_times.shape == (1, 2, 3)
        assert nbr_times[0][0][0] == 0
        assert nbr_feats.shape == (1, 2, 3, 1)  # 1 feature per edge
        assert nbr_feats[0][0][0][0] == 0.0

        batch_4 = next(batch_iter)
        nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_4)
        assert nids.shape == (1, 2)
        assert nids[0][0] == 2
        assert nids[0][1] == 0
        assert nbr_nids.shape == (1, 2, 3)
        assert nbr_nids[0][0][0] == 3
        assert nbr_nids[0][0][1] == PADDED_NODE_ID
        assert nbr_nids[0][1][0] == 1
        assert nbr_nids[0][1][1] == 2
        assert nbr_times.shape == (1, 2, 3)
        assert nbr_times[0][0][0] == 3
        assert nbr_times[0][0][1] == 0
        assert nbr_times[0][1][0] == 1
        assert nbr_times[0][1][1] == 2
        assert nbr_feats.shape == (1, 2, 3, 1)  # 1 feature per edge
        assert nbr_feats[0][0][0][0] == 5.0
        assert nbr_feats[0][0][1][0] == 0.0
        assert nbr_feats[0][1][0][0] == 1.0
        assert nbr_feats[0][1][1][0] == 2.0
