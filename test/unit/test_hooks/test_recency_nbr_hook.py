from typing import List
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from tgm import DGBatch, DGraph
from tgm.constants import PADDED_NODE_ID
from tgm.data import DGData
from tgm.hooks import (
    NeighborSamplerHook,
    RecencyNeighborHook,
    TGBNegativeEdgeSamplerHook,
)
from tgm.loader import DGDataLoader


def test_hook_dependancies():
    assert RecencyNeighborHook.requires == set()
    assert RecencyNeighborHook.produces == {
        'nids',
        'nbr_nids',
        'nbr_times',
        'nbr_feats',
        'times',
    }


def test_hook_reset_state():
    assert RecencyNeighborHook.has_state == True


def test_bad_neighbor_sampler_init():
    with pytest.raises(ValueError):
        RecencyNeighborHook(num_nbrs=[0], num_nodes=2, edge_feats_dim=1)
    with pytest.raises(ValueError):
        RecencyNeighborHook(num_nbrs=[-1], num_nodes=2, edge_feats_dim=1)
    with pytest.raises(ValueError):
        RecencyNeighborHook(num_nbrs=[], num_nodes=2, edge_feats_dim=1)


def _nbrs_2_np(batch: DGBatch) -> List[np.ndarray]:
    """Convert neighbors in batch to numpy arrays."""
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


@pytest.mark.skip('TODO: unskip')
def test_init_basic_sampled_graph_1_hop(basic_sample_graph):
    """The goal of this test is to provide a simple TG with 1-hop neighbors
    and test the basic functionality of the neighbor sampler.
    also make sure recency and uniform samplers return the same output.
    """
    dg = DGraph(basic_sample_graph)
    n_nbrs = [1]  # 1 neighbor for each node
    recency_hook = RecencyNeighborHook(
        num_nbrs=n_nbrs,
        num_nodes=dg.num_nodes,
        edge_feats_dim=dg.edge_feats_dim,
    )
    loader = DGDataLoader(dg, hook=[recency_hook], batch_size=1)
    assert loader._batch_size == 1

    batch_iter = iter(loader)
    batch_1 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_1)
    assert nids.shape == (1, 2)
    assert nids[0][0] == 0
    assert nids[0][1] == 1
    assert nbr_nids.shape == (1, 2, 1)
    assert nbr_nids[0][0][0] == PADDED_NODE_ID
    assert nbr_nids[0][1][0] == PADDED_NODE_ID
    assert nbr_times.shape == (1, 2, 1)
    assert nbr_times[0][0][0] == PADDED_NODE_ID
    assert nbr_times[0][1][0] == PADDED_NODE_ID
    assert nbr_feats.shape == (1, 2, 1, 1)  # 1 feature per edge
    assert nbr_feats[0][1][0][0] == nbr_feats[0][0][0][0] == PADDED_NODE_ID

    batch_2 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_2)
    assert nids.shape == (1, 2)
    assert nids[0][0] == 0
    assert nids[0][1] == 2
    assert nbr_nids.shape == (1, 2, 1)
    assert nbr_nids[0][0][0] == 1
    assert nbr_nids[0][1][0] == PADDED_NODE_ID
    assert nbr_times.shape == (1, 2, 1)
    assert nbr_times[0][0][0] == 1
    assert nbr_times[0][1][0] == PADDED_NODE_ID
    assert nbr_feats.shape == (1, 2, 1, 1)  # 1 feature per edge
    assert nbr_feats[0][0][0][0] == 1.0
    assert nbr_feats[0][1][0][0] == PADDED_NODE_ID

    batch_3 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_3)
    assert nids.shape == (1, 2)
    assert nids[0][0] == 2
    assert nids[0][1] == 3
    assert nbr_nids.shape == (1, 2, 1)
    assert nbr_nids[0][0][0] == 0
    assert nbr_nids[0][1][0] == PADDED_NODE_ID
    assert nbr_times.shape == (1, 2, 1)
    assert nbr_times[0][0][0] == 2
    assert nbr_times[0][1][0] == PADDED_NODE_ID
    assert nbr_feats.shape == (1, 2, 1, 1)  # 1 feature per edge
    assert nbr_feats[0][0][0][0] == 2.0
    assert nbr_feats[0][1][0][0] == PADDED_NODE_ID

    batch_4 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_4)
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


def _batch_eq_nbrs(batch_1: DGBatch, batch_2: DGBatch) -> bool:
    """Assert if two batches neighbors are equal."""
    torch.testing.assert_close(batch_1.nids, batch_2.nids)
    torch.testing.assert_close(batch_1.nbr_nids, batch_2.nbr_nids)
    torch.testing.assert_close(batch_1.nbr_times, batch_2.nbr_times)
    torch.testing.assert_close(batch_1.nbr_feats, batch_2.nbr_feats)
    return True


@pytest.mark.skip('TODO: unskip')
def test_recency_uniform_sampler_equivalence(basic_sample_graph):
    """The goal of this test is to test if uniform and recency sampler return the same output.
    when the buffer size is not exceeded for recency.
    """
    dg = DGraph(basic_sample_graph)
    n_nbrs = [1]  # 1 neighbor for each node
    recency_hook = RecencyNeighborHook(
        num_nbrs=n_nbrs,
        num_nodes=dg.num_nodes,
        edge_feats_dim=dg.edge_feats_dim,
    )
    recency_loader = DGDataLoader(dg, hook=[recency_hook], batch_size=1)
    uniform_hook = NeighborSamplerHook(num_nbrs=n_nbrs)
    uniform_loader = DGDataLoader(dg, hook=[uniform_hook], batch_size=1)
    assert recency_loader._batch_size == uniform_loader._batch_size == 1

    recency_iter = iter(recency_loader)
    uniform_iter = iter(uniform_loader)
    for idx in range(4):
        rbatch = next(recency_iter)
        ubatch = next(uniform_iter)
        assert _batch_eq_nbrs(rbatch, ubatch)

    rbatch = next(recency_iter)
    ubatch = next(uniform_iter)
    for _ in range(4):
        rbatch = next(recency_iter)
        ubatch = next(uniform_iter)
        assert _batch_eq_nbrs(rbatch, ubatch)


@pytest.fixture
def recency_buffer_graph():
    """Initializes the following graph.
    0 -> t=0 -> 1
    0 -> t=1 -> 2
    0 -> t=2 -> 3
    0 -> t=3 -> 4
    0 -> t=4 -> 5
    -- 100 edges --.
    """
    src = [0] * 100
    dst = list(range(1, 101))
    edge_index = [src, dst]
    edge_index = torch.LongTensor(edge_index)
    edge_index = edge_index.transpose(0, 1)
    edge_timestamps = torch.LongTensor(list(range(0, 100)))
    edge_feats = torch.LongTensor(
        list(range(1, 101))
    )  # edge feat is simply summing the node IDs at two end points
    edge_feats = edge_feats.view(-1, 1)  # 1 feature per edge
    data = DGData.from_raw(edge_timestamps, edge_index, edge_feats)
    return data


@pytest.mark.skip('TODO: unskip')
def test_recency_exceed_buffer(recency_buffer_graph):
    """The goal of this test is to test if the recency neighbor sampler would be able to update correctly when exceeding its max size.
    The test only has a single source node connecting to various destination nodes.
    """
    dg = DGraph(recency_buffer_graph)
    n_nbrs = [2]  # 2 neighbors for each node
    recency_hook = RecencyNeighborHook(
        num_nbrs=n_nbrs,
        num_nodes=dg.num_nodes,
        edge_feats_dim=dg.edge_feats_dim,
    )
    loader = DGDataLoader(dg, hook=[recency_hook], batch_size=2)
    assert loader._batch_size == 2

    batch_iter = iter(loader)
    batch_1 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats, _ = _nbrs_2_np(batch_1)
    assert nids.shape == (1, 4)
    assert nbr_nids.shape == (1, 4, 2)
    assert nbr_nids[0][0][0] == PADDED_NODE_ID
    assert nbr_nids[0][0][1] == PADDED_NODE_ID
    assert nbr_times.shape == (1, 4, 2)
    assert nbr_times[0][0][0] == PADDED_NODE_ID
    assert nbr_times[0][0][1] == PADDED_NODE_ID
    assert nbr_feats.shape == (1, 4, 2, 1)  # 1 feature per edge

    batch_2 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats, _ = _nbrs_2_np(batch_2)
    assert nids.shape == (1, 4)
    assert nbr_nids.shape == (1, 4, 2)
    assert nbr_nids[0][0][0] == 1
    assert nbr_nids[0][0][1] == 2
    assert nbr_times.shape == (1, 4, 2)
    assert nbr_times[0][0][0] == 0
    assert nbr_times[0][0][1] == 1
    assert nbr_feats.shape == (1, 4, 2, 1)  # 1 feature per edge

    for batch in batch_iter:
        nids, nbr_nids, nbr_times, nbr_feats, _ = _nbrs_2_np(batch)
        assert nbr_nids.shape == (1, 4, 2)
        assert nbr_times.shape == (1, 4, 2)
        assert nbr_nids[0][0][0] == nbr_times[0][0][0] + 1
        assert nbr_nids[0][0][1] == nbr_times[0][0][1] + 1
        assert nbr_feats[0][0][0][0] == nbr_times[0][0][0] + 1
        assert nbr_feats[0][0][1][0] == nbr_times[0][0][1] + 1


@pytest.fixture
def two_hop_basic_graph():
    """Initializes the following 2 hop graph.

    0 -> t=1 -> 1
                |
                v
              t=2
                |
                v
    3 -> t=3 -> 2
    4 -> t=4 -> 2
    5 -> t=5 -> 0
    5 -> t=6 -> 2
    """
    edge_index = torch.LongTensor([[0, 1], [1, 2], [3, 2], [4, 2], [5, 0], [5, 2]])
    edge_timestamps = torch.LongTensor([1, 2, 3, 4, 5, 6])
    edge_feats = torch.LongTensor(
        [[1], [3], [5], [6], [5], [7]]
    )  # edge feat is simply summing the node IDs at two end points
    data = DGData.from_raw(edge_timestamps, edge_index, edge_feats)
    return data


@pytest.mark.skip('TODO: unskip')
def test_2_hop_graph(two_hop_basic_graph):
    dg = DGraph(two_hop_basic_graph)
    n_nbrs = [1, 1]  # 1 neighbor for each node
    recency_hook = RecencyNeighborHook(
        num_nbrs=n_nbrs,
        num_nodes=dg.num_nodes,
        edge_feats_dim=dg.edge_feats_dim,
    )
    loader = DGDataLoader(dg, hook=[recency_hook], batch_size=1)
    assert loader._batch_size == 1

    batch_iter = iter(loader)
    batch_1 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_1)
    assert nids.shape == (2, 2)  # 2 hop, each has 2 node
    assert nids[0][0] == 0
    assert nids[0][1] == 1
    assert nbr_nids.shape == (2, 2, 1)
    assert nbr_nids[0][0][0] == PADDED_NODE_ID
    assert nbr_nids[1][0][0] == PADDED_NODE_ID
    assert nbr_times.shape == (2, 2, 1)
    assert nbr_times[0][0][0] == PADDED_NODE_ID
    assert nbr_times[1][0][0] == PADDED_NODE_ID
    assert nbr_feats.shape == (2, 2, 1, 1)  # 1 feature per edge

    batch_2 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_2)
    assert nids.shape == (2, 2)
    assert nbr_nids.shape == (2, 2, 1)
    assert nbr_times.shape == (2, 2, 1)
    assert nbr_feats.shape == (2, 2, 1, 1)  # 1 feature per edge
    assert nbr_nids[0][0][0] == 0  # first hop, node 1 has neighbor 0
    assert nbr_nids[1][0][0] == PADDED_NODE_ID  # no second hop neighbors
    assert nbr_nids[0][1][0] == PADDED_NODE_ID
    assert nbr_nids[1][1][0] == PADDED_NODE_ID

    batch_3 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_3)
    assert nids.shape == (2, 2)
    assert nbr_nids.shape == (2, 2, 1)
    assert nbr_times.shape == (2, 2, 1)
    assert nbr_feats.shape == (2, 2, 1, 1)  # 1 feature per edge
    assert nbr_nids[0][0][0] == PADDED_NODE_ID  # first hop, node 3 has no neighbor yet
    assert nbr_nids[0][1][0] == 1  # first hop, node 2 has neighbor 1
    assert nbr_nids[1][0][0] == PADDED_NODE_ID
    assert nbr_nids[1][1][0] == 0  # second hop, node 2 has neighbor 0

    batch_4 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_4)
    assert nids.shape == (2, 2)
    assert nbr_nids.shape == (2, 2, 1)
    assert nbr_times.shape == (2, 2, 1)
    assert nbr_feats.shape == (2, 2, 1, 1)  # 1 feature per edge
    assert nbr_nids[0][0][0] == PADDED_NODE_ID  # first hop, node 4 has no neighbor yet
    assert (
        nbr_nids[0][1][0] == 3
    )  # first hop, node 2 has neighbor 3 (replaced 1 as it is pushed out of cache)
    assert (
        nbr_nids[1][1][0] == PADDED_NODE_ID
    )  # second hop, node 2 has no neighbor now (as 1 is pushed out of cache)

    batch_5 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_5)
    assert nids.shape == (2, 2)
    assert nbr_nids.shape == (2, 2, 1)
    assert nbr_times.shape == (2, 2, 1)
    assert nbr_feats.shape == (2, 2, 1, 1)  # 1 feature per edge
    assert nbr_nids[0][0][0] == PADDED_NODE_ID
    assert nbr_nids[1][0][0] == PADDED_NODE_ID
    assert nbr_nids[0][1][0] == 1
    assert nbr_nids[1][1][0] == PADDED_NODE_ID

    batch_6 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_6)
    assert nids.shape == (2, 2)
    assert nbr_nids.shape == (2, 2, 1)
    assert nbr_times.shape == (2, 2, 1)
    assert nbr_feats.shape == (2, 2, 1, 1)  # 1 feature per edge
    assert nbr_nids[0][0][0] == 0  # node 5 first hop has neighbor 0
    assert nbr_nids[1][0][0] == 1  # node 5 second hop has neighbor 1
    assert nbr_nids[0][1][0] == 4  # node 2 first hop has neighbor 4
    assert nbr_nids[1][1][0] == PADDED_NODE_ID  # node 2 second hop has no neighbor


class FakeNegSampler:
    def query_batch(self, src, dst, time, split_mode='val'):
        return []


@pytest.mark.skip(
    'TODO: add option that set seed time to always be the current query time instead of strictly following the edge timestamp of 1 hop. This checks for non-time respecting path. TO DO'
)
def test_tgb_non_time_respecting_negative_neighbor_sampling_test(two_hop_basic_graph):
    dg = DGraph(two_hop_basic_graph)
    mock_sampler = Mock(spec=FakeNegSampler)
    mock_sampler.eval_set = {}
    mock_sampler.eval_set['val'] = {}
    neg_batch_list = [[2, 3, 4, 5]]
    mock_sampler.query_batch.return_value = neg_batch_list
    tgb_hook = TGBNegativeEdgeSamplerHook(neg_sampler=mock_sampler, split_mode='val')
    n_nbrs = [1, 1]  # 1 neighbor for each node
    recency_hook = RecencyNeighborHook(
        num_nbrs=n_nbrs,
        num_nodes=dg.num_nodes,
        edge_feats_dim=dg.edge_feats_dim,
    )
    loader = DGDataLoader(dg, hook=[tgb_hook, recency_hook], batch_size=1)
    assert loader._batch_size == 1

    batch_iter = iter(loader)
    batch_1 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_1)
    assert nids.shape == (2, 6)  # 2 hop, each has 2 node
    assert nids[0][0] == 0
    assert nids[0][1] == 1
    assert nids[0][2] == 2
    assert nids[0][3] == 3
    assert nids[0][4] == 4
    assert nids[0][5] == 5
    assert nbr_nids.shape == (2, 6, 1)
    assert nbr_nids[0][0][0] == PADDED_NODE_ID
    assert nbr_nids[1][0][0] == PADDED_NODE_ID
    assert nbr_times.shape == (2, 6, 1)
    assert nbr_times[0][0][0] == PADDED_NODE_ID
    assert nbr_times[1][0][0] == PADDED_NODE_ID
    assert nbr_feats.shape == (2, 6, 1, 1)  # 1 feature per edge

    neg_batch_list = [[0, 3, 4, 5]]
    mock_sampler.query_batch.return_value = neg_batch_list
    batch_2 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_2)
    assert nbr_nids[0][0][0] == 0  # first hop, node 1 has neighbor 0
    assert nbr_nids[1][0][0] == PADDED_NODE_ID  # no second hop neighbors
    assert nbr_nids[0][1][0] == PADDED_NODE_ID
    assert nbr_nids[1][1][0] == PADDED_NODE_ID
    assert nbr_nids[0][2][0] == 1
    assert nbr_nids[0][3][0] == nbr_nids[0][4][0] == nbr_nids[0][5][0] == PADDED_NODE_ID

    neg_batch_list = [[0, 1, 4, 5]]
    mock_sampler.query_batch.return_value = neg_batch_list
    batch_3 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_3)
    assert nbr_nids[0][0][0] == PADDED_NODE_ID  # first hop, node 3 has no neighbor yet
    assert nbr_nids[0][1][0] == 1  # first hop, node 2 has neighbor 1
    assert nbr_nids[1][0][0] == PADDED_NODE_ID
    assert nbr_nids[1][1][0] == 0  # second hop, node 2 has neighbor 0
    assert nbr_nids[0][2][0] == 1
    assert nbr_nids[1][2][0] == 2
    assert nbr_nids[0][3][0] == 2
    assert nbr_nids[1][3][0] == PADDED_NODE_ID
    assert nbr_nids[0][4][0] == PADDED_NODE_ID
    assert nbr_nids[0][5][0] == PADDED_NODE_ID

    neg_batch_list = [[0, 1, 3, 5]]
    mock_sampler.query_batch.return_value = neg_batch_list
    batch_4 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_4)
    assert (
        nbr_nids[0][1][0] == 3
    )  # first hop, node 2 has neighbor 3 (replaced 1 as it is pushed out of cache)
    assert (
        nbr_nids[1][1][0] == PADDED_NODE_ID
    )  # second hop, node 2 has no neighbor now (as 1 is pushed out of cache)
    assert nbr_nids[0][2][0] == 1
    assert nbr_nids[1][2][0] == 2
    assert nbr_nids[0][3][0] == 0
    assert nbr_nids[1][3][0] == PADDED_NODE_ID
    assert nbr_nids[0][4][0] == 2
    assert nbr_nids[1][4][0] == 1
    assert nbr_nids[0][5][0] == PADDED_NODE_ID
    assert nbr_nids[1][5][0] == PADDED_NODE_ID

    neg_batch_list = [[1, 2, 3, 4]]
    mock_sampler.query_batch.return_value = neg_batch_list
    batch_5 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_5)
    assert nbr_nids[0][0][0] == PADDED_NODE_ID
    assert nbr_nids[1][0][0] == PADDED_NODE_ID
    assert nbr_nids[0][1][0] == 1
    assert nbr_nids[1][1][0] == PADDED_NODE_ID
    assert nbr_nids[0][2][0] == 0
    assert nbr_nids[0][3][0] == 1
    assert nbr_nids[0][4][0] == 2
    assert nbr_nids[1][4][0] == 4
    assert nbr_nids[0][5][0] == 2

    neg_batch_list = [[0, 1, 3, 4]]
    mock_sampler.query_batch.return_value = neg_batch_list
    batch_6 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_6)
    assert nbr_nids[0][0][0] == 0  # node 5 first hop has neighbor 0
    assert nbr_nids[1][0][0] == 1  # node 5 second hop has neighbor 1
    assert nbr_nids[0][1][0] == 4  # node 2 first hop has neighbor 4
    assert nbr_nids[1][1][0] == PADDED_NODE_ID  # node 2 second hop has no neighbor
    assert nbr_nids[0][2][0] == 5
    assert nbr_nids[0][3][0] == 2
    assert nbr_nids[0][4][0] == 2
    assert nbr_nids[1][4][0] == 4
    assert nbr_nids[0][5][0] == 2


@pytest.mark.skip('TODO: move to a different sampler that respects time')
def test_tgb_time_respecting_negative_neighbor_sampling_test(two_hop_basic_graph):
    dg = DGraph(two_hop_basic_graph)
    mock_sampler = Mock(spec=FakeNegSampler)
    mock_sampler.eval_set = {}
    mock_sampler.eval_set['val'] = {}
    neg_batch_list = [[2, 3, 4, 5]]
    mock_sampler.query_batch.return_value = neg_batch_list
    tgb_hook = TGBNegativeEdgeSamplerHook(neg_sampler=mock_sampler, split_mode='val')
    n_nbrs = [1, 1]  # 1 neighbor for each node
    recency_hook = RecencyNeighborHook(
        num_nbrs=n_nbrs,
        num_nodes=dg.num_nodes,
        edge_feats_dim=dg.edge_feats_dim,
    )
    loader = DGDataLoader(dg, hook=[tgb_hook, recency_hook], batch_size=1)
    assert loader._batch_size == 1

    batch_iter = iter(loader)
    batch_1 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_1)
    assert nids.shape == (2, 6)  # 2 hop, each has 2 node
    assert nids[0][0] == 0
    assert nids[0][1] == 1
    assert nids[0][2] == 2
    assert nids[0][3] == 3
    assert nids[0][4] == 4
    assert nids[0][5] == 5
    assert nbr_nids.shape == (2, 6, 1)
    assert nbr_nids[0][0][0] == PADDED_NODE_ID
    assert nbr_nids[1][0][0] == PADDED_NODE_ID
    assert nbr_times.shape == (2, 6, 1)
    assert nbr_times[0][0][0] == PADDED_NODE_ID
    assert nbr_times[1][0][0] == PADDED_NODE_ID
    assert nbr_feats.shape == (2, 6, 1, 1)  # 1 feature per edge

    neg_batch_list = [[0, 3, 4, 5]]
    mock_sampler.query_batch.return_value = neg_batch_list
    batch_2 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_2)
    assert nbr_nids[0][0][0] == 0  # first hop, node 1 has neighbor 0
    assert nbr_nids[1][0][0] == PADDED_NODE_ID  # no second hop neighbors
    assert nbr_nids[0][1][0] == PADDED_NODE_ID
    assert nbr_nids[1][1][0] == PADDED_NODE_ID
    assert nbr_nids[0][2][0] == 1
    assert nbr_nids[0][3][0] == nbr_nids[0][4][0] == nbr_nids[0][5][0] == PADDED_NODE_ID

    neg_batch_list = [[0, 1, 4, 5]]
    mock_sampler.query_batch.return_value = neg_batch_list
    batch_3 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_3)
    assert nbr_nids[0][0][0] == PADDED_NODE_ID  # first hop, node 3 has no neighbor yet
    assert nbr_nids[0][1][0] == 1  # first hop, node 2 has neighbor 1
    assert nbr_nids[1][0][0] == PADDED_NODE_ID
    assert nbr_nids[1][1][0] == 0  # second hop, node 2 has neighbor 0
    assert nbr_nids[0][2][0] == 1
    assert nbr_nids[1][2][0] == PADDED_NODE_ID
    assert nbr_nids[0][3][0] == 2
    assert nbr_nids[1][3][0] == PADDED_NODE_ID
    assert nbr_nids[0][4][0] == PADDED_NODE_ID
    assert nbr_nids[0][5][0] == PADDED_NODE_ID

    neg_batch_list = [[0, 1, 3, 5]]
    mock_sampler.query_batch.return_value = neg_batch_list
    batch_4 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_4)
    assert (
        nbr_nids[0][1][0] == 3
    )  # first hop, node 2 has neighbor 3 (replaced 1 as it is pushed out of cache)
    assert (
        nbr_nids[1][1][0] == PADDED_NODE_ID
    )  # second hop, node 2 has no neighbor now (as 1 is pushed out of cache)
    assert nbr_nids[0][2][0] == 1
    assert nbr_nids[1][2][0] == PADDED_NODE_ID
    assert nbr_nids[0][3][0] == 2
    assert nbr_nids[1][3][0] == PADDED_NODE_ID
    assert nbr_nids[0][4][0] == 2
    assert nbr_nids[1][4][0] == 1
    assert nbr_nids[0][5][0] == PADDED_NODE_ID
    assert nbr_nids[1][5][0] == PADDED_NODE_ID

    neg_batch_list = [[1, 2, 3, 4]]
    mock_sampler.query_batch.return_value = neg_batch_list
    batch_5 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_5)
    assert nbr_nids[0][0][0] == PADDED_NODE_ID
    assert nbr_nids[1][0][0] == PADDED_NODE_ID
    assert nbr_nids[0][1][0] == 1
    assert nbr_nids[1][1][0] == PADDED_NODE_ID
    assert nbr_nids[0][2][0] == 2
    assert nbr_nids[0][3][0] == 4
    assert nbr_nids[0][4][0] == 2
    assert nbr_nids[1][4][0] == 1
    assert nbr_nids[0][5][0] == 2

    neg_batch_list = [[0, 1, 3, 4]]
    mock_sampler.query_batch.return_value = neg_batch_list
    batch_6 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_6)
    assert nbr_nids[0][0][0] == 0  # node 5 first hop has neighbor 0
    assert nbr_nids[1][0][0] == 1  # node 5 second hop has neighbor 1
    assert nbr_nids[0][1][0] == 4  # node 2 first hop has neighbor 4
    assert nbr_nids[1][1][0] == PADDED_NODE_ID  # node 2 second hop has no neighbor
    assert nbr_nids[0][2][0] == 5
    assert nbr_nids[0][3][0] == 2
    assert nbr_nids[0][4][0] == 2
    assert nbr_nids[1][4][0] == 1
    assert nbr_nids[0][5][0] == 2
