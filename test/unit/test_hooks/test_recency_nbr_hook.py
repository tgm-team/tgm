from typing import List
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from tgm import DGBatch, DGraph
from tgm.constants import PADDED_NODE_ID
from tgm.data import DGData, DGDataLoader
from tgm.hooks import RecencyNeighborHook, TGBNegativeEdgeSamplerHook
from tgm.hooks.hook_manager import HookManager

_PADDED_TIME_ID = 0
_PADDED_FEAT_ID = 0.0


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
    edge_index = torch.IntTensor([[0, 1], [0, 2], [2, 3], [2, 0]])
    edge_timestamps = torch.LongTensor([1, 2, 3, 4])
    edge_feats = torch.Tensor(
        [[1], [2], [5], [2]]
    )  # edge feat is simply summing the node IDs at two end points
    data = DGData.from_raw(edge_timestamps, edge_index, edge_feats)
    return data


def test_hook_dependancies():
    assert RecencyNeighborHook.requires == set()
    assert RecencyNeighborHook.produces == {
        'nids',
        'nbr_nids',
        'nbr_times',
        'nbr_feats',
        'times',
        'seed_node_nbr_mask',
    }


def test_mock_move_queues_to_device(basic_sample_graph):
    dg = DGraph(basic_sample_graph)
    hook = RecencyNeighborHook(
        num_nbrs=[1],
        num_nodes=dg.num_nodes,
        seed_nodes_keys=['src'],
        seed_times_keys=['time'],
    )
    batch = dg.materialize()
    hook._device = 'foo'  # Patch graph device to trigger queue movement
    batch = hook(dg, batch)


def test_hook_reset_state(basic_sample_graph):
    assert RecencyNeighborHook.has_state == True

    dg = DGraph(basic_sample_graph)
    n_nbrs = [1]  # 1 neighbor for each node
    recency_hook = RecencyNeighborHook(
        num_nbrs=n_nbrs,
        num_nodes=dg.num_nodes,
        seed_nodes_keys=['src', 'dst'],
        seed_times_keys=['time', 'time'],
    )
    hm = HookManager(keys=['unit'])
    hm.register('unit', recency_hook)
    hm.set_active_hooks('unit')
    loader = DGDataLoader(dg, hook_manager=hm, batch_size=1)
    assert loader._batch_size == 1

    # Iterate the entire graph, reset state, then ensure the second iteration
    # matches expected output as if the hook was freshly initialized
    for _ in loader:
        continue

    hm.reset_state()

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
    assert nbr_times[0][0][0] == _PADDED_TIME_ID
    assert nbr_times[0][1][0] == _PADDED_TIME_ID
    assert nbr_feats.shape == (1, 2, 1, 1)  # 1 feature per edge
    assert nbr_feats[0][1][0][0] == nbr_feats[0][0][0][0] == _PADDED_FEAT_ID

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
    assert nbr_times[0][1][0] == _PADDED_TIME_ID
    assert nbr_feats.shape == (1, 2, 1, 1)  # 1 feature per edge
    assert nbr_feats[0][0][0][0] == 1.0
    assert nbr_feats[0][1][0][0] == _PADDED_FEAT_ID

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
    assert nbr_times[0][1][0] == _PADDED_TIME_ID
    assert nbr_feats.shape == (1, 2, 1, 1)  # 1 feature per edge
    assert nbr_feats[0][0][0][0] == 2.0
    assert nbr_feats[0][1][0][0] == _PADDED_FEAT_ID

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


def test_bad_neighbor_sampler_init():
    with pytest.raises(ValueError):
        RecencyNeighborHook(
            num_nbrs=[0], num_nodes=2, seed_nodes_keys=['foo'], seed_times_keys=['bar']
        )
    with pytest.raises(ValueError):
        RecencyNeighborHook(
            num_nbrs=[-1], num_nodes=2, seed_nodes_keys=['foo'], seed_times_keys=['bar']
        )
    with pytest.raises(ValueError):
        RecencyNeighborHook(
            num_nbrs=[], num_nodes=2, seed_nodes_keys=['foo'], seed_times_keys=['bar']
        )
    with pytest.raises(ValueError):
        RecencyNeighborHook(
            num_nbrs=[1],
            num_nodes=2,
            seed_nodes_keys=['foo', 'bar'],
            seed_times_keys=['foo'],
        )


def test_sample_with_node_events_seeds(node_only_data):
    dg = DGraph(node_only_data)
    hook = RecencyNeighborHook(
        num_nbrs=[1],
        num_nodes=dg.num_nodes,
        seed_nodes_keys=['node_ids'],
        seed_times_keys=['node_times'],
    )
    batch = dg.materialize()
    batch = hook(dg, batch)
    assert len(batch.nids) == 1
    assert len(batch.times) == 1
    torch.testing.assert_close(batch.nids[0], batch.node_ids)
    torch.testing.assert_close(batch.times[0], batch.node_times)


def test_bad_sample_with_non_existent_seeds(basic_sample_graph):
    dg = DGraph(basic_sample_graph)
    hook = RecencyNeighborHook(
        num_nbrs=[1], num_nodes=2, seed_nodes_keys=['foo'], seed_times_keys=['bar']
    )
    batch = dg.materialize()

    with pytest.raises(ValueError):
        _ = hook(dg, batch)


def test_sample_with_none_seeds(basic_sample_graph):
    dg = DGraph(basic_sample_graph)
    hook = RecencyNeighborHook(
        num_nbrs=[1], num_nodes=2, seed_nodes_keys=['foo'], seed_times_keys=['bar']
    )
    batch = dg.materialize()
    batch.foo, batch.bar = None, None

    with pytest.warns(UserWarning):
        batch = hook(dg, batch)


def test_bad_sample_with_non_tensor_non_None_seeds(basic_sample_graph):
    dg = DGraph(basic_sample_graph)
    hook = RecencyNeighborHook(
        num_nbrs=[1], num_nodes=2, seed_nodes_keys=['foo'], seed_times_keys=['bar']
    )
    batch = dg.materialize()
    batch.foo = 'should_be_1d_tensor'
    batch.bar = 'should_be_1d_tensor'

    with pytest.raises(ValueError):
        batch = hook(dg, batch)


def test_bad_sample_with_non_1d_tensor_seeds(basic_sample_graph):
    dg = DGraph(basic_sample_graph)
    hook = RecencyNeighborHook(
        num_nbrs=[1], num_nodes=2, seed_nodes_keys=['foo'], seed_times_keys=['bar']
    )
    batch = dg.materialize()
    batch.foo = torch.rand(2, 3)  # should be 1-d
    batch.bar = torch.rand(2, 3)  # should be 1-d

    with pytest.raises(ValueError):
        batch = hook(dg, batch)


def test_bad_sample_with_seeds_id_out_of_range(basic_sample_graph):
    dg = DGraph(basic_sample_graph)
    hook = RecencyNeighborHook(
        num_nbrs=[1], num_nodes=2, seed_nodes_keys=['foo'], seed_times_keys=['bar']
    )
    batch = dg.materialize()
    batch.foo = torch.IntTensor([-1])  # should be positive
    batch.bar = torch.LongTensor([1])

    with pytest.raises(ValueError):
        batch = hook(dg, batch)

    batch.foo = torch.IntTensor([3])  # cannot be > num_nodes
    with pytest.raises(ValueError):
        batch = hook(dg, batch)


def test_bad_sample_with_seeds_time_out_of_range(basic_sample_graph):
    dg = DGraph(basic_sample_graph)
    hook = RecencyNeighborHook(
        num_nbrs=[1], num_nodes=2, seed_nodes_keys=['foo'], seed_times_keys=['bar']
    )
    batch = dg.materialize()
    batch.foo = torch.IntTensor([1])
    batch.bar = torch.LongTensor([-1])  # should be positive

    with pytest.raises(ValueError):
        batch = hook(dg, batch)


def _nbrs_2_np(batch: DGBatch) -> List[np.ndarray]:
    assert isinstance(batch, DGBatch)
    assert hasattr(batch, 'nids')
    assert hasattr(batch, 'nbr_nids')
    assert hasattr(batch, 'nbr_times')
    assert hasattr(batch, 'nbr_feats')
    assert hasattr(batch, 'seed_node_nbr_mask')

    nids = np.array(batch.nids)
    nbr_nids = np.array(batch.nbr_nids)
    nbr_times = np.array(batch.nbr_times)
    nbr_feats = np.array(batch.nbr_feats)
    return [nids, nbr_nids, nbr_times, nbr_feats]


def test_init_basic_sampled_graph_1_hop(basic_sample_graph):
    dg = DGraph(basic_sample_graph)
    n_nbrs = [1]  # 1 neighbor for each node
    recency_hook = RecencyNeighborHook(
        num_nbrs=n_nbrs,
        num_nodes=dg.num_nodes,
        seed_nodes_keys=['src', 'dst'],
        seed_times_keys=['time', 'time'],
    )
    hm = HookManager(keys=['unit'])
    hm.register('unit', recency_hook)
    hm.set_active_hooks('unit')
    loader = DGDataLoader(dg, hook_manager=hm, batch_size=1)
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
    assert nbr_times[0][0][0] == _PADDED_TIME_ID
    assert nbr_times[0][1][0] == _PADDED_TIME_ID
    assert nbr_feats.shape == (1, 2, 1, 1)  # 1 feature per edge
    assert nbr_feats[0][1][0][0] == nbr_feats[0][0][0][0] == _PADDED_FEAT_ID

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
    assert nbr_times[0][1][0] == _PADDED_TIME_ID
    assert nbr_feats.shape == (1, 2, 1, 1)  # 1 feature per edge
    assert nbr_feats[0][0][0][0] == 1.0
    assert nbr_feats[0][1][0][0] == _PADDED_FEAT_ID

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
    assert nbr_times[0][1][0] == _PADDED_TIME_ID
    assert nbr_feats.shape == (1, 2, 1, 1)  # 1 feature per edge
    assert nbr_feats[0][0][0][0] == 2.0
    assert nbr_feats[0][1][0][0] == _PADDED_FEAT_ID

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


def test_init_basic_sampled_graph_directed_1_hop(basic_sample_graph):
    dg = DGraph(basic_sample_graph)
    n_nbrs = [1]  # 1 neighbor for each node
    recency_hook = RecencyNeighborHook(
        num_nbrs=n_nbrs,
        num_nodes=dg.num_nodes,
        seed_nodes_keys=['src', 'dst'],
        seed_times_keys=['time', 'time'],
        directed=True,
    )
    hm = HookManager(keys=['unit'])
    hm.register('unit', recency_hook)
    hm.set_active_hooks('unit')
    loader = DGDataLoader(dg, hook_manager=hm, batch_size=1)
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
    assert nbr_times[0][0][0] == _PADDED_TIME_ID
    assert nbr_times[0][1][0] == _PADDED_TIME_ID
    assert nbr_feats.shape == (1, 2, 1, 1)  # 1 feature per edge
    assert nbr_feats[0][1][0][0] == nbr_feats[0][0][0][0] == _PADDED_FEAT_ID

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
    assert nbr_times[0][1][0] == _PADDED_TIME_ID
    assert nbr_feats.shape == (1, 2, 1, 1)  # 1 feature per edge
    assert nbr_feats[0][0][0][0] == 1.0
    assert nbr_feats[0][1][0][0] == _PADDED_FEAT_ID

    batch_3 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_3)
    assert nids.shape == (1, 2)
    assert nids[0][0] == 2
    assert nids[0][1] == 3
    assert nbr_nids.shape == (1, 2, 1)
    assert nbr_nids[0][0][0] == PADDED_NODE_ID
    assert nbr_nids[0][1][0] == PADDED_NODE_ID
    assert nbr_times.shape == (1, 2, 1)
    assert nbr_times[0][0][0] == _PADDED_TIME_ID
    assert nbr_times[0][1][0] == _PADDED_TIME_ID
    assert nbr_feats.shape == (1, 2, 1, 1)  # 1 feature per edge
    assert nbr_feats[0][0][0][0] == _PADDED_FEAT_ID
    assert nbr_feats[0][1][0][0] == _PADDED_FEAT_ID

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
    edge_index = torch.IntTensor(edge_index)
    edge_index = edge_index.transpose(0, 1)
    edge_timestamps = torch.LongTensor(list(range(0, 100)))
    edge_feats = torch.Tensor(
        list(range(1, 101))
    )  # edge feat is simply summing the node IDs at two end points
    edge_feats = edge_feats.view(-1, 1)  # 1 feature per edge
    data = DGData.from_raw(edge_timestamps, edge_index, edge_feats)
    return data


def test_recency_exceed_buffer(recency_buffer_graph):
    dg = DGraph(recency_buffer_graph)
    n_nbrs = [2]  # 2 neighbors for each node
    recency_hook = RecencyNeighborHook(
        num_nbrs=n_nbrs,
        num_nodes=dg.num_nodes,
        seed_nodes_keys=['src', 'dst'],
        seed_times_keys=['time', 'time'],
    )
    hm = HookManager(keys=['unit'])
    hm.register('unit', recency_hook)
    hm.set_active_hooks('unit')
    loader = DGDataLoader(dg, hook_manager=hm, batch_size=2)
    assert loader._batch_size == 2

    batch_iter = iter(loader)
    batch_1 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_1)
    assert nids.shape == (1, 4)
    assert nbr_nids.shape == (1, 4, 2)
    assert nbr_nids[0][0][0] == PADDED_NODE_ID
    assert nbr_nids[0][0][1] == PADDED_NODE_ID
    assert nbr_times.shape == (1, 4, 2)
    assert nbr_times[0][0][0] == _PADDED_TIME_ID
    assert nbr_times[0][0][1] == _PADDED_TIME_ID
    assert nbr_feats.shape == (1, 4, 2, 1)  # 1 feature per edge

    batch_2 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_2)
    assert nids.shape == (1, 4)
    assert nbr_nids.shape == (1, 4, 2)
    assert nbr_nids[0][0][0] == 1
    assert nbr_nids[0][0][1] == 2
    assert nbr_times.shape == (1, 4, 2)
    assert nbr_times[0][0][0] == _PADDED_TIME_ID
    assert nbr_times[0][0][1] == 1
    assert nbr_feats.shape == (1, 4, 2, 1)  # 1 feature per edge

    for batch in batch_iter:
        nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch)
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
    edge_index = torch.IntTensor([[0, 1], [1, 2], [3, 2], [4, 2], [5, 0], [5, 2]])
    edge_timestamps = torch.LongTensor([1, 2, 3, 4, 5, 6])
    edge_feats = torch.Tensor(
        [[1], [3], [5], [6], [5], [7]]
    )  # edge feat is simply summing the node IDs at two end points
    data = DGData.from_raw(edge_timestamps, edge_index, edge_feats)
    return data


def test_2_hop_graph(two_hop_basic_graph):
    dg = DGraph(two_hop_basic_graph)
    n_nbrs = [1, 1]  # 1 neighbor for each node
    recency_hook = RecencyNeighborHook(
        num_nbrs=n_nbrs,
        num_nodes=dg.num_nodes,
        seed_nodes_keys=['src', 'dst'],
        seed_times_keys=['time', 'time'],
    )
    hm = HookManager(keys=['unit'])
    hm.register('unit', recency_hook)
    hm.set_active_hooks('unit')
    loader = DGDataLoader(dg, hook_manager=hm, batch_size=1)
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
    assert nbr_times[0][0][0] == _PADDED_TIME_ID
    assert nbr_times[1][0][0] == _PADDED_TIME_ID
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
    assert nbr_nids[1][1][0] == PADDED_NODE_ID  # second hop, node 2 has neighbor 0

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
    assert nbr_nids[1][0][0] == PADDED_NODE_ID  # node 5 second hop has neighbor 1
    assert nbr_nids[0][1][0] == 4  # node 2 first hop has neighbor 4
    assert nbr_nids[1][1][0] == PADDED_NODE_ID  # node 2 second hop has no neighbor


def test_2_hop_directed_graph(two_hop_basic_graph):
    dg = DGraph(two_hop_basic_graph)
    n_nbrs = [1, 1]  # 1 neighbor for each node
    recency_hook = RecencyNeighborHook(
        num_nbrs=n_nbrs,
        num_nodes=dg.num_nodes,
        seed_nodes_keys=['src', 'dst'],
        seed_times_keys=['time', 'time'],
        directed=True,
    )
    hm = HookManager(keys=['unit'])
    hm.register('unit', recency_hook)
    hm.set_active_hooks('unit')
    loader = DGDataLoader(dg, hook_manager=hm, batch_size=1)
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
    assert nbr_times[0][0][0] == _PADDED_TIME_ID
    assert nbr_times[1][0][0] == _PADDED_TIME_ID
    assert nbr_feats.shape == (2, 2, 1, 1)  # 1 feature per edge

    batch_2 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_2)
    assert nids.shape == (2, 2)
    assert nbr_nids.shape == (2, 2, 1)
    assert nbr_times.shape == (2, 2, 1)
    assert nbr_feats.shape == (2, 2, 1, 1)  # 1 feature per edge
    assert nbr_nids[0][0][0] == PADDED_NODE_ID
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
    assert nbr_nids[0][1][0] == PADDED_NODE_ID
    assert nbr_nids[1][0][0] == PADDED_NODE_ID
    assert nbr_nids[1][1][0] == PADDED_NODE_ID  # second hop, node 2 has neighbor 0

    batch_4 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_4)
    assert nids.shape == (2, 2)
    assert nbr_nids.shape == (2, 2, 1)
    assert nbr_times.shape == (2, 2, 1)
    assert nbr_feats.shape == (2, 2, 1, 1)  # 1 feature per edge
    assert nbr_nids[0][0][0] == PADDED_NODE_ID  # first hop, node 4 has no neighbor yet
    assert nbr_nids[0][1][0] == PADDED_NODE_ID
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
    assert nbr_nids[1][0][0] == 1
    assert nbr_nids[0][1][0] == PADDED_NODE_ID
    assert nbr_nids[1][1][0] == PADDED_NODE_ID


@patch('tgb.linkproppred.negative_sampler.NegativeEdgeSampler')
def test_tgb_non_time_respecting_negative_neighbor_sampling_test(
    MockNegSampler, two_hop_basic_graph
):
    dg = DGraph(two_hop_basic_graph)
    neg_batch_list = [[2, 3, 4, 5]]

    mock_sampler = Mock()
    mock_sampler.eval_set = {'val': {}}
    mock_sampler.query_batch.return_value = neg_batch_list
    MockNegSampler.return_value = mock_sampler

    tgb_hook = TGBNegativeEdgeSamplerHook(dataset_name='foo', split_mode='val')

    n_nbrs = [1, 1]  # 1 neighbor for each node
    recency_hook = RecencyNeighborHook(
        num_nbrs=n_nbrs,
        num_nodes=dg.num_nodes,
        seed_nodes_keys=['src', 'dst', 'neg'],
        seed_times_keys=['time', 'time', 'neg_time'],
    )
    hm = HookManager(keys=['unit'])
    hm.register('unit', tgb_hook)
    hm.register('unit', recency_hook)
    hm.set_active_hooks('unit')
    loader = DGDataLoader(dg, hook_manager=hm, batch_size=1)
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
    assert nbr_times[0][0][0] == _PADDED_TIME_ID
    assert nbr_times[1][0][0] == _PADDED_TIME_ID
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
    assert nbr_nids[1][1][0] == PADDED_NODE_ID  # second hop, node 2 has neighbor 0
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
    assert nbr_nids[1][4][0] == PADDED_NODE_ID
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
    assert nbr_nids[1][4][0] == PADDED_NODE_ID
    assert nbr_nids[0][5][0] == 2

    neg_batch_list = [[0, 1, 3, 4]]
    mock_sampler.query_batch.return_value = neg_batch_list
    batch_6 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_6)
    assert nbr_nids[0][0][0] == 0  # node 5 first hop has neighbor 0
    assert nbr_nids[1][0][0] == PADDED_NODE_ID
    assert nbr_nids[0][1][0] == 4  # node 2 first hop has neighbor 4
    assert nbr_nids[1][1][0] == PADDED_NODE_ID  # node 2 second hop has no neighbor
    assert nbr_nids[0][2][0] == 5
    assert nbr_nids[0][3][0] == 2
    assert nbr_nids[0][4][0] == 2
    assert nbr_nids[1][4][0] == PADDED_NODE_ID
    assert nbr_nids[0][5][0] == 2


@pytest.fixture
def no_edge_feat_data():
    edge_index = torch.IntTensor([[1, 2], [2, 3], [3, 4]])
    edge_timestamps = torch.IntTensor([1, 2, 3])
    return DGData.from_raw(
        edge_timestamps,
        edge_index,
    )


def test_no_edge_feat_recency_nbr_sampler(no_edge_feat_data):
    dg = DGraph(no_edge_feat_data)
    hm = HookManager(keys=['unit'])
    n_nbrs = [1]  # 1 neighbor for each node
    recency_hook = RecencyNeighborHook(
        num_nbrs=n_nbrs,
        num_nodes=dg.num_nodes,
        seed_nodes_keys=['src', 'dst'],
        seed_times_keys=['time', 'time'],
        directed=True,
    )
    hm = HookManager(keys=['unit'])
    hm.register('unit', recency_hook)
    loader = DGDataLoader(dg, batch_size=3, hook_manager=hm)
    with hm.activate('unit'):
        batch_iter = iter(loader)
        batch_1 = next(batch_iter)
        assert isinstance(batch_1, DGBatch)
        nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_1)
        assert nids.shape == (1, 6)
        assert nbr_nids.shape == (1, 6, 1)
        assert nbr_times.shape == (1, 6, 1)
        assert nbr_feats.shape == (1, 6, 1, 0)  # No edge features


@pytest.fixture
def node_only_data():
    edge_index = torch.IntTensor([[1, 2], [2, 3], [3, 4]])
    edge_timestamps = torch.IntTensor([1, 2, 3])
    edge_feats = torch.IntTensor([[1], [2], [3]])
    dynamic_node_feats = torch.rand(2, 5)
    node_timestamps = torch.IntTensor([4, 5])
    node_ids = torch.IntTensor([5, 6])
    return DGData.from_raw(
        edge_timestamps,
        edge_index,
        edge_feats=edge_feats,
        dynamic_node_feats=dynamic_node_feats,
        node_timestamps=node_timestamps,
        node_ids=node_ids,
    )


def test_node_only_batch_recency_nbr_sampler(node_only_data):
    dg = DGraph(node_only_data)
    hm = HookManager(keys=['unit'])
    n_nbrs = [1]  # 1 neighbor for each node
    recency_hook = RecencyNeighborHook(
        num_nbrs=n_nbrs,
        num_nodes=dg.num_nodes,
        seed_nodes_keys=['src', 'dst'],
        seed_times_keys=['time', 'time'],
        directed=True,
    )
    hm = HookManager(keys=['unit'])
    hm.register('unit', recency_hook)
    loader = DGDataLoader(dg, batch_size=3, hook_manager=hm)
    with hm.activate('unit'):
        batch_iter = iter(loader)
        batch_1 = next(batch_iter)
        assert isinstance(batch_1, DGBatch)
        nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_1)
        assert nids.shape == (1, 6)
        assert nbr_nids.shape == (1, 6, 1)
        assert nbr_times.shape == (1, 6, 1)
        assert nbr_feats.shape == (1, 6, 1, 1)

        batch_2 = next(batch_iter)
        assert isinstance(batch_2, DGBatch)
        torch.testing.assert_close(batch_2.nids[0], torch.empty(0, dtype=torch.int32))
        torch.testing.assert_close(
            batch_2.nbr_nids[0], torch.empty(0, dtype=torch.int32)
        )
        torch.testing.assert_close(
            batch_2.nbr_times[0], torch.empty(0, dtype=torch.int64)
        )
        torch.testing.assert_close(
            batch_2.nbr_feats[0], torch.empty((0, 1), dtype=torch.float32)
        )


def test_hook_nbr_mask(basic_sample_graph):
    dg = DGraph(basic_sample_graph)
    n_nbrs = [1]  # 1 neighbor for each node
    recency_hook = RecencyNeighborHook(
        num_nbrs=n_nbrs,
        num_nodes=dg.num_nodes,
        seed_nodes_keys=['src', 'dst'],
        seed_times_keys=['time', 'time'],
    )
    hm = HookManager(keys=['unit'])
    hm.register('unit', recency_hook)
    hm.set_active_hooks('unit')
    loader = DGDataLoader(dg, hook_manager=hm, batch_size=1)

    batch_iter = iter(loader)
    batch_1 = next(batch_iter)
    nbr_mask = batch_1.seed_node_nbr_mask
    assert 'src' in nbr_mask
    assert 'dst' in nbr_mask
    assert nbr_mask['src'].shape[0] == 1
    assert nbr_mask['dst'].shape[0] == 1

    assert nbr_mask['src'] == np.array([0])
    assert nbr_mask['dst'] == np.array([1])
