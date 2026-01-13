from typing import List

import numpy as np
import pytest
import torch

from tgm import DGBatch, DGraph
from tgm.constants import PADDED_NODE_ID
from tgm.data import DGData, DGDataLoader
from tgm.hooks import HookManager, NeighborSamplerHook


@pytest.fixture
def data():
    edge_index = torch.IntTensor([[2, 2], [2, 4], [1, 8]])
    edge_timestamps = torch.LongTensor([1, 5, 20])
    edge_x = torch.rand(3, 5)
    return DGData.from_raw(edge_timestamps, edge_index, edge_x)


def test_hook_dependancies():
    assert NeighborSamplerHook.requires == {'src', 'dst', 'edge_event_time'}
    assert NeighborSamplerHook.produces == {
        'nids',
        'nbr_nids',
        'nbr_times',
        'nbr_feats',
        'seed_node_nbr_mask',
    }


def test_hook_reset_state():
    assert NeighborSamplerHook.has_state == False


def test_bad_neighbor_sampler_init():
    with pytest.raises(ValueError):
        NeighborSamplerHook(
            num_nbrs=[], seed_nodes_keys=['foo'], seed_times_keys=['bar']
        )
    with pytest.raises(ValueError):
        NeighborSamplerHook(
            num_nbrs=[1, 0], seed_nodes_keys=['foo'], seed_times_keys=['bar']
        )
    with pytest.raises(ValueError):
        NeighborSamplerHook(
            num_nbrs=[1], seed_nodes_keys=['foo', 'bar'], seed_times_keys=['foo']
        )


def test_sample_with_node_events_seeds(node_only_data):
    dg = DGraph(node_only_data)
    hook = NeighborSamplerHook(
        num_nbrs=[1],
        seed_nodes_keys=['node_event_node_ids'],
        seed_times_keys=['node_event_time'],
    )
    batch = dg.materialize()
    batch = hook(dg, batch)
    assert len(batch.nids) == 1
    assert len(batch.times) == 1
    torch.testing.assert_close(batch.nids[0], batch.node_event_node_ids)
    torch.testing.assert_close(batch.times[0], batch.node_event_time)


def test_bad_sample_with_non_existent_seeds(data):
    dg = DGraph(data)
    hook = NeighborSamplerHook(
        num_nbrs=[1], seed_nodes_keys=['foo'], seed_times_keys=['bar']
    )
    batch = dg.materialize()

    with pytest.raises(ValueError):
        _ = hook(dg, batch)


def test_sample_with_none_seeds(data):
    dg = DGraph(data)
    hook = NeighborSamplerHook(
        num_nbrs=[1], seed_nodes_keys=['foo'], seed_times_keys=['bar']
    )
    batch = dg.materialize()
    batch.foo, batch.bar = None, None

    with pytest.warns(UserWarning):
        batch = hook(dg, batch)


def test_bad_sample_with_non_tensor_non_None_seeds(data):
    dg = DGraph(data)
    hook = NeighborSamplerHook(
        num_nbrs=[1], seed_nodes_keys=['foo'], seed_times_keys=['bar']
    )
    batch = dg.materialize()
    batch.foo = 'should_be_1d_tensor'
    batch.bar = 'should_be_1d_tensor'

    with pytest.raises(ValueError):
        batch = hook(dg, batch)


def test_bad_sample_with_non_1d_tensor_seeds(data):
    dg = DGraph(data)
    hook = NeighborSamplerHook(
        num_nbrs=[1], seed_nodes_keys=['foo'], seed_times_keys=['bar']
    )
    batch = dg.materialize()
    batch.foo = torch.rand(2, 3)  # should be 1-d
    batch.bar = torch.rand(2, 3)  # should be 1-d

    with pytest.raises(ValueError):
        batch = hook(dg, batch)


def test_bad_sample_with_seeds_id_out_of_range(data):
    dg = DGraph(data)
    hook = NeighborSamplerHook(
        num_nbrs=[1], seed_nodes_keys=['foo'], seed_times_keys=['bar']
    )
    batch = dg.materialize()
    batch.foo = torch.IntTensor([-1])  # should be positive
    batch.bar = torch.LongTensor([1])

    with pytest.raises(ValueError):
        batch = hook(dg, batch)


def test_bad_sample_with_seeds_time_out_of_range(data):
    dg = DGraph(data)
    hook = NeighborSamplerHook(
        num_nbrs=[1], seed_nodes_keys=['foo'], seed_times_keys=['bar']
    )
    batch = dg.materialize()
    batch.foo = torch.IntTensor([1])
    batch.bar = torch.LongTensor([-1])  # should be positive

    with pytest.raises(ValueError):
        batch = hook(dg, batch)


def test_neighbor_sampler_hook_link_pred(data):
    dg = DGraph(data)
    hook = NeighborSamplerHook(
        num_nbrs=[2],
        seed_nodes_keys=['src', 'dst'],
        seed_times_keys=['edge_event_time', 'edge_event_time'],
    )
    batch = dg.materialize()

    # Link Prediction will add negative edges to seed nodes for sampling
    batch.neg = torch.IntTensor([0] * len(batch.dst))
    batch.neg_time = torch.IntTensor([0] * len(batch.dst))
    batch = hook(dg, batch)
    assert isinstance(batch, DGBatch)
    assert hasattr(batch, 'nids')
    assert hasattr(batch, 'nbr_nids')
    assert hasattr(batch, 'nbr_times')
    assert hasattr(batch, 'nbr_feats')


def test_neighbor_sampler_hook_node_pred(data):
    dg = DGraph(data)
    hook = NeighborSamplerHook(
        num_nbrs=[2],
        seed_nodes_keys=['src', 'dst'],
        seed_times_keys=['edge_event_time', 'edge_event_time'],
    )
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
    edge_index = torch.IntTensor([[0, 1], [0, 2], [2, 3], [2, 0]])
    edge_timestamps = torch.LongTensor([1, 2, 3, 4])
    edge_x = torch.Tensor(
        [[1], [2], [5], [2]]
    )  # edge feat is simply summing the node IDs at two end points
    data = DGData.from_raw(edge_timestamps, edge_index, edge_x)
    return data


def test_init_basic_sampled_graph_1_hop(basic_sample_graph):
    dg = DGraph(basic_sample_graph)
    n_nbrs = [3]  # 3 neighbor for each node
    uniform_hook = NeighborSamplerHook(
        num_nbrs=n_nbrs,
        seed_nodes_keys=['src', 'dst'],
        seed_times_keys=['edge_event_time', 'edge_event_time'],
    )
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


def test_init_basic_sampled_graph_2_hop(basic_sample_graph):
    dg = DGraph(basic_sample_graph)
    n_nbrs = [1, 1]  # 3 neighbor for each node
    uniform_hook = NeighborSamplerHook(
        num_nbrs=n_nbrs,
        seed_nodes_keys=['src', 'dst'],
        seed_times_keys=['edge_event_time', 'edge_event_time'],
    )
    hm = HookManager(keys=['unit'])
    hm.register_shared(uniform_hook)
    loader = DGDataLoader(dg, batch_size=1, hook_manager=hm)
    assert loader._batch_size == 1
    with hm.activate('unit'):
        batch_iter = iter(loader)
        batch_1 = next(batch_iter)
        nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_1)
        assert nids.shape == (2, 2)
        assert nids[0][0] == 0
        assert nids[0][1] == 1
        assert nbr_nids.shape == (2, 2, 1)
        assert nbr_nids[0][0][0] == PADDED_NODE_ID
        assert nbr_nids[0][1][0] == PADDED_NODE_ID
        assert nbr_times.shape == (2, 2, 1)
        assert nbr_feats.shape == (2, 2, 1, 1)  # 1 feature per edge

        batch_2 = next(batch_iter)
        nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_2)
        assert nids.shape == (2, 2)
        assert nids[0][0] == 0
        assert nids[0][1] == 2
        assert nbr_nids.shape == (2, 2, 1)
        assert nbr_nids[0][0][0] == 1
        assert nbr_nids[0][1][0] == PADDED_NODE_ID
        assert nbr_times.shape == (2, 2, 1)
        assert nbr_feats.shape == (2, 2, 1, 1)  # 1 feature per edge
        assert nbr_feats[0][0][0][0] == 1.0

        batch_3 = next(batch_iter)
        nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_3)
        assert nids.shape == (2, 2)
        assert nids[0][0] == 2
        assert nids[0][1] == 3
        assert nbr_nids.shape == (2, 2, 1)
        assert nbr_nids[0][0][0] == 0
        assert nbr_nids[0][1][0] == PADDED_NODE_ID
        assert nbr_times.shape == (2, 2, 1)
        assert nbr_times[0][0][0] == 2
        assert nbr_feats.shape == (2, 2, 1, 1)  # 1 feature per edge
        assert nbr_feats[0][0][0][0] == 2.0

        batch_4 = next(batch_iter)
        nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_4)
        assert nids.shape == (2, 2)
        assert nids[0][0] == 2
        assert nids[0][1] == 0
        assert nbr_nids.shape == (2, 2, 1)
        assert nbr_nids[0][0][0] == 3
        assert nbr_nids[0][1][0] == 1
        assert nbr_times.shape == (2, 2, 1)
        assert nbr_times[0][0][0] == 3
        assert nbr_times[0][1][0] == 1
        assert nbr_feats.shape == (2, 2, 1, 1)  # 1 feature per edge
        assert nbr_feats[0][0][0][0] == 5.0
        assert nbr_feats[0][1][0][0] == 1.0


def test_init_basic_sampled_graph_directed_1_hop(basic_sample_graph):
    dg = DGraph(basic_sample_graph)
    n_nbrs = [3]  # 3 neighbor for each node
    uniform_hook = NeighborSamplerHook(
        num_nbrs=n_nbrs,
        seed_nodes_keys=['src', 'dst'],
        seed_times_keys=['edge_event_time', 'edge_event_time'],
        directed=True,
    )
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


@pytest.fixture
def no_edge_feat_data():
    edge_index = torch.IntTensor([[1, 2], [2, 3], [3, 4]])
    edge_timestamps = torch.IntTensor([1, 2, 3])
    return DGData.from_raw(
        edge_timestamps,
        edge_index,
    )


def test_no_edge_feat_data_neighbor_sampler(no_edge_feat_data):
    dg = DGraph(no_edge_feat_data)
    n_nbrs = [1]
    uniform_hook = NeighborSamplerHook(
        num_nbrs=n_nbrs,
        seed_nodes_keys=['src', 'dst'],
        seed_times_keys=['edge_event_time', 'edge_event_time'],
    )
    hm = HookManager(keys=['unit'])
    hm.register_shared(uniform_hook)
    loader = DGDataLoader(dg, batch_size=3, hook_manager=hm)
    assert loader._batch_size == 3
    with hm.activate('unit'):
        batch_iter = iter(loader)
        batch_1 = next(batch_iter)
        nids, nbr_nids, nbr_times, nbr_feats = _nbrs_2_np(batch_1)
        assert nids.shape == (1, 6)
        assert nbr_nids.shape == (1, 6, 1)
        assert nbr_times.shape == (1, 6, 1)
        assert nbr_feats.shape == (1, 6, 1, 0)


@pytest.fixture
def node_only_data():
    edge_index = torch.IntTensor([[1, 2], [2, 3], [3, 4]])
    edge_timestamps = torch.IntTensor([1, 2, 3])
    edge_x = torch.IntTensor([[1], [2], [3]])
    node_x = torch.rand(2, 5)
    node_timestamps = torch.IntTensor([4, 5])
    node_ids = torch.IntTensor([5, 6])
    return DGData.from_raw(
        edge_timestamps,
        edge_index,
        edge_x=edge_x,
        node_x=node_x,
        node_timestamps=node_timestamps,
        node_ids=node_ids,
    )


def test_node_only_batch_recency_nbr_sampler(node_only_data):
    dg = DGraph(node_only_data)
    hm = HookManager(keys=['unit'])
    n_nbrs = [1]  # 1 neighbor for each node
    uniform_hook = NeighborSamplerHook(
        num_nbrs=n_nbrs,
        seed_nodes_keys=['src', 'dst'],
        seed_times_keys=['edge_event_time', 'edge_event_time'],
    )
    hm = HookManager(keys=['unit'])
    hm.register_shared(uniform_hook)
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
    recency_hook = NeighborSamplerHook(
        num_nbrs=n_nbrs,
        seed_nodes_keys=['src', 'dst'],
        seed_times_keys=['edge_event_time', 'edge_event_time'],
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
