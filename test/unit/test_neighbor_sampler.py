from typing import List

import numpy as np
import torch

from tgm import DGBatch, DGraph
from tgm.data import DGData
from tgm.hooks import DGHook, NeighborSamplerHook, RecencyNeighborHook
from tgm.loader import DGDataLoader


def _init_hooks(dg: DGraph, sampling_type: str, n_nbrs: List) -> List[DGHook]:
    if sampling_type == 'uniform':
        nbr_hook = NeighborSamplerHook(num_nbrs=n_nbrs)
    elif sampling_type == 'recency':
        nbr_hook = RecencyNeighborHook(
            num_nbrs=n_nbrs,
            num_nodes=dg.num_nodes,
            edge_feats_dim=dg.edge_feats_dim,
        )
    else:
        raise ValueError(f'Unknown sampling type: {sampling_type}')
    return [nbr_hook]


def _batch_eq_nbrs(batch_1: DGBatch, batch_2: DGBatch) -> bool:
    """Assert if two batches neighbors are equal."""
    assert np.array_equal(batch_1.nids, batch_2.nids)
    assert np.array_equal(batch_1.nbr_nids, batch_2.nbr_nids)
    assert np.array_equal(batch_1.nbr_times, batch_2.nbr_times)
    assert np.array_equal(batch_1.nbr_feats, batch_2.nbr_feats)
    assert np.array_equal(batch_1.nbr_mask, batch_2.nbr_mask)
    return True


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


def _init_basic_sample_graph():
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
    return edge_index, edge_timestamps, edge_feats


def test_init_basic_sampled_graph_1_hop():
    """The goal of this test is to provide a simple TG with 1-hop neighbors
    and test the basic functionality of the neighbor sampler.
    also make sure recency and uniform samplers return the same output.
    """
    edge_index, edge_timestamps, edge_feats = _init_basic_sample_graph()
    data = DGData.from_raw(edge_timestamps, edge_index, edge_feats)
    dg = DGraph(data)
    n_nbrs = [1]  # 1 neighbor for each node
    hook = _init_hooks(dg, 'recency', n_nbrs=n_nbrs)
    loader = DGDataLoader(dg, hook=hook, batch_size=1)
    assert loader._batch_size == 1

    batch_iter = iter(loader)
    batch_1 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats, nbr_mask = _nbrs_2_np(batch_1)
    assert nids.shape == (1, 2)
    assert nids[0][0] == 0
    assert nids[0][1] == 1
    assert nbr_nids.shape == (1, 2, 1)
    assert nbr_nids[0][0][0] == -1
    assert nbr_nids[0][1][0] == -1
    assert nbr_times.shape == (1, 2, 1)
    assert nbr_times[0][0][0] == 0
    assert nbr_times[0][1][0] == 0
    assert nbr_feats.shape == (1, 2, 1, 1)  # 1 feature per edge
    assert nbr_feats[0][1][0][0] == nbr_feats[0][0][0][0] == 0.0
    assert nbr_mask.shape == (1, 2, 1)

    batch_2 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats, nbr_mask = _nbrs_2_np(batch_2)
    assert nids.shape == (1, 2)
    assert nids[0][0] == 0
    assert nids[0][1] == 2
    assert nbr_nids.shape == (1, 2, 1)
    assert nbr_nids[0][0][0] == 1
    assert nbr_nids[0][1][0] == -1
    assert nbr_times.shape == (1, 2, 1)
    assert nbr_times[0][0][0] == 1
    assert nbr_times[0][1][0] == 0
    assert nbr_feats.shape == (1, 2, 1, 1)  # 1 feature per edge
    assert nbr_feats[0][0][0][0] == 1.0
    assert nbr_feats[0][1][0][0] == 0.0
    assert nbr_mask.shape == (1, 2, 1)

    batch_3 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats, nbr_mask = _nbrs_2_np(batch_3)
    assert nids.shape == (1, 2)
    assert nids[0][0] == 2
    assert nids[0][1] == 3
    assert nbr_nids.shape == (1, 2, 1)
    assert nbr_nids[0][0][0] == 0
    assert nbr_nids[0][1][0] == -1
    assert nbr_times.shape == (1, 2, 1)
    assert nbr_times[0][0][0] == 2
    assert nbr_times[0][1][0] == 0
    assert nbr_feats.shape == (1, 2, 1, 1)  # 1 feature per edge
    assert nbr_feats[0][0][0][0] == 2.0
    assert nbr_feats[0][1][0][0] == 0.0
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

    # starting test for uniform sampler
    hook = _init_hooks(dg, 'uniform', n_nbrs=n_nbrs)
    loader = DGDataLoader(dg, hook=hook, batch_size=1)
    assert loader._batch_size == 1

    batch = next(iter(loader))
    nids, nbr_nids, nbr_times, nbr_feats, nbr_mask = _nbrs_2_np(batch)
    assert nids.shape == (1, 2)
    assert nbr_nids.shape == (1, 2, 1)
    assert nbr_times.shape == (1, 2, 1)
    assert nbr_feats.shape == (1, 2, 1, 1)  # 1 feature per edge
    assert nbr_mask.shape == (1, 2, 1)
    assert _batch_eq_nbrs(
        batch_1, batch
    )  # test recency and uniform return the same output

    batch = next(iter(loader))
    assert _batch_eq_nbrs(batch_2, batch)

    batch = next(iter(loader))
    assert _batch_eq_nbrs(batch_3, batch)

    batch = next(iter(loader))
    assert _batch_eq_nbrs(batch_4, batch)


def _init_recency_buffer_graph():
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
    return edge_index, edge_timestamps, edge_feats


def test_recency_exceed_buffer():
    """The goal of this test is to test if the recency neighbor sampler would be able to update correctly when exceeding its max size.
    The test only has a single source node connecting to various destination nodes.
    """
    edge_index, edge_timestamps, edge_feats = _init_recency_buffer_graph()
    data = DGData.from_raw(edge_timestamps, edge_index, edge_feats)
    dg = DGraph(data)
    n_nbrs = [2]  # 2 neighbors for each node
    hook = _init_hooks(dg, 'recency', n_nbrs=n_nbrs)
    loader = DGDataLoader(dg, hook=hook, batch_size=2)
    assert loader._batch_size == 2

    batch_iter = iter(loader)
    batch_1 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats, _ = _nbrs_2_np(batch_1)
    assert nids.shape == (1, 4)
    assert nbr_nids.shape == (1, 4, 2)
    assert nbr_nids[0][0][0] == -1
    assert nbr_nids[0][0][1] == -1
    assert nbr_times.shape == (1, 4, 2)
    assert nbr_times[0][0][0] == 0
    assert nbr_times[0][0][1] == 0
    assert nbr_feats.shape == (1, 4, 2, 1)  # 1 feature per edge

    batch_2 = next(batch_iter)
    nids, nbr_nids, nbr_times, nbr_feats, _ = _nbrs_2_np(batch_2)
    print(nbr_nids[0][0])
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
