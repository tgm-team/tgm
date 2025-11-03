from typing import Set

import pytest
import torch

from tgm import DGBatch, DGraph
from tgm.constants import PADDED_NODE_ID
from tgm.data import DGData, DGDataLoader
from tgm.hooks import HookManager, StatelessHook, TemporalSubgraphHook


@pytest.fixture
def dummy_data():
    # This is just to create a dummy instance of DGData to feed to DGLoader
    edge_index = torch.IntTensor([[2, 2], [2, 4], [1, 20]])
    edge_timestamps = torch.LongTensor([1, 5, 20])
    edge_feats = torch.rand(3, 5)
    node_timestamps = torch.LongTensor([2, 3, 4])
    node_ids = torch.IntTensor([2, 2, 2])
    dynamic_node_feats = torch.rand(3, 3)
    return DGData.from_raw(
        edge_timestamps,
        edge_index,
        edge_feats,
        node_timestamps=node_timestamps,
        node_ids=node_ids,
        dynamic_node_feats=dynamic_node_feats,
    )


class RecencyNeighborHookMock(StatelessHook):
    requires: Set[str] = set()
    produces = {'nbr_nids', 'nbr_times', 'nbr_feats'}

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        batch.nbr_nids = [torch.Tensor([[2, 3, 4], [5, 6, 7]])]

        batch.nbr_feats = [
            torch.Tensor([[[1, -1], [2, 2], [3, 3]], [[4, -4], [5, -5], [6, -6]]])
        ]

        batch.nbr_times = [torch.Tensor([[10, 20, 30], [40, 50, 60]])]

        # overwrite for testing
        batch.node_ids = torch.Tensor([8, 9])
        batch.node_times = torch.Tensor([80, 90])
        return batch


class RecencyNeighborHookMockWithPaddedNode(StatelessHook):
    requires: Set[str] = set()
    produces = {'nbr_nids', 'nbr_times', 'nbr_feats'}

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        batch.nbr_nids = [torch.Tensor([[2, PADDED_NODE_ID], [3, PADDED_NODE_ID]])]

        batch.nbr_feats = [torch.Tensor([[[1, -1], [0, 0]], [[2, -2], [0, 0]]])]

        batch.nbr_times = [torch.Tensor([[20, 0], [30, 0]])]

        # overwrite for testing
        batch.node_ids = torch.Tensor([8, 9])
        batch.node_times = torch.Tensor([80, 90])
        return batch


def test_bad_subgraph_sampler_init():
    with pytest.raises(ValueError):
        TemporalSubgraphHook(
            num_nodes=1, seed_nodes_keys=['foo', 'bar'], seed_times_keys=['foo']
        )


def test_bad_sample_with_non_existent_seeds(dummy_data):
    dg = DGraph(dummy_data)
    hook = TemporalSubgraphHook(
        num_nodes=1, seed_nodes_keys=['foo'], seed_times_keys=['bar']
    )
    batch = dg.materialize()

    with pytest.raises(ValueError):
        _ = hook(dg, batch)


def test_sample_with_none_seeds(dummy_data):
    dg = DGraph(dummy_data)
    hook = TemporalSubgraphHook(
        num_nodes=1, seed_nodes_keys=['foo'], seed_times_keys=['bar']
    )
    batch = dg.materialize()
    batch.foo, batch.bar = None, None

    with pytest.warns(UserWarning):
        batch = hook(dg, batch)


def test_bad_sample_with_non_tensor_non_None_seeds(dummy_data):
    dg = DGraph(dummy_data)
    hook = TemporalSubgraphHook(
        num_nodes=1, seed_nodes_keys=['foo'], seed_times_keys=['bar']
    )
    batch = dg.materialize()
    batch.foo = 'should_be_1d_tensor'
    batch.bar = 'should_be_1d_tensor'

    with pytest.raises(ValueError):
        batch = hook(dg, batch)


def test_bad_sample_with_non_1d_tensor_seeds(dummy_data):
    dg = DGraph(dummy_data)
    hook = TemporalSubgraphHook(
        num_nodes=1, seed_nodes_keys=['foo'], seed_times_keys=['bar']
    )
    batch = dg.materialize()
    batch.foo = torch.rand(2, 3)  # should be 1-d
    batch.bar = torch.rand(2, 3)  # should be 1-d

    with pytest.raises(ValueError):
        batch = hook(dg, batch)


def test_bad_sample_with_seeds_id_out_of_range(dummy_data):
    dg = DGraph(dummy_data)
    hook = TemporalSubgraphHook(
        num_nodes=1, seed_nodes_keys=['foo'], seed_times_keys=['bar']
    )
    batch = dg.materialize()
    batch.foo = torch.IntTensor([-1])  # should be positive
    batch.bar = torch.LongTensor([1])

    with pytest.raises(ValueError):
        batch = hook(dg, batch)


def test_bad_sample_with_seeds_time_out_of_range(dummy_data):
    dg = DGraph(dummy_data)
    hook = TemporalSubgraphHook(
        num_nodes=1, seed_nodes_keys=['foo'], seed_times_keys=['bar']
    )
    batch = dg.materialize()
    batch.foo = torch.IntTensor([1])
    batch.bar = torch.LongTensor([-1])  # should be positive

    with pytest.raises(ValueError):
        batch = hook(dg, batch)


def test_normal_temporal_subgraph(dummy_data):
    dg = DGraph(dummy_data)

    hm = HookManager(keys=['unit'])
    hm.register_shared(RecencyNeighborHookMock())
    hm.register_shared(
        TemporalSubgraphHook(
            num_nodes=dg.num_nodes,
            seed_nodes_keys=['node_ids'],
            seed_times_keys=['node_times'],
        )
    )
    loader = DGDataLoader(dg, hook_manager=hm)
    with hm.activate('unit'):
        batch_iter = iter(loader)
        batch_1 = next(batch_iter)
        expected_nodes = torch.Tensor([2, 3, 4, 5, 6, 7, 8, 9])
        assert torch.equal(expected_nodes, batch_1.sg_unique_nids)

        for i in range(2, 10):
            assert batch_1.sg_global_to_local(i).item() == i - 2

        assert torch.equal(torch.Tensor([8, 8, 8, 9, 9, 9]), batch_1.sg_src)
        assert torch.equal(torch.Tensor([2, 3, 4, 5, 6, 7]), batch_1.sg_dst)
        assert torch.equal(torch.Tensor([10, 20, 30, 40, 50, 60]), batch_1.sg_time)
        assert torch.equal(
            torch.Tensor([[1, -1], [2, 2], [3, 3], [4, -4], [5, -5], [6, -6]]),
            batch_1.sg_edge_feats,
        )


def test_normal_temporal_subgraph_with_padded_node(dummy_data):
    dg = DGraph(dummy_data)

    hm = HookManager(keys=['unit'])
    hm.register_shared(RecencyNeighborHookMockWithPaddedNode())
    hm.register_shared(
        TemporalSubgraphHook(
            num_nodes=dg.num_nodes,
            seed_nodes_keys=['node_ids'],
            seed_times_keys=['node_times'],
        )
    )
    loader = DGDataLoader(dg, hook_manager=hm)
    with hm.activate('unit'):
        batch_iter = iter(loader)
        batch_1 = next(batch_iter)
        expected_nodes = torch.Tensor([2, 3, 8, 9])
        assert torch.equal(expected_nodes, batch_1.sg_unique_nids)

        for input, expect in zip([2, 3, 8, 9], [0, 1, 2, 3]):
            assert batch_1.sg_global_to_local(input).item() == expect

        assert torch.equal(torch.Tensor([8, 9]), batch_1.sg_src)
        assert torch.equal(torch.Tensor([2, 3]), batch_1.sg_dst)
        assert torch.equal(torch.Tensor([20, 30]), batch_1.sg_time)
        assert torch.equal(torch.Tensor([[1, -1], [2, -2]]), batch_1.sg_edge_feats)
