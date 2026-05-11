from unittest.mock import patch

import pytest
import torch

from tgm import DGBatch, DGraph
from tgm.constants import PADDED_NODE_ID
from tgm.data import DGData, DGDataLoader
from tgm.hooks import HookManager, NegativeEdgeSamplerHook


@pytest.fixture
def data():
    edge_index = torch.IntTensor([[2, 2], [2, 4], [1, 8]])
    edge_time = torch.LongTensor([1, 5, 20])
    return DGData.from_raw(edge_time, edge_index)


def test_hook_dependancies():
    hook = NegativeEdgeSamplerHook(low=0, high=10)
    assert hook.requires == {'edge_src', 'edge_dst', 'edge_time'}
    assert hook.produces == {'neg', 'neg_time'}

    hook_with_id = NegativeEdgeSamplerHook(low=0, high=10, id='foo')
    assert hook_with_id.requires == {'edge_src', 'edge_dst', 'edge_time'}
    assert hook_with_id.produces == {'neg_foo', 'neg_time_foo'}


def test_hook_repre():
    hook_with_id = NegativeEdgeSamplerHook(low=0, high=10, id='foo')
    assert 'foo' in hook_with_id.__repr__()


def test_hook_reset_state():
    assert NegativeEdgeSamplerHook.has_state == True


def test_bad_negative_edge_sampler_init():
    with pytest.raises(ValueError):
        NegativeEdgeSamplerHook(low=0, high=0)
    with pytest.raises(ValueError):
        NegativeEdgeSamplerHook(low=0, high=1, neg_ratio=0)
    with pytest.raises(ValueError):
        NegativeEdgeSamplerHook(low=0, high=1, neg_ratio=2)


def test_negative_edge_sampler(data):
    dg = DGraph(data)
    hook = NegativeEdgeSamplerHook(low=0, high=10)
    batch = hook(dg, dg.materialize())
    assert isinstance(batch, DGBatch)
    assert torch.is_tensor(batch.neg)
    assert torch.is_tensor(batch.neg_time)
    assert batch.neg.shape == batch.edge_dst.shape
    assert batch.neg_time.shape == batch.neg.shape


def test_negative_edge_sampler_with_id(data):
    dg = DGraph(data)
    hook = NegativeEdgeSamplerHook(low=0, high=10, id='foo')
    batch = hook(dg, dg.materialize())
    assert isinstance(batch, DGBatch)
    assert torch.is_tensor(batch.neg_foo)
    assert torch.is_tensor(batch.neg_time_foo)
    assert batch.neg_foo.shape == batch.edge_dst.shape
    assert batch.neg_time_foo.shape == batch.neg_foo.shape


@pytest.fixture
def node_only_data():
    edge_index = torch.IntTensor([[1, 2], [2, 3], [3, 4]])
    edge_time = torch.IntTensor([1, 2, 3])
    node_x = torch.rand(2, 5)
    node_x_time = torch.IntTensor([4, 5])
    node_x_nids = torch.IntTensor([5, 6])
    return DGData.from_raw(
        edge_time,
        edge_index,
        node_x=node_x,
        node_x_time=node_x_time,
        node_x_nids=node_x_nids,
    )


def test_node_only_batch_negative_edge_sampler(node_only_data):
    dg = DGraph(node_only_data)
    hm = HookManager(keys=['unit'])
    hm.register('unit', NegativeEdgeSamplerHook(low=0, high=6))
    loader = DGDataLoader(dg, batch_size=3, hook_manager=hm)
    with hm.activate('unit'):
        batch_iter = iter(loader)
        batch_1 = next(batch_iter)
        assert isinstance(batch_1, DGBatch)
        assert torch.is_tensor(batch_1.neg)
        assert torch.is_tensor(batch_1.neg_time)
        assert batch_1.neg.shape == batch_1.edge_dst.shape
        assert batch_1.neg_time.shape == batch_1.neg.shape

        batch_2 = next(batch_iter)
        assert isinstance(batch_2, DGBatch)
        assert batch_2.neg.shape == (0,)
        assert batch_2.neg_time.shape == (0,)


@pytest.fixture
def data_test_hst_rnd():
    edge_index = torch.IntTensor(
        [
            [1, 5],
            [7, 6],
            [2, 8],
            [7, 8],  # 1st batch
            [1, 7],
            [2, 9],
            [3, 10],
            [4, 5],  # 2nd batch
            [3, 11],  # 3rd batch
        ]
    )
    edge_time = torch.arange(edge_index.size(0))
    return DGData.from_raw(edge_time, edge_index)


def test_hst_rnd(data_test_hst_rnd):
    dg = DGraph(data_test_hst_rnd)

    hm = HookManager(keys=['unit'])
    sampler = NegativeEdgeSamplerHook(low=0, high=6, strategy='hist_rnd')

    def mock_random_sampling(dg: DGraph, batch: DGBatch):
        neg = torch.full((batch.edge_src.size(0),), PADDED_NODE_ID, dtype=torch.int32)
        neg_time = batch.edge_time.clone()
        return neg, neg_time

    hm.register('unit', sampler)
    loader = DGDataLoader(dg, batch_size=4, hook_manager=hm)

    with patch.object(sampler, '_random_sampling', mock_random_sampling):
        with hm.activate('unit'):
            batch_iter = iter(loader)
            batch_1 = next(batch_iter)
            assert batch_1.neg.shape == (4,)
            assert torch.equal(
                batch_1.neg,
                torch.Tensor(
                    [PADDED_NODE_ID, PADDED_NODE_ID, PADDED_NODE_ID, PADDED_NODE_ID]
                ),
            )
            assert sampler._memory is not None
            assert sampler._memory.shape == (2, 8)
            assert sampler._count == 4

            batch_2 = next(batch_iter)
            assert batch_2.neg.shape == (4,)
            assert torch.equal(
                batch_2.neg, torch.Tensor([5, 8, PADDED_NODE_ID, PADDED_NODE_ID])
            )
            assert sampler._memory is not None
            assert sampler._memory.shape == (2, 8)
            assert sampler._count == 8

            batch_3 = next(batch_iter)
            assert batch_3.neg.shape == (1,)
            assert torch.equal(batch_3.neg, torch.Tensor([10]))

            assert sampler._memory is not None
            assert sampler._memory.shape == (2, 18)
            assert sampler._count != 0
            assert sampler._count == 9

            sampler.reset_state()
            assert sampler._memory is None
            assert sampler._count == 0


def test_invalid_strategy():
    with pytest.raises(ValueError):
        NegativeEdgeSamplerHook(low=0, high=6, strategy='foo')
