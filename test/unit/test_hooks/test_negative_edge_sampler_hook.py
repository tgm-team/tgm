import pytest
import torch

from tgm import DGBatch, DGraph
from tgm.constants import PADDED_NODE_ID
from tgm.data import DGData, DGDataLoader
from tgm.hooks import (
    HistoricalNegativeEdgeSamplerHook,
    HookManager,
    RandomNegativeEdgeSamplerHook,
)


@pytest.fixture
def data():
    edge_index = torch.IntTensor([[2, 2], [2, 4], [1, 8]])
    edge_time = torch.LongTensor([1, 5, 20])
    return DGData.from_raw(edge_time, edge_index)


def test_hook_dependancies():
    hook = RandomNegativeEdgeSamplerHook(low=0, high=10)
    assert hook.requires == {'edge_src', 'edge_dst', 'edge_time'}
    assert hook.produces == {'neg', 'neg_time'}

    hook_with_id = RandomNegativeEdgeSamplerHook(low=0, high=10, id='foo')
    assert hook_with_id.requires == {'edge_src', 'edge_dst', 'edge_time'}
    assert hook_with_id.produces == {'neg_foo', 'neg_time_foo'}

    hook = HistoricalNegativeEdgeSamplerHook()
    assert hook.requires == {'edge_src', 'edge_dst', 'edge_time'}
    assert hook.produces == {'neg', 'neg_time', 'valid_neg_mask'}

    hook_with_id = HistoricalNegativeEdgeSamplerHook(id='foo')
    assert hook_with_id.requires == {'edge_src', 'edge_dst', 'edge_time'}
    assert hook_with_id.produces == {'neg_foo', 'neg_time_foo', 'valid_neg_mask_foo'}


def test_hook_repre():
    hook_with_id = RandomNegativeEdgeSamplerHook(low=0, high=10, id='foo')
    assert 'foo' in hook_with_id.__repr__()

    hook_with_id = HistoricalNegativeEdgeSamplerHook(id='foo')
    assert 'foo' in hook_with_id.__repr__()


def test_hook_reset_state():
    assert RandomNegativeEdgeSamplerHook.has_state == False
    assert HistoricalNegativeEdgeSamplerHook.has_state == True


def test_bad_negative_edge_sampler_init():
    with pytest.raises(ValueError):
        RandomNegativeEdgeSamplerHook(low=0, high=0)
    with pytest.raises(ValueError):
        RandomNegativeEdgeSamplerHook(low=0, high=1, neg_ratio=0)
    with pytest.raises(ValueError):
        RandomNegativeEdgeSamplerHook(low=0, high=1, neg_ratio=2)


def test_negative_edge_sampler(data):
    dg = DGraph(data)
    hook = RandomNegativeEdgeSamplerHook(low=0, high=10)
    batch = hook(dg, dg.materialize())
    assert isinstance(batch, DGBatch)
    assert torch.is_tensor(batch.neg)
    assert torch.is_tensor(batch.neg_time)
    assert batch.neg.shape == batch.edge_dst.shape
    assert batch.neg_time.shape == batch.neg.shape


def test_negative_edge_sampler_with_id(data):
    dg = DGraph(data)
    hook = RandomNegativeEdgeSamplerHook(low=0, high=10, id='foo')
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
    hm.register('unit', RandomNegativeEdgeSamplerHook(low=0, high=6))
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
def data_test_hst_sampling():
    edge_index = torch.IntTensor(
        [
            # 1st batch
            [1, 5],
            [7, 6],
            [2, 8],
            [7, 8],
            # 2nd batch
            [1, 7],
            [9, 10],
            [3, 10],
            [1, 9],
            # 3rd batch
            [3, 11],
            [2, 10],
            [7, 2],
            [3, 5],
        ]
    )
    edge_time = torch.arange(edge_index.size(0))
    return DGData.from_raw(edge_time, edge_index)


def test_hst_sampling(data_test_hst_sampling):
    dg = DGraph(data_test_hst_sampling)

    hm = HookManager(keys=['unit'])
    sampler = HistoricalNegativeEdgeSamplerHook()

    hm.register('unit', sampler)
    loader = DGDataLoader(dg, batch_size=4, hook_manager=hm)

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
        assert torch.equal(
            batch_1.valid_neg_mask,
            torch.Tensor([False, False, False, False]),
        )
        assert sampler._memory is not None
        assert sampler._memory.shape == (2, 8)
        assert sampler._count == 4

        batch_2 = next(batch_iter)
        assert batch_2.neg.shape == (4,)
        assert torch.equal(
            batch_2.neg, torch.Tensor([5, PADDED_NODE_ID, PADDED_NODE_ID, 5])
        )
        assert torch.equal(
            batch_2.valid_neg_mask,
            torch.Tensor([True, False, False, True]),
        )
        assert sampler._memory is not None
        assert sampler._memory.shape == (2, 8)
        assert sampler._count == 8

        batch_3 = next(batch_iter)
        assert batch_3.neg.shape == (4,)
        assert torch.equal(batch_3.neg, torch.Tensor([10, 8, 8, 10])) or torch.equal(
            batch_3.neg, torch.Tensor([10, 8, 6, 10])
        )
        assert torch.equal(
            batch_3.valid_neg_mask,
            torch.Tensor([True, True, True, True]),
        )
        assert sampler._memory is not None
        assert sampler._memory.shape == (2, 24)
        assert sampler._count != 0
        assert sampler._count == 12

        sampler.reset_state()
        assert sampler._memory is None
        assert sampler._count == 0
