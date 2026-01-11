import pytest
import torch

from tgm import DGBatch, DGraph
from tgm.data import DGData, DGDataLoader
from tgm.hooks import HookManager, NegativeEdgeSamplerHook


@pytest.fixture
def data():
    edge_index = torch.IntTensor([[2, 2], [2, 4], [1, 8]])
    edge_timestamps = torch.LongTensor([1, 5, 20])
    return DGData.from_raw(edge_timestamps, edge_index)


def test_hook_dependancies():
    assert NegativeEdgeSamplerHook.requires == {'src', 'dst', 'time'}
    assert NegativeEdgeSamplerHook.produces == {'neg', 'neg_time'}


def test_hook_reset_state():
    assert NegativeEdgeSamplerHook.has_state == False


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
    assert batch.neg.shape == batch.dst.shape
    assert batch.neg_time.shape == batch.neg.shape


@pytest.fixture
def node_only_data():
    edge_index = torch.IntTensor([[1, 2], [2, 3], [3, 4]])
    edge_timestamps = torch.IntTensor([1, 2, 3])
    dynamic_node_feats = torch.rand(2, 5)
    node_timestamps = torch.IntTensor([4, 5])
    node_ids = torch.IntTensor([5, 6])
    return DGData.from_raw(
        edge_timestamps,
        edge_index,
        node_x=dynamic_node_feats,
        node_timestamps=node_timestamps,
        node_ids=node_ids,
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
        assert batch_1.neg.shape == batch_1.dst.shape
        assert batch_1.neg_time.shape == batch_1.neg.shape

        batch_2 = next(batch_iter)
        assert isinstance(batch_2, DGBatch)
        assert batch_2.neg.shape == (0,)
        assert batch_2.neg_time.shape == (0,)
