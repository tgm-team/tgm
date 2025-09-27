import pytest
import torch

from tgm import DGBatch, DGraph
from tgm.data import DGData
from tgm.hooks import NegativeEdgeSamplerHook


@pytest.fixture
def data():
    edge_index = torch.IntTensor([[2, 2], [2, 4], [1, 8]])
    edge_timestamps = torch.LongTensor([1, 5, 20])
    return DGData.from_raw(edge_timestamps, edge_index)


def test_hook_dependancies():
    assert NegativeEdgeSamplerHook.requires == set()
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
