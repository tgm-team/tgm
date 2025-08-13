import pytest
import torch

from tgm import DGBatch, DGraph
from tgm.data import DGData
from tgm.hooks import NegativeEdgeSamplerHook


@pytest.fixture
def dg():
    edge_index = torch.LongTensor([[2, 2], [2, 4], [1, 8]])
    edge_timestamps = torch.LongTensor([1, 5, 20])
    data = DGData.from_raw(edge_timestamps, edge_index)
    return DGraph(data, discretize_time_delta='r')


def test_hook_dependancies():
    assert NegativeEdgeSamplerHook.requires == set()
    assert NegativeEdgeSamplerHook.produces == {'neg'}


def test_bad_negative_edge_sampler_init():
    with pytest.raises(ValueError):
        NegativeEdgeSamplerHook(low=0, high=0)
    with pytest.raises(ValueError):
        NegativeEdgeSamplerHook(low=0, high=1, neg_ratio=0)
    with pytest.raises(ValueError):
        NegativeEdgeSamplerHook(low=0, high=1, neg_ratio=2)


def test_negative_edge_sampler(dg):
    hook = NegativeEdgeSamplerHook(low=0, high=10)
    batch = hook(dg, dg.materialize())
    assert isinstance(batch, DGBatch)
    assert torch.is_tensor(batch.neg)
    assert batch.neg.shape == batch.dst.shape
