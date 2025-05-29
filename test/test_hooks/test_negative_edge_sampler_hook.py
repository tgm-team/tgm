import pytest
import torch

from opendg.data import DGData
from opendg.graph import DGBatch, DGraph
from opendg.hooks import NegativeEdgeSamplerHook


@pytest.fixture
def data():
    # TODO: Missing node events
    edge_index = torch.Tensor([[2, 2], [2, 4], [1, 8]])
    timestamps = torch.Tensor([1, 5, 20])
    return DGData(edge_index, timestamps)
    # return [
    #    NodeEvent(t=1, src=2),
    #    EdgeEvent(t=1, src=2, dst=2),
    #    NodeEvent(t=5, src=4),
    #    EdgeEvent(t=5, src=2, dst=4),
    #    NodeEvent(t=10, src=6),
    #    EdgeEvent(t=20, src=1, dst=8),
    # ]


def test_bad_negative_edge_sampler_init():
    with pytest.raises(ValueError):
        NegativeEdgeSamplerHook(low=0, high=0)
    with pytest.raises(ValueError):
        NegativeEdgeSamplerHook(low=0, high=1, neg_sampling_ratio=0)
    with pytest.raises(ValueError):
        NegativeEdgeSamplerHook(low=0, high=1, neg_sampling_ratio=2)


def test_negative_edge_sampler(data):
    dg = DGraph(data)
    hook = NegativeEdgeSamplerHook(low=0, high=10)
    batch = hook(dg)
    assert isinstance(batch, DGBatch)
    assert torch.is_tensor(batch.neg)
    assert batch.neg.shape == batch.dst.shape
