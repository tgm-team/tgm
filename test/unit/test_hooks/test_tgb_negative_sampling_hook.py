from unittest.mock import Mock

import pytest
import torch

from tgm import DGBatch, DGraph
from tgm.data import DGData
from tgm.hooks import TGBNegativeEdgeSamplerHook


@pytest.fixture
def data():
    edge_index = torch.LongTensor([[2, 2], [2, 4], [1, 8]])
    edge_timestamps = torch.LongTensor([1, 5, 20])
    return DGData.from_raw(edge_timestamps=edge_timestamps, edge_index=edge_index)


def test_hook_dependancies():
    assert TGBNegativeEdgeSamplerHook.requires == set()
    assert TGBNegativeEdgeSamplerHook.produces == {'neg', 'neg_batch_list'}


class FakeNegSampler:
    def query_batch(self, src, dst, time, split_mode='val'):
        return []


def test_bad_tgb_negative_edge_sampler_init():
    mock_sampler = Mock(spec=FakeNegSampler)
    mock_sampler.eval_set = {}
    mock_sampler.eval_set['val'] = None
    with pytest.raises(ValueError):
        TGBNegativeEdgeSamplerHook(neg_sampler=None, split_mode='val')
    with pytest.raises(ValueError):
        TGBNegativeEdgeSamplerHook(neg_sampler=mock_sampler, split_mode='invalid_mode')
    with pytest.raises(ValueError):
        TGBNegativeEdgeSamplerHook(neg_sampler=mock_sampler, split_mode='val')


def test_negative_edge_sampler(data):
    dg = DGraph(data, discretize_time_delta='r')
    mock_sampler = Mock(spec=FakeNegSampler)
    mock_sampler.eval_set = {}
    mock_sampler.eval_set['val'] = 'cool'
    mock_sampler.eval_set['test'] = 'cool'

    neg_batch_list = []
    for i in range(3):
        neg_batch_list.append([0])
    mock_sampler.query_batch.return_value = neg_batch_list
    hook = TGBNegativeEdgeSamplerHook(neg_sampler=mock_sampler, split_mode='val')
    batch = hook(dg, dg.materialize())
    assert isinstance(batch, DGBatch)
    assert torch.is_tensor(batch.neg)
    assert len(batch.neg_batch_list) == batch.src.shape[0]
