from unittest.mock import Mock, patch

import pytest
import torch

from tgm import DGBatch, DGraph
from tgm.data import DGData
from tgm.hooks import TGBNegativeEdgeSamplerHook


@pytest.fixture
def data():
    edge_index = torch.IntTensor([[2, 2], [2, 4], [1, 8]])
    edge_timestamps = torch.IntTensor([1, 5, 20])
    return DGData.from_raw(edge_timestamps=edge_timestamps, edge_index=edge_index)


def test_hook_dependancies():
    assert TGBNegativeEdgeSamplerHook.requires == set()
    assert TGBNegativeEdgeSamplerHook.produces == {'neg', 'neg_batch_list', 'neg_time'}


def test_hook_reset_state():
    assert TGBNegativeEdgeSamplerHook.has_state == False


class FakeNegSampler:
    def query_batch(self, src, dst, time, split_mode='val'):
        return []


def test_bad_tgb_negative_edge_sampler_init():
    mock_sampler = Mock(spec=FakeNegSampler)
    mock_sampler.eval_set = {}
    mock_sampler.eval_set['val'] = None
    with pytest.raises(ValueError):
        TGBNegativeEdgeSamplerHook(dataset_name='tgbl-wiki', split_mode='invalid_mode')


@patch('tgb.linkproppred.negative_sampler.NegativeEdgeSampler')
def test_negative_edge_sampler(MockNegSampler, data):
    dg = DGraph(data)
    mock_sampler = Mock()
    mock_sampler.eval_set = {'val': {'cool'}, 'test': {'cool'}}
    mock_sampler.query_batch.return_value = [[0] for _ in range(3)]
    MockNegSampler.return_value = mock_sampler

    hook = TGBNegativeEdgeSamplerHook(dataset_name='foo', split_mode='val')
    batch = hook(dg, dg.materialize())
    assert isinstance(batch, DGBatch)
    assert torch.is_tensor(batch.neg)
    assert torch.is_tensor(batch.neg_time)
    assert len(batch.neg_batch_list) == batch.src.shape[0]
    assert batch.neg_time.shape == batch.neg.shape
