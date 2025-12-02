from unittest.mock import Mock, patch

import pytest
import torch

from tgm import DGBatch, DGraph
from tgm.data import DGData, DGDataLoader
from tgm.hooks import HookManager, TGBNegativeEdgeSamplerHook


@pytest.fixture
def data():
    edge_index = torch.IntTensor([[2, 2], [2, 4], [1, 8]])
    edge_timestamps = torch.LongTensor([1, 5, 20])
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


def test_attempt_init_tgb_negative_edge_sampler_on_tgbn_dataset():
    with pytest.raises(ValueError):
        TGBNegativeEdgeSamplerHook(dataset_name='tgbn-trade', split_mode='val')


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


@patch('tgb.utils.info.DATA_VERSION_DICT', {'foo': 1})
@patch('tgb.linkproppred.negative_sampler.NegativeEdgeSampler')
def test_negative_edge_sampler_with_version_info(MockNegSampler, data):
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


@patch('tgb.linkproppred.negative_sampler.NegativeEdgeSampler')
def test_negative_edge_sampler_throws_value_error(MockNegSampler, data):
    dg = DGraph(data)
    mock_sampler = Mock()
    mock_sampler.eval_set = {'val': {'cool'}, 'test': {'cool'}}
    mock_sampler.query_batch.return_value = [[0] for _ in range(3)]
    mock_sampler.query_batch.side_effect = ValueError('foo')
    MockNegSampler.return_value = mock_sampler

    hook = TGBNegativeEdgeSamplerHook(dataset_name='foo', split_mode='val')

    with pytest.raises(ValueError, match='`pip install --upgrade py-tgb`'):
        hook(dg, dg.materialize())


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
        dynamic_node_feats=dynamic_node_feats,
        node_timestamps=node_timestamps,
        node_ids=node_ids,
    )


@patch('tgb.linkproppred.negative_sampler.NegativeEdgeSampler')
def test_node_only_batch_negative_edge_sampler(MockNegSampler, node_only_data):
    dg = DGraph(node_only_data)
    mock_sampler = Mock()
    mock_sampler.eval_set = {'val': {'cool'}, 'test': {'cool'}}
    mock_sampler.query_batch.return_value = [[0] for _ in range(3)]
    MockNegSampler.return_value = mock_sampler

    hm = HookManager(keys=['val', 'test'])
    hm.register(
        'val', hook=TGBNegativeEdgeSamplerHook(dataset_name='foo', split_mode='val')
    )
    loader = DGDataLoader(dg, batch_size=3, hook_manager=hm)
    with hm.activate('val'):
        batch_iter = iter(loader)
        batch_1 = next(batch_iter)
        assert isinstance(batch_1, DGBatch)
        assert torch.is_tensor(batch_1.neg)
        assert torch.is_tensor(batch_1.neg_time)
        assert len(batch_1.neg_batch_list) == batch_1.src.shape[0]
        assert batch_1.neg_time.shape == batch_1.neg.shape

        batch_2 = next(batch_iter)
        assert isinstance(batch_2, DGBatch)
        assert batch_2.neg.shape == (0,)
        assert batch_2.neg_time.shape == (0,)
        assert len(batch_2.neg_batch_list) == 0  # empty list
