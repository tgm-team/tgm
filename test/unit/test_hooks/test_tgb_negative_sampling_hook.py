from unittest.mock import Mock, patch

import pytest
import torch

from tgm import DGBatch, DGraph
from tgm.data import DGData, DGDataLoader
from tgm.hooks import (
    HookManager,
    TGBNegativeEdgeSamplerHook,
    TGBTHGNegativeEdgeSamplerHook,
    TGBTKGNegativeEdgeSamplerHook,
)


@pytest.fixture
def data():
    edge_index = torch.IntTensor([[2, 2], [2, 4], [1, 8]])
    edge_timestamps = torch.LongTensor([1, 5, 20])
    return DGData.from_raw(edge_timestamps=edge_timestamps, edge_index=edge_index)


@pytest.fixture
def thg_data():
    edge_index = torch.IntTensor([[2, 2], [2, 4], [1, 8]])
    edge_timestamps = torch.LongTensor([1, 5, 20])
    edge_type = torch.IntTensor([0, 1, 2])
    node_type = torch.arange(9, dtype=torch.int32)
    return DGData.from_raw(
        edge_timestamps=edge_timestamps,
        edge_index=edge_index,
        edge_type=edge_type,
        node_type=node_type,
    )


@pytest.fixture
def tkg_data():
    edge_index = torch.IntTensor([[2, 2], [2, 4], [1, 8]])
    edge_timestamps = torch.LongTensor([1, 5, 20])
    edge_type = torch.IntTensor([0, 1, 2])
    return DGData.from_raw(
        edge_timestamps=edge_timestamps,
        edge_index=edge_index,
        edge_type=edge_type,
    )


def test_hook_dependancies():
    assert TGBNegativeEdgeSamplerHook.requires == set()
    assert TGBNegativeEdgeSamplerHook.produces == {'neg', 'neg_batch_list', 'neg_time'}
    assert TGBTHGNegativeEdgeSamplerHook.requires == set()
    assert TGBTHGNegativeEdgeSamplerHook.produces == {
        'neg',
        'neg_batch_list',
        'neg_time',
    }


def test_hook_reset_state():
    assert TGBNegativeEdgeSamplerHook.has_state == False
    assert TGBTHGNegativeEdgeSamplerHook.has_state == False


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

    hook = TGBNegativeEdgeSamplerHook(dataset_name='tgbl-foo', split_mode='val')
    batch = hook(dg, dg.materialize())
    assert isinstance(batch, DGBatch)
    assert torch.is_tensor(batch.neg)
    assert torch.is_tensor(batch.neg_time)
    assert len(batch.neg_batch_list) == batch.src.shape[0]
    assert batch.neg_time.shape == batch.neg.shape


@patch('tgb.utils.info.DATA_VERSION_DICT', {'tgbl-foo': 1})
@patch('tgb.linkproppred.negative_sampler.NegativeEdgeSampler')
def test_negative_edge_sampler_with_version_info(MockNegSampler, data):
    dg = DGraph(data)
    mock_sampler = Mock()
    mock_sampler.eval_set = {'val': {'cool'}, 'test': {'cool'}}
    mock_sampler.query_batch.return_value = [[0] for _ in range(3)]
    MockNegSampler.return_value = mock_sampler

    hook = TGBNegativeEdgeSamplerHook(dataset_name='tgbl-foo', split_mode='val')
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
    mock_sampler.query_batch.side_effect = ValueError('tgbl-foo')
    MockNegSampler.return_value = mock_sampler

    hook = TGBNegativeEdgeSamplerHook(dataset_name='tgbl-foo', split_mode='val')

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
        'val',
        hook=TGBNegativeEdgeSamplerHook(dataset_name='tgbl-foo', split_mode='val'),
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


def test_bad_tgb_thg_negative_edge_sampler_init():
    min_id = 0
    max_id = 10
    node_type = torch.arange(max_id + 1, dtype=torch.int32)
    with pytest.raises(ValueError):
        TGBTHGNegativeEdgeSamplerHook(
            dataset_name='thgl-software',
            split_mode='invalid_mode',
            first_node_id=min_id,
            last_node_id=max_id,
            node_type=node_type,
        )

    with pytest.raises(ValueError):
        TGBTHGNegativeEdgeSamplerHook(
            dataset_name='thgl-software',
            split_mode='val',
            first_node_id=-1,
            last_node_id=max_id,
            node_type=node_type,
        )

    with pytest.raises(ValueError):
        TGBTHGNegativeEdgeSamplerHook(
            dataset_name='thgl-software',
            split_mode='val',
            first_node_id=min_id,
            last_node_id=100,
            node_type=node_type,
        )

    with pytest.raises(ValueError):
        TGBTHGNegativeEdgeSamplerHook(
            dataset_name='thgl-software',
            split_mode='val',
            first_node_id=min_id,
            last_node_id=max_id,
            node_type=None,
        )


def test_attempt_init_tgb_thg_negative_edge_sampler_not_thg_dataset():
    min_id = 0
    max_id = 10
    node_type = torch.arange(max_id + 1, dtype=torch.int32)
    with pytest.raises(ValueError):
        TGBTHGNegativeEdgeSamplerHook(
            dataset_name='tgbl-wiki',
            split_mode='val',
            first_node_id=min_id,
            last_node_id=max_id,
            node_type=node_type,
        )

    with pytest.raises(ValueError):
        TGBTHGNegativeEdgeSamplerHook(
            dataset_name='tgbn-trade',
            split_mode='val',
            first_node_id=min_id,
            last_node_id=max_id,
            node_type=node_type,
        )

    with pytest.raises(ValueError):
        TGBTHGNegativeEdgeSamplerHook(
            dataset_name='foo',
            split_mode='val',
            first_node_id=min_id,
            last_node_id=max_id,
            node_type=node_type,
        )


@patch('tgb.linkproppred.thg_negative_sampler.THGNegativeEdgeSampler')
def test_thg_negative_edge_sampler(MockNegSampler, thg_data):
    dg = DGraph(thg_data)
    mock_sampler = Mock()
    mock_sampler.eval_set = {'val': {'cool'}, 'test': {'cool'}}
    mock_sampler.query_batch.return_value = [[0] for _ in range(3)]
    MockNegSampler.return_value = mock_sampler

    min_id = 0
    max_id = 10
    node_type = torch.arange(max_id + 1, dtype=torch.int32)

    hook = TGBTHGNegativeEdgeSamplerHook(
        dataset_name='thgl-foo',
        split_mode='val',
        first_node_id=min_id,
        last_node_id=max_id,
        node_type=node_type,
    )
    batch = hook(dg, dg.materialize())
    assert isinstance(batch, DGBatch)
    assert torch.is_tensor(batch.neg)
    assert torch.is_tensor(batch.neg_time)
    assert len(batch.neg_batch_list) == batch.src.shape[0]
    assert batch.neg_time.shape == batch.neg.shape


@patch('tgb.utils.info.DATA_VERSION_DICT', {'thgl-foo': 1})
@patch('tgb.linkproppred.thg_negative_sampler.THGNegativeEdgeSampler')
def test_thg_negative_edge_sampler_with_version_info(MockNegSampler, thg_data):
    dg = DGraph(thg_data)
    mock_sampler = Mock()
    mock_sampler.eval_set = {'val': {'cool'}, 'test': {'cool'}}
    mock_sampler.query_batch.return_value = [[0] for _ in range(3)]
    MockNegSampler.return_value = mock_sampler

    min_id = 0
    max_id = 10
    node_type = torch.arange(max_id + 1, dtype=torch.int32)

    hook = TGBTHGNegativeEdgeSamplerHook(
        dataset_name='thgl-foo',
        split_mode='val',
        first_node_id=min_id,
        last_node_id=max_id,
        node_type=node_type,
    )
    batch = hook(dg, dg.materialize())
    assert isinstance(batch, DGBatch)
    assert torch.is_tensor(batch.neg)
    assert torch.is_tensor(batch.neg_time)
    assert len(batch.neg_batch_list) == batch.src.shape[0]
    assert batch.neg_time.shape == batch.neg.shape


@patch('tgb.linkproppred.thg_negative_sampler.THGNegativeEdgeSampler')
def test_thg_negative_edge_sampler_throws_value_error(MockNegSampler, thg_data):
    dg = DGraph(thg_data)
    mock_sampler = Mock()
    mock_sampler.eval_set = {'val': {'cool'}, 'test': {'cool'}}
    mock_sampler.query_batch.return_value = [[0] for _ in range(3)]
    mock_sampler.query_batch.side_effect = ValueError('thgl-foo')
    MockNegSampler.return_value = mock_sampler

    min_id = 0
    max_id = 10
    node_type = torch.arange(max_id + 1, dtype=torch.int32)

    hook = TGBTHGNegativeEdgeSamplerHook(
        dataset_name='thgl-foo',
        split_mode='val',
        first_node_id=min_id,
        last_node_id=max_id,
        node_type=node_type,
    )

    with pytest.raises(ValueError, match='`pip install --upgrade py-tgb`'):
        hook(dg, dg.materialize())


@pytest.fixture
def thgl_node_only_data():
    edge_index = torch.IntTensor([[1, 2], [2, 3], [3, 4]])
    edge_timestamps = torch.IntTensor([1, 2, 3])
    edge_type = torch.IntTensor([0, 1, 2])
    node_type = torch.arange(7, dtype=torch.int32)
    dynamic_node_feats = torch.rand(2, 5)
    node_timestamps = torch.IntTensor([4, 5])
    node_ids = torch.IntTensor([5, 6])
    return DGData.from_raw(
        edge_timestamps,
        edge_index,
        dynamic_node_feats=dynamic_node_feats,
        node_timestamps=node_timestamps,
        node_ids=node_ids,
        edge_type=edge_type,
        node_type=node_type,
    )


@patch('tgb.linkproppred.thg_negative_sampler.THGNegativeEdgeSampler')
def test_node_only_batch_negative_edge_sampler(MockNegSampler, thgl_node_only_data):
    dg = DGraph(thgl_node_only_data)
    mock_sampler = Mock()
    mock_sampler.eval_set = {'val': {'cool'}, 'test': {'cool'}}
    mock_sampler.query_batch.return_value = [[0] for _ in range(3)]
    MockNegSampler.return_value = mock_sampler

    min_id = 0
    max_id = 10
    node_type = torch.arange(max_id + 1, dtype=torch.int32)

    hm = HookManager(keys=['val', 'test'])
    hm.register(
        'val',
        hook=TGBTHGNegativeEdgeSamplerHook(
            dataset_name='thgl-foo',
            split_mode='val',
            first_node_id=min_id,
            last_node_id=max_id,
            node_type=node_type,
        ),
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


def test_bad_tgb_tkg_negative_edge_sampler_init():
    min_id = 0
    max_id = 10
    with pytest.raises(ValueError):
        TGBTKGNegativeEdgeSamplerHook(
            dataset_name='tkgl-smallpedia',
            split_mode='invalid_mode',
            first_dst_id=min_id,
            last_dst_id=max_id,
        )

    with pytest.raises(ValueError):
        TGBTKGNegativeEdgeSamplerHook(
            dataset_name='tkgl-smallpedia',
            split_mode='val',
            first_dst_id=-1,
            last_dst_id=max_id,
        )

    with pytest.raises(ValueError):
        TGBTKGNegativeEdgeSamplerHook(
            dataset_name='tkgl-smallpedia',
            split_mode='val',
            first_dst_id=min_id,
            last_dst_id=-1,
        )


def test_attempt_init_tgb_tkg_negative_edge_sampler_not_tkg_dataset():
    min_id = 0
    max_id = 10
    with pytest.raises(ValueError):
        TGBTKGNegativeEdgeSamplerHook(
            dataset_name='tgbl-wiki',
            split_mode='val',
            first_dst_id=min_id,
            last_dst_id=max_id,
        )

    with pytest.raises(ValueError):
        TGBTKGNegativeEdgeSamplerHook(
            dataset_name='tgbn-trade',
            split_mode='val',
            first_dst_id=min_id,
            last_dst_id=max_id,
        )

    with pytest.raises(ValueError):
        TGBTKGNegativeEdgeSamplerHook(
            dataset_name='foo',
            split_mode='val',
            first_dst_id=min_id,
            last_dst_id=max_id,
        )


@patch('tgb.linkproppred.tkg_negative_sampler.TKGNegativeEdgeSampler')
def test_tkg_negative_edge_sampler(MockNegSampler, tkg_data):
    dg = DGraph(tkg_data)
    mock_sampler = Mock()
    mock_sampler.eval_set = {'val': {'cool'}, 'test': {'cool'}}
    mock_sampler.query_batch.return_value = [[0] for _ in range(3)]
    MockNegSampler.return_value = mock_sampler

    min_id = 0
    max_id = 10

    hook = TGBTKGNegativeEdgeSamplerHook(
        dataset_name='tkgl-foo',
        split_mode='val',
        first_dst_id=min_id,
        last_dst_id=max_id,
    )
    batch = hook(dg, dg.materialize())
    assert isinstance(batch, DGBatch)
    assert torch.is_tensor(batch.neg)
    assert torch.is_tensor(batch.neg_time)
    assert len(batch.neg_batch_list) == batch.src.shape[0]
    assert batch.neg_time.shape == batch.neg.shape


@patch('tgb.utils.info.DATA_VERSION_DICT', {'tkgl-foo': 1})
@patch('tgb.linkproppred.tkg_negative_sampler.TKGNegativeEdgeSampler')
def test_tkg_negative_edge_sampler_with_version_info(MockNegSampler, tkg_data):
    dg = DGraph(tkg_data)
    mock_sampler = Mock()
    mock_sampler.eval_set = {'val': {'cool'}, 'test': {'cool'}}
    mock_sampler.query_batch.return_value = [[0] for _ in range(3)]
    MockNegSampler.return_value = mock_sampler

    min_id = 0
    max_id = 10

    hook = TGBTKGNegativeEdgeSamplerHook(
        dataset_name='tkgl-foo',
        split_mode='val',
        first_dst_id=min_id,
        last_dst_id=max_id,
    )
    batch = hook(dg, dg.materialize())
    assert isinstance(batch, DGBatch)
    assert torch.is_tensor(batch.neg)
    assert torch.is_tensor(batch.neg_time)
    assert len(batch.neg_batch_list) == batch.src.shape[0]
    assert batch.neg_time.shape == batch.neg.shape


@patch('tgb.linkproppred.tkg_negative_sampler.TKGNegativeEdgeSampler')
def test_tkg_negative_edge_sampler_throws_value_error(MockNegSampler, tkg_data):
    dg = DGraph(tkg_data)
    mock_sampler = Mock()
    mock_sampler.eval_set = {'val': {'cool'}, 'test': {'cool'}}
    mock_sampler.query_batch.return_value = [[0] for _ in range(3)]
    mock_sampler.query_batch.side_effect = ValueError('tkgl-foo')
    MockNegSampler.return_value = mock_sampler

    min_id = 0
    max_id = 10

    hook = TGBTKGNegativeEdgeSamplerHook(
        dataset_name='tkgl-foo',
        split_mode='val',
        first_dst_id=min_id,
        last_dst_id=max_id,
    )

    with pytest.raises(ValueError, match='`pip install --upgrade py-tgb`'):
        hook(dg, dg.materialize())


@pytest.fixture
def tkgl_node_only_data():
    edge_index = torch.IntTensor([[1, 2], [2, 3], [3, 4]])
    edge_timestamps = torch.IntTensor([1, 2, 3])
    edge_type = torch.IntTensor([0, 1, 2])
    dynamic_node_feats = torch.rand(2, 5)
    node_timestamps = torch.IntTensor([4, 5])
    node_ids = torch.IntTensor([5, 6])
    return DGData.from_raw(
        edge_timestamps,
        edge_index,
        dynamic_node_feats=dynamic_node_feats,
        node_timestamps=node_timestamps,
        node_ids=node_ids,
        edge_type=edge_type,
    )


@patch('tgb.linkproppred.tkg_negative_sampler.TKGNegativeEdgeSampler')
def test_node_only_batch_negative_edge_sampler(MockNegSampler, tkgl_node_only_data):
    dg = DGraph(tkgl_node_only_data)
    mock_sampler = Mock()
    mock_sampler.eval_set = {'val': {'cool'}, 'test': {'cool'}}
    mock_sampler.query_batch.return_value = [[0] for _ in range(3)]
    MockNegSampler.return_value = mock_sampler

    min_id = 0
    max_id = 10

    hm = HookManager(keys=['val', 'test'])
    hm.register(
        'val',
        hook=TGBTKGNegativeEdgeSamplerHook(
            dataset_name='tkgl-foo',
            split_mode='val',
            first_dst_id=min_id,
            last_dst_id=max_id,
        ),
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
