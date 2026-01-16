from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from tgm import TimeDeltaDG
from tgm.data import DGData, TemporalRatioSplit, TemporalSplit


def test_time_split_bad_args():
    with pytest.raises(ValueError):
        TemporalSplit(val_time=-1, test_time=0)
    with pytest.raises(ValueError):
        TemporalSplit(val_time=2, test_time=1)


def test_temporal_split():
    edge_time = torch.LongTensor([1, 2, 3, 4])
    edge_index = torch.IntTensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    num_nodes = edge_index.max() + 1
    static_node_x = torch.rand(num_nodes, 5)

    data = DGData.from_raw(
        edge_time=edge_time,
        edge_index=edge_index,
        static_node_x=static_node_x,
    )
    split = TemporalSplit(val_time=3, test_time=4)
    train, val, test = split.apply(data)

    assert train.time_delta == TimeDeltaDG('r')
    assert val.time_delta == TimeDeltaDG('r')
    assert test.time_delta == TimeDeltaDG('r')

    assert train.time.tolist() == [1, 2]
    assert val.time.tolist() == [3]
    assert test.time.tolist() == [4]

    assert train.edge_mask.tolist() == [0, 1]
    assert val.edge_mask.tolist() == [0]
    assert test.edge_mask.tolist() == [0]

    assert train.node_mask is None
    assert val.node_mask is None
    assert test.node_mask is None

    assert id(train.static_node_x) == id(data.static_node_x)
    assert id(val.static_node_x) == id(data.static_node_x)
    assert id(test.static_node_x) == id(data.static_node_x)


def test_temporal_split_with_node_feats():
    edge_time = torch.LongTensor([1, 2, 3, 4])
    edge_index = torch.IntTensor([[1, 2], [3, 4], [5, 6], [7, 8]])

    node_times = torch.LongTensor([1, 2, 4])
    node_x_nids = torch.IntTensor([1, 2, 3])
    node_x = torch.rand(3, 7)
    num_nodes = edge_index.max() + 1
    static_node_x = torch.rand(num_nodes, 5)

    data = DGData.from_raw(
        edge_time=edge_time,
        edge_index=edge_index,
        static_node_x=static_node_x,
        node_x_time=node_times,
        node_x_nids=node_x_nids,
        node_x=node_x,
    )
    split = TemporalSplit(val_time=3, test_time=5)
    train, val = split.apply(data)

    assert train.time_delta == TimeDeltaDG('r')
    assert val.time_delta == TimeDeltaDG('r')

    assert train.time.tolist() == [1, 1, 2, 2]
    assert val.time.tolist() == [3, 4, 4]

    assert train.edge_mask.tolist() == [0, 2]
    assert val.edge_mask.tolist() == [0, 1]

    assert train.node_mask.tolist() == [1, 3]
    assert val.node_mask.tolist() == [2]

    assert train.node_x_nids.tolist() == [1, 2]
    assert val.node_x_nids.tolist() == [3]

    torch.testing.assert_close(train.node_x, data.node_x[:2])
    torch.testing.assert_close(val.node_x, data.node_x[2:])

    assert id(train.static_node_x) == id(data.static_node_x)
    assert id(val.static_node_x) == id(data.static_node_x)


def test_temporal_split_only_train_split():
    edge_time = torch.LongTensor([1, 2, 3, 4])
    edge_index = torch.IntTensor([[1, 2], [3, 4], [5, 6], [7, 8]])

    node_times = torch.LongTensor([0, 2, 4])
    node_x_nids = torch.IntTensor([1, 2, 3])
    node_x = torch.rand(3, 7)
    num_nodes = edge_index.max() + 1
    static_node_x = torch.rand(num_nodes, 5)

    data = DGData.from_raw(
        edge_time=edge_time,
        edge_index=edge_index,
        static_node_x=static_node_x,
        node_x_time=node_times,
        node_x_nids=node_x_nids,
        node_x=node_x,
    )
    split = TemporalSplit(val_time=5, test_time=5)
    (train,) = split.apply(data)
    assert isinstance(train, DGData)


def test_temporal_ratio_split_bad_args():
    with pytest.raises(ValueError):
        TemporalRatioSplit(train_ratio=0.1)
    with pytest.raises(ValueError):
        TemporalRatioSplit(train_ratio=-1, val_ratio=0, test_ratio=1)
    with pytest.raises(ValueError):
        TemporalRatioSplit(train_ratio=0.1, val_ratio=0.1, test_ratio=0.1)
    with pytest.raises(ValueError):
        TemporalRatioSplit(train_ratio=0.4, val_ratio=0.4, test_ratio=0.4)


def test_temporal_ratio_split():
    edge_time = torch.LongTensor([1, 2, 3, 4])
    edge_index = torch.IntTensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    num_nodes = edge_index.max() + 1
    static_node_x = torch.rand(num_nodes, 5)

    data = DGData.from_raw(
        edge_time=edge_time,
        edge_index=edge_index,
        static_node_x=static_node_x,
    )
    split = TemporalRatioSplit(train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
    train, val, test = split.apply(data)

    assert train.time_delta == TimeDeltaDG('r')
    assert val.time_delta == TimeDeltaDG('r')
    assert test.time_delta == TimeDeltaDG('r')

    assert train.time.tolist() == [1, 2]
    assert val.time.tolist() == [3]
    assert test.time.tolist() == [4]

    assert train.edge_mask.tolist() == [0, 1]
    assert val.edge_mask.tolist() == [0]
    assert test.edge_mask.tolist() == [0]

    assert train.node_mask is None
    assert val.node_mask is None
    assert test.node_mask is None

    assert id(train.static_node_x) == id(data.static_node_x)
    assert id(val.static_node_x) == id(data.static_node_x)
    assert id(test.static_node_x) == id(data.static_node_x)


def test_temporal_ratio_split_with_node_type():
    edge_time = torch.LongTensor([1, 2, 3, 4])
    edge_index = torch.IntTensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    num_nodes = edge_index.max() + 1
    node_type = torch.arange(num_nodes, dtype=torch.int32)

    data = DGData.from_raw(
        edge_time=edge_time,
        edge_index=edge_index,
        node_type=node_type,
    )
    split = TemporalRatioSplit(train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
    train, val, test = split.apply(data)

    assert train.time_delta == TimeDeltaDG('r')
    assert val.time_delta == TimeDeltaDG('r')
    assert test.time_delta == TimeDeltaDG('r')

    assert train.time.tolist() == [1, 2]
    assert val.time.tolist() == [3]
    assert test.time.tolist() == [4]

    assert train.edge_mask.tolist() == [0, 1]
    assert val.edge_mask.tolist() == [0]
    assert test.edge_mask.tolist() == [0]

    assert train.node_mask is None
    assert val.node_mask is None
    assert test.node_mask is None

    assert id(train.node_type) == id(data.node_type)
    assert id(val.node_type) == id(data.node_type)
    assert id(test.node_type) == id(data.node_type)


def test_temporal_ratio_split_with_node_feats():
    edge_time = torch.LongTensor([1, 2, 3, 4])
    edge_index = torch.IntTensor([[1, 2], [3, 4], [5, 6], [7, 8]])

    node_times = torch.LongTensor([1, 2, 4])
    node_x_nids = torch.IntTensor([1, 2, 3])
    node_x = torch.rand(3, 7)
    num_nodes = edge_index.max() + 1
    static_node_x = torch.rand(num_nodes, 5)

    data = DGData.from_raw(
        edge_time=edge_time,
        edge_index=edge_index,
        static_node_x=static_node_x,
        node_x_time=node_times,
        node_x_nids=node_x_nids,
        node_x=node_x,
    )
    split = TemporalRatioSplit(train_ratio=0.5, val_ratio=0.5, test_ratio=0)
    train, val = split.apply(data)

    assert isinstance(train, DGData)
    assert isinstance(val, DGData)

    assert train.time_delta == TimeDeltaDG('r')
    assert val.time_delta == TimeDeltaDG('r')

    assert train.time.tolist() == [1, 1, 2, 2]
    assert val.time.tolist() == [3, 4, 4]

    assert train.edge_mask.tolist() == [0, 2]
    assert val.edge_mask.tolist() == [0, 1]

    assert train.node_mask.tolist() == [1, 3]
    assert val.node_mask.tolist() == [2]

    assert train.node_x_nids.tolist() == [1, 2]
    assert val.node_x_nids.tolist() == [3]

    torch.testing.assert_close(train.node_x, data.node_x[:2])
    torch.testing.assert_close(val.node_x, data.node_x[2:])

    assert id(train.static_node_x) == id(data.static_node_x)
    assert id(val.static_node_x) == id(data.static_node_x)


def test_temporal_ratio_split_only_train_split():
    edge_time = torch.LongTensor([1, 2, 3, 4])
    edge_index = torch.IntTensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    edge_type = torch.IntTensor([0, 1, 2, 3])

    node_times = torch.LongTensor([0, 2, 4])
    node_x_nids = torch.IntTensor([1, 2, 3])
    node_x = torch.rand(3, 7)
    num_nodes = edge_index.max() + 1
    static_node_x = torch.rand(num_nodes, 5)
    node_type = torch.arange(9, dtype=torch.int32)

    data = DGData.from_raw(
        edge_time=edge_time,
        edge_index=edge_index,
        static_node_x=static_node_x,
        node_x_time=node_times,
        node_x_nids=node_x_nids,
        node_x=node_x,
        edge_type=edge_type,
        node_type=node_type,
    )
    split = TemporalRatioSplit(train_ratio=1, val_ratio=0, test_ratio=0)
    (train,) = split.apply(data)
    assert isinstance(train, DGData)


@pytest.fixture
def tgb_dataset_factory():
    def _make_dataset(split: str = 'all', with_node_feats: bool = False, thgl=False):
        splits = {'train': 7, 'val': 2, 'test': 1, 'all': 10}
        num_events = splits['all']

        sources = np.random.randint(0, 1000, size=num_events)
        destinations = np.random.randint(0, 1000, size=num_events)
        edge_type = np.arange(num_events) if thgl else None
        timestamps = np.arange(num_events)

        train_indices = np.arange(0, splits['train'])
        val_indices = np.arange(splits['train'], splits['train'] + splits['val'])
        test_indices = np.arange(splits['train'] + splits['val'], num_events)

        train_mask = np.zeros(num_events, dtype=bool)
        val_mask = np.zeros(num_events, dtype=bool)
        test_mask = np.zeros(num_events, dtype=bool)
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

        mock_dataset = MagicMock()
        mock_dataset.num_edges = num_events

        data = {
            'sources': sources,
            'destinations': destinations,
            'timestamps': timestamps,
            'edge_feat': None,
            'edge_type': edge_type,
        }
        mock_dataset.train_mask = train_mask
        mock_dataset.val_mask = val_mask
        mock_dataset.test_mask = test_mask

        if split != 'all':
            mask = {'train': train_mask, 'val': val_mask, 'test': test_mask}[split]
            data['sources'] = data['sources'][mask]
            data['destinations'] = data['destinations'][mask]
            data['timestamps'] = data['timestamps'][mask]
            if thgl:
                data['edge_type'] = data['edge_type'][mask]

            # fabricate dummy masks that match this sliced view
            n = len(data['timestamps'])
            mock_dataset.train_mask = np.ones(n, dtype=bool)
            mock_dataset.val_mask = np.ones(n, dtype=bool)
            mock_dataset.test_mask = np.ones(n, dtype=bool)

        num_nodes = 1 + max(np.max(data['sources']), np.max(data['destinations']))

        mock_dataset.full_data = data
        mock_dataset.node_feat = (
            np.random.rand(num_nodes, 10) if with_node_feats else None
        )

        if thgl:
            mock_dataset.node_type = np.arange(num_nodes) if thgl else None

        mock_dataset.full_data['node_label_dict'] = {}
        for i in range(5):
            mock_dataset.full_data['node_label_dict'][i] = {i: np.zeros(10)}

        return mock_dataset

    return _make_dataset


def test_tgbl_split_matches(tgb_dataset_factory):
    name = 'tgbl-wiki'
    loader_path = 'tgb.linkproppred.dataset.LinkPropPredDataset'

    dataset = tgb_dataset_factory(split='all', with_node_feats=True)
    with patch(loader_path, return_value=dataset):
        data = DGData.from_tgb(name)
        train, val, test = data.split()

    split_map = {'train': train, 'val': val, 'test': test}
    for split in ['train', 'val', 'test']:
        dataset_split = tgb_dataset_factory(split=split, with_node_feats=True)
        with patch(loader_path, return_value=dataset_split):
            expected = DGData.from_tgb(name)
            actual = split_map[split]

            torch.testing.assert_close(expected.time, actual.time)
            torch.testing.assert_close(expected.edge_mask, actual.edge_mask)
            torch.testing.assert_close(data.node_x, actual.node_x)
            torch.testing.assert_close(data.static_node_x, actual.static_node_x)


def test_tgbn_split_matches(tgb_dataset_factory):
    name = 'tgbn-trade'
    loader_path = 'tgb.nodeproppred.dataset.NodePropPredDataset'

    dataset = tgb_dataset_factory(split='all', with_node_feats=True)
    with patch(loader_path, return_value=dataset):
        data = DGData.from_tgb(name)
        train, val, test = data.split()

    split_map = {'train': train, 'val': val, 'test': test}
    for split in ['train', 'val', 'test']:
        dataset_split = tgb_dataset_factory(split=split, with_node_feats=True)
        with patch(loader_path, return_value=dataset_split):
            expected = DGData.from_tgb(name)
            actual = split_map[split]

            assert expected.time_delta == actual.time_delta
            torch.testing.assert_close(expected.time, actual.time)
            torch.testing.assert_close(expected.edge_mask, actual.edge_mask)
            torch.testing.assert_close(data.static_node_x, actual.static_node_x)

            if split == 'train':
                torch.testing.assert_close(expected.node_mask, actual.node_mask)
                torch.testing.assert_close(expected.node_x_nids, actual.node_x_nids)
                torch.testing.assert_close(data.node_x, actual.node_x)
            else:
                assert actual.node_mask is None
                assert actual.node_x_nids is None
                assert actual.node_x is None


def test_thgl_split_matches(tgb_dataset_factory):
    name = 'thgl-software'
    loader_path = 'tgb.linkproppred.dataset.LinkPropPredDataset'

    dataset = tgb_dataset_factory(split='all', with_node_feats=False, thgl=True)
    with patch(loader_path, return_value=dataset):
        data = DGData.from_tgb(name)
        train, val, test = data.split()

    split_map = {'train': train, 'val': val, 'test': test}
    for split in ['train', 'val', 'test']:
        dataset_split = tgb_dataset_factory(
            split=split, with_node_feats=False, thgl=True
        )
        with patch(loader_path, return_value=dataset_split):
            expected = DGData.from_tgb(name)
            actual = split_map[split]

            torch.testing.assert_close(expected.time, actual.time)
            torch.testing.assert_close(expected.edge_mask, actual.edge_mask)
            torch.testing.assert_close(expected.edge_type, actual.edge_type)
            torch.testing.assert_close(data.static_node_x, actual.static_node_x)
            torch.testing.assert_close(data.node_type, actual.node_type)
