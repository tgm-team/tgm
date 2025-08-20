from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from tgm import DGData
from tgm.split import TemporalRatioSplit, TemporalSplit
from tgm.timedelta import TimeDeltaDG


def test_time_split_bad_args():
    with pytest.raises(ValueError):
        TemporalSplit(val_time=-1, test_time=0)
    with pytest.raises(ValueError):
        TemporalSplit(val_time=2, test_time=1)


def test_temporal_split():
    edge_times = torch.tensor([1, 2, 3, 4])
    edge_index = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    num_nodes = edge_index.max() + 1
    static_node_feats = torch.rand(num_nodes, 5)

    data = DGData.from_raw(
        edge_timestamps=edge_times,
        edge_index=edge_index,
        static_node_feats=static_node_feats,
    )
    split = TemporalSplit(val_time=3, test_time=4)
    train, val, test = split.apply(data)

    assert isinstance(train, DGData)
    assert isinstance(val, DGData)
    assert isinstance(test, DGData)

    assert train.time_delta == TimeDeltaDG('r')
    assert val.time_delta == TimeDeltaDG('r')
    assert test.time_delta == TimeDeltaDG('r')

    assert train.timestamps.tolist() == [1, 2]
    assert val.timestamps.tolist() == [3]
    assert test.timestamps.tolist() == [4]

    assert train.edge_event_idx.tolist() == [0, 1]
    assert val.edge_event_idx.tolist() == [0]
    assert test.edge_event_idx.tolist() == [0]

    assert train.edge_feats is None
    assert val.edge_feats is None
    assert test.edge_feats is None

    assert train.node_event_idx is None
    assert val.node_event_idx is None
    assert test.node_event_idx is None

    assert id(train.static_node_feats) == id(data.static_node_feats)
    assert id(val.static_node_feats) == id(data.static_node_feats)
    assert id(test.static_node_feats) == id(data.static_node_feats)


def test_temporal_split_with_node_feats():
    edge_times = torch.tensor([1, 2, 3, 4])
    edge_index = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])

    node_times = torch.tensor([0, 2, 4])
    node_ids = torch.tensor([1, 2, 3])
    dynamic_node_feats = torch.rand(3, 7)
    num_nodes = edge_index.max() + 1
    static_node_feats = torch.rand(num_nodes, 5)

    data = DGData.from_raw(
        edge_timestamps=edge_times,
        edge_index=edge_index,
        static_node_feats=static_node_feats,
        node_timestamps=node_times,
        node_ids=node_ids,
        dynamic_node_feats=dynamic_node_feats,
    )
    split = TemporalSplit(val_time=3, test_time=5)
    train, val = split.apply(data)

    assert isinstance(train, DGData)
    assert isinstance(val, DGData)

    assert train.time_delta == TimeDeltaDG('r')
    assert val.time_delta == TimeDeltaDG('r')

    assert train.timestamps.tolist() == [0, 1, 2, 2]
    assert val.timestamps.tolist() == [3, 4, 4]

    assert train.edge_event_idx.tolist() == [1, 2]
    assert val.edge_event_idx.tolist() == [0, 1]

    assert train.edge_feats is None
    assert val.edge_feats is None

    assert train.node_event_idx.tolist() == [0, 3]
    assert val.node_event_idx.tolist() == [2]

    assert train.node_ids.tolist() == [1, 2]
    assert val.node_ids.tolist() == [3]

    torch.testing.assert_close(train.dynamic_node_feats, data.dynamic_node_feats[:2])
    torch.testing.assert_close(val.dynamic_node_feats, data.dynamic_node_feats[2:])

    assert id(train.static_node_feats) == id(data.static_node_feats)
    assert id(val.static_node_feats) == id(data.static_node_feats)


def test_temporal_split_only_train_split():
    edge_times = torch.tensor([1, 2, 3, 4])
    edge_index = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])

    node_times = torch.tensor([0, 2, 4])
    node_ids = torch.tensor([1, 2, 3])
    dynamic_node_feats = torch.rand(3, 7)
    num_nodes = edge_index.max() + 1
    static_node_feats = torch.rand(num_nodes, 5)

    data = DGData.from_raw(
        edge_timestamps=edge_times,
        edge_index=edge_index,
        static_node_feats=static_node_feats,
        node_timestamps=node_times,
        node_ids=node_ids,
        dynamic_node_feats=dynamic_node_feats,
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
    edge_times = torch.tensor([1, 2, 3, 4])
    edge_index = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    num_nodes = edge_index.max() + 1
    static_node_feats = torch.rand(num_nodes, 5)

    data = DGData.from_raw(
        edge_timestamps=edge_times,
        edge_index=edge_index,
        static_node_feats=static_node_feats,
    )
    split = TemporalRatioSplit(train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
    train, val, test = split.apply(data)

    assert isinstance(train, DGData)
    assert isinstance(val, DGData)
    assert isinstance(test, DGData)

    assert train.time_delta == TimeDeltaDG('r')
    assert val.time_delta == TimeDeltaDG('r')
    assert test.time_delta == TimeDeltaDG('r')

    assert train.timestamps.tolist() == [1, 2]
    assert val.timestamps.tolist() == [3]
    assert test.timestamps.tolist() == [4]

    assert train.edge_event_idx.tolist() == [0, 1]
    assert val.edge_event_idx.tolist() == [0]
    assert test.edge_event_idx.tolist() == [0]

    assert train.edge_feats is None
    assert val.edge_feats is None
    assert test.edge_feats is None

    assert train.node_event_idx is None
    assert val.node_event_idx is None
    assert test.node_event_idx is None

    assert id(train.static_node_feats) == id(data.static_node_feats)
    assert id(val.static_node_feats) == id(data.static_node_feats)
    assert id(test.static_node_feats) == id(data.static_node_feats)


def test_temporal_ratio_split_with_node_feats():
    edge_times = torch.tensor([1, 2, 3, 4])
    edge_index = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])

    node_times = torch.tensor([0, 2, 4])
    node_ids = torch.tensor([1, 2, 3])
    dynamic_node_feats = torch.rand(3, 7)
    num_nodes = edge_index.max() + 1
    static_node_feats = torch.rand(num_nodes, 5)

    data = DGData.from_raw(
        edge_timestamps=edge_times,
        edge_index=edge_index,
        static_node_feats=static_node_feats,
        node_timestamps=node_times,
        node_ids=node_ids,
        dynamic_node_feats=dynamic_node_feats,
    )
    split = TemporalRatioSplit(train_ratio=0.5, val_ratio=0.5, test_ratio=0)
    train, val = split.apply(data)

    assert isinstance(train, DGData)
    assert isinstance(val, DGData)

    assert train.time_delta == TimeDeltaDG('r')
    assert val.time_delta == TimeDeltaDG('r')

    assert train.timestamps.tolist() == [0, 1, 2, 2]
    assert val.timestamps.tolist() == [3, 4, 4]

    assert train.edge_event_idx.tolist() == [1, 2]
    assert val.edge_event_idx.tolist() == [0, 1]

    assert train.edge_feats is None
    assert val.edge_feats is None

    assert train.node_event_idx.tolist() == [0, 3]
    assert val.node_event_idx.tolist() == [2]

    assert train.node_ids.tolist() == [1, 2]
    assert val.node_ids.tolist() == [3]

    torch.testing.assert_close(train.dynamic_node_feats, data.dynamic_node_feats[:2])
    torch.testing.assert_close(val.dynamic_node_feats, data.dynamic_node_feats[2:])

    assert id(train.static_node_feats) == id(data.static_node_feats)
    assert id(val.static_node_feats) == id(data.static_node_feats)


def test_temporal_ratio_split_only_train_split():
    edge_times = torch.tensor([1, 2, 3, 4])
    edge_index = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])

    node_times = torch.tensor([0, 2, 4])
    node_ids = torch.tensor([1, 2, 3])
    dynamic_node_feats = torch.rand(3, 7)
    num_nodes = edge_index.max() + 1
    static_node_feats = torch.rand(num_nodes, 5)

    data = DGData.from_raw(
        edge_timestamps=edge_times,
        edge_index=edge_index,
        static_node_feats=static_node_feats,
        node_timestamps=node_times,
        node_ids=node_ids,
        dynamic_node_feats=dynamic_node_feats,
    )
    split = TemporalRatioSplit(train_ratio=1, val_ratio=0, test_ratio=0)
    (train,) = split.apply(data)
    assert isinstance(train, DGData)


@pytest.fixture
def tgb_dataset_factory():
    def _make_dataset(split: str = 'all', with_node_feats: bool = False):
        splits = {'train': 7, 'val': 2, 'test': 1, 'all': 10}
        num_events = splits['all']

        sources = np.random.randint(0, 1000, size=num_events)
        destinations = np.random.randint(0, 1000, size=num_events)
        timestamps = np.arange(num_events)

        train_indices = np.arange(0, splits['train'])
        val_indices = np.arange(splits['train'], splits['train'] + splits['val'])
        test_indices = np.arange(splits['train'] + splits['val'], num_events)

        train_mask_full = np.zeros(num_events, dtype=bool)
        val_mask_full = np.zeros(num_events, dtype=bool)
        test_mask_full = np.zeros(num_events, dtype=bool)
        train_mask_full[train_indices] = True
        val_mask_full[val_indices] = True
        test_mask_full[test_indices] = True

        mock_dataset = MagicMock()
        mock_dataset.num_edges = num_events

        if split == 'all':
            # Full dataset view
            data = {
                'sources': sources,
                'destinations': destinations,
                'timestamps': timestamps,
                'edge_feat': None,
            }
            # keep true masks
            mock_dataset.train_mask = train_mask_full
            mock_dataset.val_mask = val_mask_full
            mock_dataset.test_mask = test_mask_full

        else:
            # Slice the subset
            mask_map = {
                'train': train_mask_full,
                'val': val_mask_full,
                'test': test_mask_full,
            }
            mask = mask_map[split]

            data = {
                'sources': sources[mask],
                'destinations': destinations[mask],
                'timestamps': timestamps[mask],
                'edge_feat': None,
            }

            n = len(data['timestamps'])
            # fabricate dummy masks that match this sliced view
            train_mask = np.ones(n, dtype=bool)
            val_mask = np.ones(n, dtype=bool)
            test_mask = np.ones(n, dtype=bool)

            if split == 'train':
                train_mask[:] = True
            elif split == 'val':
                val_mask[:] = True
            elif split == 'test':
                test_mask[:] = True

            mock_dataset.train_mask = train_mask
            mock_dataset.val_mask = val_mask
            mock_dataset.test_mask = test_mask

        mock_dataset.full_data = data

        # Node count depends on what’s visible
        if split == 'all':
            num_nodes = 1 + max(np.max(sources), np.max(destinations))
        else:
            valid_src, valid_dst = data['sources'], data['destinations']
            num_nodes = 1 + max(np.max(valid_src), np.max(valid_dst))

        mock_dataset.node_feat = (
            np.random.rand(num_nodes, 10) if with_node_feats else None
        )

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
            expected_data = DGData.from_tgb(name)
            actual_data = split_map[split]

            assert expected_data.time_delta == actual_data.time_delta
            assert expected_data.timestamps.tolist() == actual_data.timestamps.tolist()
            assert (
                expected_data.edge_event_idx.tolist()
                == actual_data.edge_event_idx.tolist()
            )
            torch.testing.assert_close(
                data.static_node_feats, actual_data.static_node_feats
            )


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
            expected_data = DGData.from_tgb(name)
            actual_data = split_map[split]

            assert expected_data.time_delta == actual_data.time_delta
            assert expected_data.timestamps.tolist() == actual_data.timestamps.tolist()
            assert (
                expected_data.edge_event_idx.tolist()
                == actual_data.edge_event_idx.tolist()
            )
            torch.testing.assert_close(
                data.static_node_feats, actual_data.static_node_feats
            )

            if split == 'train':
                assert (
                    expected_data.node_event_idx.tolist()
                    == actual_data.node_event_idx.tolist()
                )
                assert expected_data.node_ids.tolist() == actual_data.node_ids.tolist()
                assert (
                    expected_data.dynamic_node_feats.tolist()
                    == actual_data.dynamic_node_feats.tolist()
                )
            else:
                assert actual_data.node_event_idx is None
                assert actual_data.node_ids is None
                assert actual_data.dynamic_node_feats is None
