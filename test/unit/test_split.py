from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from tgm import DGData
from tgm.split import RatioSplit, TGBSplit, TimeSplit


def make_simple_dgdata(
    edge_times,
    node_times=None,
    with_node_feats=False,
):
    edge_index = torch.arange(len(edge_times)).repeat(2, 1)  # trivial self-loop edges
    torch.arange(len(edge_times))
    node_ids = None
    dynamic_node_feats = None
    if node_times is not None:
        node_ids = torch.arange(len(node_times))
        torch.arange(len(node_times))
        if with_node_feats:
            dynamic_node_feats = torch.randn(len(node_times), 3)

    return DGData.from_raw(
        time_delta=1,
        edge_timestamps=torch.tensor(edge_times),
        edge_index=edge_index,
        edge_feats=None,
        node_timestamps=torch.tensor(node_times) if node_times is not None else None,
        node_ids=node_ids,
        dynamic_node_feats=dynamic_node_feats,
        static_node_feats=None,
    )


def test_time_split_bad_args():
    with pytest.raises(ValueError):
        TimeSplit(val_time=-1, test_time=0)
    with pytest.raises(ValueError):
        TimeSplit(val_time=2, test_time=1)


def test_time_split():
    data = make_simple_dgdata(edge_times=[1, 2, 3, 4])
    split = TimeSplit(val_time=3, test_time=4)
    train, val, test = split.apply(data)

    assert train.edge_timestamps.tolist() == [1, 2]
    assert val.edge_timestamps.tolist() == [3]
    assert test.edge_timestamps.tolist() == [4]


def test_time_split_with_node_feats():
    data = make_simple_dgdata(
        edge_times=[1, 2, 3, 4], node_times=[0, 2, 3, 5], with_node_feats=True
    )
    split = TimeSplit(val_time=3, test_time=5)
    train, val = split.apply(data)[:2]  # test is empty

    # val should include node with time=3, but not node with time=5
    assert 3 in val.node_timestamps.tolist()
    assert 5 not in val.node_timestamps.tolist()
    assert val.dynamic_node_feats is not None


def test_time_split_no_val_split():
    data = make_simple_dgdata(edge_times=[1, 2, 3, 4])
    split = TimeSplit(val_time=5, test_time=6)  # val empty, test non-empty
    train, test = split.apply(data)
    assert len(train.edge_timestamps) == 4
    assert len(test.edge_timestamps) == 0


def test_ratio_split_bad_args():
    with pytest.raises(ValueError):
        RatioSplit(train_ratio=0.1)
    with pytest.raises(ValueError):
        RatioSplit(train_ratio=-1, val_ratio=0, test_ratio=1)
    with pytest.raises(ValueError):
        RatioSplit(train_ratio=0.1, val_ratio=0.1, test_ratio=0.1)
    with pytest.raises(ValueError):
        RatioSplit(train_ratio=0.4, val_ratio=0.4, test_ratio=0.4)


def test_ratio_split():
    data = make_simple_dgdata(edge_times=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    split = RatioSplit(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    train, val, test = split.apply(data)
    # check approximate ratios
    assert len(train.edge_timestamps) == 5
    assert len(val.edge_timestamps) == 2
    assert len(test.edge_timestamps) == 2


def test_ratio_split_with_node_feats():
    data = make_simple_dgdata(
        edge_times=list(range(1, 11)),
        node_times=list(range(0, 12)),
        with_node_feats=True,
    )
    split = RatioSplit(0.5, 0.25, 0.25)
    train, val, test = split.apply(data)
    assert train.dynamic_node_feats is not None
    assert val.dynamic_node_feats is not None
    assert test.dynamic_node_feats is not None


def test_ratio_split_no_train_split():
    data = make_simple_dgdata(edge_times=[1, 2, 3])
    split = RatioSplit(0.0, 0.5, 0.5)
    val, test = split.apply(data)
    assert val.edge_timestamps.numel() > 0
    assert test.edge_timestamps.numel() > 0


def test_ratio_split_no_val_split():
    data = make_simple_dgdata(edge_times=[1, 2, 3, 4])
    split = RatioSplit(0.5, 0.0, 0.5)
    train, test = split.apply(data)
    assert len(train.edge_timestamps) > 0
    assert len(test.edge_timestamps) > 0


def test_ratio_split_train_only_split():
    data = make_simple_dgdata(edge_times=[1, 2, 3])
    split = RatioSplit(1.0, 0.0, 0.0)
    (train,) = split.apply(data)
    assert len(train.edge_timestamps) == 3


# TODO: Share this with other class
@pytest.fixture
def tgb_dataset_factory():
    def _make_dataset(split: str = 'all', with_node_feats: bool = False):
        num_events, num_train, num_val = 10, 7, 2
        train_indices = np.arange(0, num_train)
        val_indices = np.arange(num_train, num_train + num_val)
        test_indices = np.arange(num_train + num_val, num_events)

        sources = np.random.randint(0, 1000, size=num_events)
        destinations = np.random.randint(0, 1000, size=num_events)
        timestamps = np.arange(num_events)
        edge_feat = None

        train_mask = np.zeros(num_events, dtype=bool)
        val_mask = np.zeros(num_events, dtype=bool)
        test_mask = np.zeros(num_events, dtype=bool)

        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

        mock_dataset = MagicMock()
        mock_dataset.train_mask = train_mask
        mock_dataset.val_mask = val_mask
        mock_dataset.test_mask = test_mask
        mock_dataset.num_edges = num_events
        mock_dataset.full_data = {
            'sources': sources,
            'destinations': destinations,
            'timestamps': timestamps,
            'edge_feat': edge_feat,
        }

        if split == 'all':
            num_nodes = 1 + max(np.max(sources), np.max(destinations))
        else:
            mask = {'train': train_mask, 'val': val_mask, 'test': test_mask}[split]
            valid_src, valid_dst = sources[mask], destinations[mask]
            num_nodes = 1 + max(np.max(valid_src), np.max(valid_dst))

        if with_node_feats:
            mock_dataset.node_feat = np.random.rand(num_nodes, 10)
        else:
            mock_dataset.node_feat = None

        mock_dataset.full_data['node_label_dict'] = {}
        for i in range(5):
            mock_dataset.full_data['node_label_dict'][i] = {i: np.zeros(10)}

        return mock_dataset

    return _make_dataset


@pytest.mark.parametrize('name', ['tgbl-wiki', 'tgbn-trade'])
def test_tgb_split_matches_dataset(name, tgb_dataset_factory):
    # get mock dataset
    dataset = tgb_dataset_factory(split='all', with_node_feats=True)

    # patch the loader class depending on dataset type
    if name.startswith('tgbl'):
        loader_path = 'tgb.linkproppred.dataset.LinkPropPredDataset'
    else:
        loader_path = 'tgb.nodeproppred.dataset.NodePropPredDataset'

    with patch(loader_path, return_value=dataset):
        # DGData constructed with no split
        dgdata = DGData.from_tgb(name=name)

        # DGData with split directly from loader
        dg_train = DGData.from_tgb(name=name, split='train')
        dg_val = DGData.from_tgb(name=name, split='val')
        dg_test = DGData.from_tgb(name=name, split='test')

        # TGBSplit applied manually
        split_strategy = TGBSplit(
            dataset.train_mask, dataset.val_mask, dataset.test_mask
        )
        train, val, test = split_strategy.apply(dgdata)

        # now assert they match
        assert torch.equal(train.edge_index, dg_train.edge_index)
        assert torch.equal(val.edge_index, dg_val.edge_index)
        assert torch.equal(test.edge_index, dg_test.edge_index)
