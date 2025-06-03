import csv
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from opendg.data import DGData


def test_init_dg_data_no_node_events():
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    data = DGData.from_raw(edge_timestamps, edge_index)
    torch.testing.assert_close(data.edge_index, edge_index)
    torch.testing.assert_close(data.timestamps, edge_timestamps)
    torch.testing.assert_close(data.edge_event_idx, torch.LongTensor([0, 1]))
    assert data.edge_feats is None
    assert data.node_event_idx is None
    assert data.node_ids is None
    assert data.dynamic_node_feats is None
    assert data.static_node_feats is None


def test_init_dg_data_no_node_events_with_edge_features():
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    edge_feats = torch.rand(2, 5)
    data = DGData.from_raw(edge_timestamps, edge_index, edge_feats)
    torch.testing.assert_close(data.edge_index, edge_index)
    torch.testing.assert_close(data.timestamps, edge_timestamps)
    torch.testing.assert_close(data.edge_feats, edge_feats)
    torch.testing.assert_close(data.edge_event_idx, torch.LongTensor([0, 1]))
    assert data.node_event_idx is None
    assert data.node_ids is None
    assert data.dynamic_node_feats is None
    assert data.static_node_feats is None


def test_init_dg_data_node_events():
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    edge_feats = torch.rand(2, 5)
    node_ids = torch.LongTensor([1, 2, 3])
    node_timestamps = torch.LongTensor([6, 7, 8])
    data = DGData.from_raw(
        edge_timestamps, edge_index, edge_feats, node_timestamps, node_ids
    )
    torch.testing.assert_close(data.edge_index, edge_index)
    torch.testing.assert_close(data.timestamps, torch.LongTensor([1, 5, 6, 7, 8]))
    torch.testing.assert_close(data.edge_feats, edge_feats)
    torch.testing.assert_close(data.edge_event_idx, torch.LongTensor([0, 1]))
    torch.testing.assert_close(data.node_event_idx, torch.LongTensor([2, 3, 4]))
    torch.testing.assert_close(data.node_ids, node_ids)
    assert data.dynamic_node_feats is None
    assert data.static_node_feats is None


def test_init_dg_data_node_events_and_node_features():
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    edge_feats = torch.rand(2, 5)
    node_ids = torch.LongTensor([1, 2, 3])
    node_timestamps = torch.LongTensor([6, 7, 8])
    dynamic_node_feats = torch.rand(3, 7)
    static_node_feats = torch.rand(21, 11)
    data = DGData.from_raw(
        edge_timestamps,
        edge_index,
        edge_feats,
        node_timestamps,
        node_ids,
        dynamic_node_feats,
        static_node_feats,
    )
    torch.testing.assert_close(data.edge_index, edge_index)
    torch.testing.assert_close(data.timestamps, torch.LongTensor([1, 5, 6, 7, 8]))
    torch.testing.assert_close(data.edge_feats, edge_feats)
    torch.testing.assert_close(data.edge_event_idx, torch.LongTensor([0, 1]))
    torch.testing.assert_close(data.node_event_idx, torch.LongTensor([2, 3, 4]))
    torch.testing.assert_close(data.node_ids, node_ids)
    torch.testing.assert_close(data.dynamic_node_feats, dynamic_node_feats)
    torch.testing.assert_close(data.static_node_feats, static_node_feats)


def test_init_dg_data_sort_required():
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([5, 1])
    edge_feats = torch.rand(2, 5)
    node_ids = torch.LongTensor([1, 2, 3])
    node_timestamps = torch.LongTensor([8, 7, 6])
    dynamic_node_feats = torch.rand(3, 7)
    static_node_feats = torch.rand(21, 11)
    data = DGData.from_raw(
        edge_timestamps,
        edge_index,
        edge_feats,
        node_timestamps,
        node_ids,
        dynamic_node_feats,
        static_node_feats,
    )

    exp_edge_index = torch.LongTensor([[10, 20], [2, 3]])
    exp_node_ids = torch.LongTensor([3, 2, 1])
    exp_edge_feats = torch.Tensor([edge_feats[1].tolist(), edge_feats[0].tolist()])
    exp_dynamic_node_feats = torch.Tensor(
        [
            dynamic_node_feats[2].tolist(),
            dynamic_node_feats[1].tolist(),
            dynamic_node_feats[0].tolist(),
        ]
    )
    torch.testing.assert_close(data.edge_index, exp_edge_index)
    torch.testing.assert_close(data.timestamps, torch.LongTensor([1, 5, 6, 7, 8]))
    torch.testing.assert_close(data.edge_feats, exp_edge_feats)
    torch.testing.assert_close(data.edge_event_idx, torch.LongTensor([1, 0]))
    torch.testing.assert_close(
        data.node_event_idx,
        torch.LongTensor(
            [4, 3, 2],
        ),
    )
    torch.testing.assert_close(data.node_ids, exp_node_ids)
    torch.testing.assert_close(data.dynamic_node_feats, exp_dynamic_node_feats)
    torch.testing.assert_close(data.static_node_feats, static_node_feats)


def test_init_dg_data_bad_args_empty_graph():
    # Empty graph not supported
    with pytest.raises(ValueError):
        _ = DGData.from_raw(torch.empty((0, 2)), torch.empty(0))


def test_init_dg_data_bad_args_negative_timestamps():
    # Negative timestamps not supported
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([-1, 5])
    with pytest.raises(ValueError):
        _ = DGData.from_raw(edge_index, edge_timestamps)


def test_init_dg_data_bad_args_bad_types():
    # Num edges = 2, Num nodes = 21, D_edge = 5, Num node events = 3, D_node_dynamic = 7
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    edge_feats = torch.rand(2, 5)
    node_ids = torch.LongTensor([1, 2, 3])
    node_timestamps = torch.LongTensor([6, 7, 8])
    dynamic_node_feats = torch.rand(3, 7)

    with pytest.raises(TypeError):
        _ = DGData.from_raw(edge_timestamps, 'foo')
    with pytest.raises(TypeError):
        _ = DGData.from_raw(edge_timestamps, edge_index, 'foo')
    with pytest.raises(TypeError):
        _ = DGData.from_raw(edge_timestamps, edge_index, edge_feats, 'foo')
    with pytest.raises(TypeError):
        _ = DGData.from_raw(edge_timestamps, edge_index, edge_feats, node_timestamps)
    with pytest.raises(TypeError):
        _ = DGData.from_raw(
            edge_timestamps, edge_index, edge_feats, node_timestamps, node_ids, 'foo'
        )
    with pytest.raises(TypeError):
        _ = DGData.from_raw(
            edge_timestamps,
            edge_index,
            edge_feats,
            node_timestamps,
            node_ids,
            dynamic_node_feats,
            'foo',
        )


def test_init_dg_data_bad_static_node_feats_shape():
    # Num nodes = 21
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    node_ids = torch.LongTensor([1, 2, 3])
    node_timestamps = torch.LongTensor([6, 7, 8])

    with pytest.raises(ValueError):
        _ = DGData.from_raw(
            edge_timestamps,
            edge_index,
            None,
            node_timestamps,
            node_ids,
            None,
            torch.rand(20, 11),  # should be [21, ...]
        )

    # Num nodes = 21
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])

    with pytest.raises(ValueError):
        _ = DGData.from_raw(
            edge_timestamps,
            edge_index,
            None,
            None,
            None,
            None,
            torch.rand(20, 11),  # should be [21, ...]
        )

    # Num nodes = 101
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    node_ids = torch.LongTensor([1, 2, 100])
    node_timestamps = torch.LongTensor([6, 7, 8])

    with pytest.raises(ValueError):
        _ = DGData.from_raw(
            edge_timestamps,
            edge_index,
            None,
            None,
            None,
            None,
            torch.rand(20, 11),  # should be [101, ...]
        )


def test_from_csv_no_features():
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    timestamps = torch.LongTensor([1, 1])
    data = DGData.from_raw(edge_timestamps=timestamps, edge_index=edge_index)

    col_names = {'src_col': 'src', 'dst_col': 'dst', 'time_col': 't'}
    with tempfile.NamedTemporaryFile() as f:
        _write_csv(data, f.name, **col_names)
        recovered_data = DGData.from_csv(f.name, **col_names)

    torch.testing.assert_close(data.edge_index, recovered_data.edge_index)
    torch.testing.assert_close(data.timestamps, recovered_data.timestamps)


def test_from_csv_with_edge_features():
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    timestamps = torch.LongTensor([1, 5])
    edge_feats = torch.rand(2, 5)
    data = DGData.from_raw(
        edge_timestamps=timestamps, edge_index=edge_index, edge_feats=edge_feats
    )

    edge_feature_col = [f'dim_{i}' for i in range(5)]
    col_names = {'src_col': 'src', 'dst_col': 'dst', 'time_col': 't'}
    with tempfile.NamedTemporaryFile() as f:
        _write_csv(data, f.name, edge_feature_col=edge_feature_col, **col_names)
        recovered_data = DGData.from_csv(
            f.name, edge_feature_col=edge_feature_col, **col_names
        )

    torch.testing.assert_close(data.edge_index, recovered_data.edge_index)
    torch.testing.assert_close(data.timestamps, recovered_data.timestamps)
    torch.testing.assert_close(data.edge_feats, recovered_data.edge_feats)


@pytest.mark.skip('TODO: Add node features to IO')
def test_from_csv_with_node_events():
    pass


@pytest.mark.skip('TODO: Add node features to IO')
def test_from_csv_with_node_features():
    pass


def test_from_pandas_no_features():
    events_dict = {
        'src': [2, 10],
        'dst': [3, 20],
        't': [1337, 1338],
    }
    events_df = pd.DataFrame(events_dict)

    data = DGData.from_pandas(events_df, src_col='src', dst_col='dst', time_col='t')
    assert isinstance(data, DGData)
    assert data.edge_index.tolist() == [[2, 3], [10, 20]]
    assert data.timestamps.tolist() == [1337, 1338]


def test_from_pandas_with_edge_features():
    events_dict = {
        'src': [2, 10],
        'dst': [3, 20],
        't': [1337, 1338],
        'edge_features': [torch.rand(5).tolist(), torch.rand(5).tolist()],
    }
    events_df = pd.DataFrame(events_dict)

    data = DGData.from_pandas(
        events_df,
        src_col='src',
        dst_col='dst',
        time_col='t',
        edge_feature_col='edge_features',
    )
    assert isinstance(data, DGData)
    assert data.edge_index.tolist() == [[2, 3], [10, 20]]
    assert data.timestamps.tolist() == [1337, 1338]
    torch.testing.assert_close(
        data.edge_feats.tolist(), events_df.edge_features.tolist()
    )


@pytest.mark.skip('TODO: Add node features to IO')
def test_from_pandas_with_node_events():
    pass


@pytest.mark.skip('TODO: Add node features to IO')
def test_from_pandas_with_node_features():
    pass


# Constants for split sizes
num_events = 157474
num_train = 110232
num_val = 23621
num_test = 23621

# Index boundaries for each split
train_indices = np.arange(0, num_train)
val_indices = np.arange(num_train, num_train + num_val)
test_indices = np.arange(num_train + num_val, num_events)


@pytest.mark.parametrize(
    'split,expected_indices',
    [
        ('train', train_indices),
        ('valid', val_indices),
        ('test', test_indices),
        ('all', np.arange(num_events)),
    ],
)
@patch('tgb.linkproppred.dataset.LinkPropPredDataset')
def test_from_tgb(mock_dataset_cls, split, expected_indices):
    sources = np.random.randint(0, 1000, size=num_events)
    destinations = np.random.randint(0, 1000, size=num_events)
    timestamps = np.arange(num_events)
    edge_feat = None

    train_mask = np.zeros(num_events, dtype=bool)
    train_mask[train_indices] = True

    val_mask = np.zeros(num_events, dtype=bool)
    val_mask[val_indices] = True

    test_mask = np.zeros(num_events, dtype=bool)
    test_mask[test_indices] = True

    mock_dataset = MagicMock()
    mock_dataset.full_data = {
        'sources': sources,
        'destinations': destinations,
        'timestamps': timestamps,
        'edge_feat': edge_feat,
    }
    mock_dataset.train_mask = train_mask
    mock_dataset.val_mask = val_mask
    mock_dataset.test_mask = test_mask
    mock_dataset.num_edges = num_events

    mock_dataset_cls.return_value = mock_dataset

    # Run the function
    data = DGData.from_tgb(name='tgbl-wiki', split=split)

    # Assertions
    assert isinstance(data, DGData)

    edges_list = data.edge_index.tolist()
    times_list = data.timestamps.tolist()
    for i, idx in enumerate(expected_indices[:5]):  # sample a few for sanity check
        assert times_list[i] == int(timestamps[idx])
        assert edges_list[i] == [int(sources[idx]), int(destinations[idx])]

    # Confirm correct dataset instantiation
    mock_dataset_cls.assert_called_once_with(name='tgbl-wiki')


# @pytest.mark.skip('TODO: Add node features to IO')
def test_from_tgb_with_node_events():
    data = DGData.from_tgb(name='tgbn-trade', split='test')


@pytest.mark.skip('TODO: Add node features to IO')
def test_from_tgb_with_node_features():
    pass


def _write_csv(data, fp, src_col, dst_col, time_col, edge_feature_col=None):
    with open(fp, 'w', newline='') as f:
        fieldnames = [src_col, dst_col, time_col]
        if edge_feature_col is not None:
            fieldnames += edge_feature_col
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(data.edge_index)):
            row = {
                src_col: int(data.edge_index[i][0]),
                dst_col: int(data.edge_index[i][1]),
                time_col: int(data.timestamps[i]),
            }
            if data.edge_feats is not None:
                if edge_feature_col is None:
                    raise ValueError(
                        'No feature column provided but events had features'
                    )

                feats = data.edge_feats[i]

                if len(feats.shape) > 1:
                    raise ValueError('Multi-dimensional features not supported')

                if len(feats) != len(edge_feature_col):
                    raise ValueError(
                        f'Got {len(feats)}-dimensional feature tensor but only '
                        f'specified {len(edge_feature_col)} feature column names.'
                    )

                features_list = feats.tolist()
                for feature_col, feature_val in zip(edge_feature_col, features_list):
                    row[feature_col] = feature_val

            writer.writerow(row)
