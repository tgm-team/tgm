import csv
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from tgm.data import DGData
from tgm.timedelta import TimeDeltaDG


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
        _ = DGData.from_raw(torch.empty(0), torch.empty((0, 2)))


def test_init_dg_data_bad_args_bad_timestamps():
    # Negative timestamps not supported
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    with pytest.raises(ValueError):
        _ = DGData.from_raw(torch.LongTensor([-1, 5]), edge_index)

    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    with pytest.raises(TypeError):
        _ = DGData.from_raw('foo', edge_index)


def test_init_dg_data_bad_args_bad_edge_index():
    edge_timestamps = torch.LongTensor([-1, 5])
    with pytest.raises(TypeError):
        _ = DGData.from_raw(edge_timestamps, 'foo')

    with pytest.raises(ValueError):
        _ = DGData.from_raw(edge_timestamps, torch.LongTensor([1, 2]))


def test_init_dg_data_bad_args_bad_edge_feats():
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([-1, 5])
    with pytest.raises(TypeError):
        _ = DGData.from_raw(edge_timestamps, edge_index, 'foo')

    with pytest.raises(ValueError):
        _ = DGData.from_raw(edge_timestamps, edge_index, torch.rand(1))


def test_init_dg_data_bad_args_bad_node_ids():
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    edge_feats = torch.rand(2, 5)
    node_timestamps = torch.LongTensor([6, 7, 8])

    with pytest.raises(TypeError):
        _ = DGData.from_raw(
            edge_timestamps, edge_index, edge_feats, node_timestamps, node_ids='foo'
        )

    with pytest.raises(ValueError):
        _ = DGData.from_raw(
            edge_timestamps,
            edge_index,
            edge_feats,
            node_timestamps,
            node_ids=torch.LongTensor([0]),
        )


def test_init_dg_data_bad_args_bad_dynamic_node_feats():
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    edge_feats = torch.rand(2, 5)
    node_ids = torch.LongTensor([1, 2, 3])
    node_timestamps = torch.LongTensor([6, 7, 8])

    with pytest.raises(TypeError):
        _ = DGData.from_raw(
            edge_timestamps,
            edge_index,
            edge_feats,
            node_timestamps,
            node_ids,
            'foo',
        )
    with pytest.raises(ValueError):
        _ = DGData.from_raw(
            edge_timestamps,
            edge_index,
            edge_feats,
            node_timestamps,
            node_ids,
            torch.rand(1, 7),
        )


def test_init_dg_data_bad_args_bad_static_node_feats():
    # Num nodes = 21
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    node_ids = torch.LongTensor([1, 2, 3])
    node_timestamps = torch.LongTensor([6, 7, 8])

    with pytest.raises(TypeError):
        _ = DGData.from_raw(
            edge_timestamps, edge_index, None, node_timestamps, node_ids, None, 'foo'
        )

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


def test_from_csv_with_edge_features():
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    timestamps = torch.LongTensor([1, 5])
    edge_feats = torch.rand(2, 5)
    data = DGData.from_raw(
        edge_timestamps=timestamps, edge_index=edge_index, edge_feats=edge_feats
    )

    edge_feats_col = [f'dim_{i}' for i in range(5)]
    col_names = {'edge_src_col': 'src', 'edge_dst_col': 'dst', 'edge_time_col': 't'}
    with tempfile.NamedTemporaryFile(mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(list(col_names.values()) + edge_feats_col)
        writer.writerows(
            zip(
                edge_index[:, 0].tolist(),
                edge_index[:, 1].tolist(),
                timestamps.tolist(),
                edge_feats[:, 0].tolist(),
                edge_feats[:, 1].tolist(),
                edge_feats[:, 2].tolist(),
                edge_feats[:, 3].tolist(),
                edge_feats[:, 4].tolist(),
            )
        )
        f.flush()

        recovered_data = DGData.from_csv(
            f.name, edge_feats_col=edge_feats_col, **col_names
        )

    torch.testing.assert_close(data.edge_index, recovered_data.edge_index)
    torch.testing.assert_close(data.timestamps, recovered_data.timestamps)
    torch.testing.assert_close(data.edge_feats, recovered_data.edge_feats)


def test_from_csv_with_node_events():
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 1])
    node_ids = torch.LongTensor([7, 8])
    node_timestamps = torch.LongTensor([3, 6])
    data = DGData.from_raw(
        edge_timestamps=edge_timestamps,
        edge_index=edge_index,
        node_ids=node_ids,
        node_timestamps=node_timestamps,
    )

    edge_col_names = {
        'edge_src_col': 'src',
        'edge_dst_col': 'dst',
        'edge_time_col': 't',
    }
    node_col_names = {'node_id_col': 'node_id', 'node_time_col': 'node_time'}
    with (
        tempfile.NamedTemporaryFile(mode='w') as edge_file,
        tempfile.NamedTemporaryFile(mode='w') as node_file,
    ):
        writer = csv.writer(edge_file)
        writer.writerow(list(edge_col_names.values()))
        writer.writerows(
            zip(
                edge_index[:, 0].tolist(),
                edge_index[:, 1].tolist(),
                edge_timestamps.tolist(),
            )
        )
        edge_file.flush()

        writer = csv.writer(node_file)
        writer.writerow(list(node_col_names.values()))
        writer.writerows(
            zip(
                node_ids.tolist(),
                node_timestamps.tolist(),
            )
        )
        node_file.flush()

        recovered_data = DGData.from_csv(
            edge_file_path=edge_file.name,
            node_file_path=node_file.name,
            **edge_col_names,
            **node_col_names,
        )

    torch.testing.assert_close(data.edge_index, recovered_data.edge_index)
    torch.testing.assert_close(data.timestamps, recovered_data.timestamps)
    torch.testing.assert_close(data.node_ids, recovered_data.node_ids)
    torch.testing.assert_close(
        data.dynamic_node_feats, recovered_data.dynamic_node_feats
    )


def test_from_csv_with_node_features():
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 1])
    node_ids = torch.LongTensor([7, 8])
    node_timestamps = torch.LongTensor([3, 6])
    dynamic_node_feats = torch.rand(2, 5)
    static_node_feats = torch.rand(21, 3)

    data = DGData.from_raw(
        edge_timestamps=edge_timestamps,
        edge_index=edge_index,
        node_ids=node_ids,
        node_timestamps=node_timestamps,
        dynamic_node_feats=dynamic_node_feats,
        static_node_feats=static_node_feats,
    )

    edge_col_names = {
        'edge_src_col': 'src',
        'edge_dst_col': 'dst',
        'edge_time_col': 't',
    }
    node_col_names = {'node_id_col': 'node_id', 'node_time_col': 'node_time'}
    node_feats_col = [f'dim_{i}' for i in range(5)]
    static_node_feats_col = [f'sdim_{i}' for i in range(3)]
    with (
        tempfile.NamedTemporaryFile(mode='w') as edge_file,
        tempfile.NamedTemporaryFile(mode='w') as node_file,
        tempfile.NamedTemporaryFile(mode='w') as static_node_file,
    ):
        writer = csv.writer(edge_file)
        writer.writerow(list(edge_col_names.values()))
        writer.writerows(
            zip(
                edge_index[:, 0].tolist(),
                edge_index[:, 1].tolist(),
                edge_timestamps.tolist(),
            )
        )
        edge_file.flush()

        writer = csv.writer(node_file)
        writer.writerow(list(node_col_names.values()) + node_feats_col)
        writer.writerows(
            zip(
                node_ids.tolist(),
                node_timestamps.tolist(),
                dynamic_node_feats[:, 0].tolist(),
                dynamic_node_feats[:, 1].tolist(),
                dynamic_node_feats[:, 2].tolist(),
                dynamic_node_feats[:, 3].tolist(),
                dynamic_node_feats[:, 4].tolist(),
            )
        )
        node_file.flush()

        writer = csv.writer(static_node_file)
        writer.writerow(static_node_feats_col)
        writer.writerows(
            zip(
                static_node_feats[:, 0].tolist(),
                static_node_feats[:, 1].tolist(),
                static_node_feats[:, 2].tolist(),
            )
        )
        static_node_file.flush()

        recovered_data = DGData.from_csv(
            edge_file_path=edge_file.name,
            node_file_path=node_file.name,
            static_node_feats_file_path=static_node_file.name,
            dynamic_node_feats_col=node_feats_col,
            static_node_feats_col=static_node_feats_col,
            **edge_col_names,
            **node_col_names,
        )

    torch.testing.assert_close(data.edge_index, recovered_data.edge_index)
    torch.testing.assert_close(data.timestamps, recovered_data.timestamps)
    torch.testing.assert_close(data.node_ids, recovered_data.node_ids)
    torch.testing.assert_close(
        data.dynamic_node_feats, recovered_data.dynamic_node_feats
    )
    torch.testing.assert_close(data.static_node_feats, recovered_data.static_node_feats)


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
        edge_src_col='src',
        edge_dst_col='dst',
        edge_time_col='t',
        edge_feats_col='edge_features',
    )
    assert isinstance(data, DGData)
    assert data.edge_index.tolist() == [[2, 3], [10, 20]]
    assert data.timestamps.tolist() == [1337, 1338]
    torch.testing.assert_close(
        data.edge_feats.tolist(), events_df.edge_features.tolist()
    )


def test_from_pandas_with_node_events():
    edge_dict = {
        'src': [2, 10],
        'dst': [3, 20],
        't': [1337, 1338],
    }
    node_dict = {'node': [7, 8], 't': [3, 6]}

    data = DGData.from_pandas(
        edge_df=pd.DataFrame(edge_dict),
        edge_src_col='src',
        edge_dst_col='dst',
        edge_time_col='t',
        node_df=pd.DataFrame(node_dict),
        node_id_col='node',
        node_time_col='t',
    )
    assert isinstance(data, DGData)
    assert data.edge_index.tolist() == [[2, 3], [10, 20]]
    assert data.timestamps.tolist() == [3, 6, 1337, 1338]
    assert data.node_ids.tolist() == [7, 8]
    assert data.dynamic_node_feats is None


def test_from_pandas_with_node_features():
    edge_dict = {
        'src': [2, 10],
        'dst': [3, 20],
        't': [1337, 1338],
    }
    edge_df = pd.DataFrame(edge_dict)
    node_dict = {
        'node': [7, 8],
        't': [3, 6],
        'node_features': [torch.rand(5).tolist(), torch.rand(5).tolist()],
    }
    node_df = pd.DataFrame(node_dict)

    data = DGData.from_pandas(
        edge_df=edge_df,
        edge_src_col='src',
        edge_dst_col='dst',
        edge_time_col='t',
        node_df=node_df,
        node_id_col='node',
        node_time_col='t',
        dynamic_node_feats_col='node_features',
    )
    assert isinstance(data, DGData)
    assert data.edge_index.tolist() == [[2, 3], [10, 20]]
    assert data.timestamps.tolist() == [3, 6, 1337, 1338]
    assert data.node_ids.tolist() == [7, 8]
    torch.testing.assert_close(
        data.dynamic_node_feats.tolist(), node_df.node_features.tolist()
    )


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


@pytest.mark.parametrize(
    'split,with_node_feats',
    [('train', False), ('train', True), ('val', True), ('test', True), ('all', True)],
)
@patch('tgb.linkproppred.dataset.LinkPropPredDataset')
def test_from_tgbl(mock_dataset_cls, tgb_dataset_factory, split, with_node_feats):
    dataset = tgb_dataset_factory(split, with_node_feats)
    mock_dataset_cls.return_value = dataset

    def _get_exp_edges():
        src, dst = dataset.full_data['sources'], dataset.full_data['destinations']
        edges = np.stack([src, dst], axis=1)
        if split == 'all':
            return edges
        mask = getattr(dataset, f'{split}_mask')
        return edges[mask]

    def _get_exp_times():
        times = dataset.full_data['timestamps']
        if split == 'all':
            return times
        mask = getattr(dataset, f'{split}_mask')
        return times[mask]

    data = DGData.from_tgb(name='tgbl-wiki', split=split)
    assert isinstance(data, DGData)
    np.testing.assert_allclose(data.edge_index.numpy(), _get_exp_edges())
    np.testing.assert_allclose(data.timestamps.numpy(), _get_exp_times())

    # Confirm correct dataset instantiation
    mock_dataset_cls.assert_called_once_with(name='tgbl-wiki')

    if with_node_feats:
        torch.testing.assert_close(
            data.static_node_feats, torch.Tensor(dataset.node_feat).double()
        )
    else:
        assert data.static_node_feats is None


@pytest.mark.parametrize('split', ['train', 'val', 'test', 'all'])
@patch('tgb.nodeproppred.dataset.NodePropPredDataset')
def test_from_tgbn(mock_dataset_cls, tgb_dataset_factory, split):
    dataset = tgb_dataset_factory(split)
    mock_dataset_cls.return_value = dataset

    def _get_exp_edges():
        src, dst = dataset.full_data['sources'], dataset.full_data['destinations']
        edges = np.stack([src, dst], axis=1)
        if split == 'all':
            return edges
        mask = getattr(dataset, f'{split}_mask')
        return edges[mask]

    def _get_exp_times():
        times = dataset.full_data['timestamps']
        if split == 'all':
            edge_times = times
        else:
            mask = getattr(dataset, f'{split}_mask')
            edge_times = times[mask]

        node_times = list(dataset.full_data['node_label_dict'].keys())
        node_times = [t for t in node_times if edge_times.min() <= t < edge_times.max()]
        return edge_times, node_times

    data = DGData.from_tgb(name='tgbn-trade', split=split)
    assert isinstance(data, DGData)
    np.testing.assert_allclose(data.edge_index.numpy(), _get_exp_edges())

    edge_times, node_times = _get_exp_times()
    exp_times = np.concatenate([edge_times, node_times])
    exp_times.sort()
    np.testing.assert_allclose(data.timestamps.numpy(), exp_times)

    split_node_dict = {
        t: v
        for t, v in dataset.full_data['node_label_dict'].items()
        if edge_times.min() <= t < edge_times.max()
    }
    if not len(split_node_dict):
        assert data.node_ids is None
        assert data.dynamic_node_feats is None
    else:
        valid_node_ids, valid_node_feats = [], []
        for v in split_node_dict.values():
            valid_node_ids.append(list(v.keys())[0])
            valid_node_feats.append(list(v.values())[0].tolist())
        assert data.node_ids.tolist() == valid_node_ids
        assert data.dynamic_node_feats.tolist() == valid_node_feats

    # Confirm correct dataset instantiation
    mock_dataset_cls.assert_called_once_with(name='tgbn-trade')


@patch('tgb.linkproppred.dataset.LinkPropPredDataset')
def test_from_tgb_time_remap_required_coarser(mock_dataset_cls, tgb_dataset_factory):
    dataset = tgb_dataset_factory()
    mock_dataset_cls.return_value = dataset

    custom_td = TimeDeltaDG('Y')
    mock_td = {'tgbl-foo': TimeDeltaDG('s')}
    with patch.dict('tgm.timedelta.TGB_TIME_DELTAS', mock_td):
        with pytest.raises(ValueError):
            DGData.from_tgb(name='tgbl-foo', time_delta=custom_td)


@patch('tgb.linkproppred.dataset.LinkPropPredDataset')
def test_from_tgb_time_remap_required_finer(mock_dataset_cls, tgb_dataset_factory):
    dataset = tgb_dataset_factory()
    mock_dataset_cls.return_value = dataset

    # Save raw (unmapped timestamps) from the original dataset
    # We do a copy since the TGB constructor modifies timestamp info
    # on the underlying tgb array
    raw_times = dataset.full_data['timestamps'].copy()

    custom_td = TimeDeltaDG('s')
    mock_td = {'tgbl-foo': TimeDeltaDG('Y')}
    with patch.dict('tgm.timedelta.TGB_TIME_DELTAS', mock_td):
        data = DGData.from_tgb(name='tgbl-foo', time_delta=custom_td)

    def _get_exp_edges():
        src, dst = dataset.full_data['sources'], dataset.full_data['destinations']
        return np.stack([src, dst], axis=1)

    def _get_exp_times():
        time_factor = int(mock_td['tgbl-foo'].convert(custom_td))
        return raw_times * time_factor

    assert isinstance(data, DGData)
    np.testing.assert_allclose(data.edge_index.numpy(), _get_exp_edges())
    np.testing.assert_allclose(data.timestamps.numpy(), _get_exp_times())

    # Confirm correct dataset instantiation
    mock_dataset_cls.assert_called_once_with(name='tgbl-foo')


def test_from_any():
    data = 'tgbl-mock'
    with patch.object(DGData, 'from_tgb') as mock_tgb:
        _ = DGData.from_any(data)
        mock_tgb.assert_called_once_with(name=data, time_delta=None)

    data = 'foo.csv'
    with patch.object(DGData, 'from_csv') as mock_csv:
        _ = DGData.from_any(data)
        mock_csv.assert_called_once_with(data)

    data = pd.DataFrame()
    with patch.object(DGData, 'from_pandas') as mock_pandas:
        _ = DGData.from_any(data)
        mock_pandas.assert_called_once_with(data)

    with pytest.raises(ValueError):
        _ = DGData.from_any('foo')
