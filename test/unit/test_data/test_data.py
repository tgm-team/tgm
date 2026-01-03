import csv
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from tgm import TimeDeltaDG
from tgm.constants import PADDED_NODE_ID
from tgm.data import DGData, TemporalRatioSplit
from tgm.data.split import TGBSplit
from tgm.exceptions import (
    EmptyGraphError,
    EventOrderedConversionError,
    InvalidDiscretizationError,
    InvalidNodeIDError,
)


def test_init_dg_data():
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    data = DGData.from_raw(edge_timestamps, edge_index)
    torch.testing.assert_close(data.edge_index, edge_index)
    torch.testing.assert_close(data.timestamps, edge_timestamps)
    torch.testing.assert_close(data.edge_event_idx, torch.IntTensor([0, 1]))
    assert data.edge_feats is None
    assert data.node_event_idx is None
    assert data.node_ids is None
    assert data.dynamic_node_feats is None
    assert data.static_node_feats is None
    assert data.edge_type is None
    assert data.node_type is None
    assert data.time_delta == TimeDeltaDG('r')
    assert data.num_nodes == 21


def test_init_dg_data_with_time_delta():
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    data = DGData.from_raw(edge_timestamps, edge_index, time_delta=TimeDeltaDG('s'))
    assert data.time_delta == TimeDeltaDG('s')


def test_init_dg_data_with_string_time_delta():
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    data = DGData.from_raw(edge_timestamps, edge_index, time_delta='s')
    assert data.time_delta == TimeDeltaDG('s')


def test_init_dg_data_no_node_events_with_edge_features():
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    edge_feats = torch.rand(2, 5)
    data = DGData.from_raw(edge_timestamps, edge_index, edge_feats)
    torch.testing.assert_close(data.edge_index, edge_index)
    torch.testing.assert_close(data.timestamps, edge_timestamps)
    torch.testing.assert_close(data.edge_feats, edge_feats)
    torch.testing.assert_close(data.edge_event_idx, torch.IntTensor([0, 1]))
    assert data.node_event_idx is None
    assert data.node_ids is None
    assert data.dynamic_node_feats is None
    assert data.static_node_feats is None
    assert data.edge_type is None
    assert data.node_type is None
    assert data.time_delta == TimeDeltaDG('r')


def test_init_dg_data_node_events():
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    edge_feats = torch.rand(2, 5)
    node_ids = torch.IntTensor([1, 2, 3])
    node_timestamps = torch.LongTensor([6, 7, 8])
    data = DGData.from_raw(
        edge_timestamps, edge_index, edge_feats, node_timestamps, node_ids
    )
    torch.testing.assert_close(data.edge_index, edge_index)
    torch.testing.assert_close(data.timestamps, torch.LongTensor([1, 5, 6, 7, 8]))
    torch.testing.assert_close(data.edge_feats, edge_feats)
    torch.testing.assert_close(data.edge_event_idx, torch.IntTensor([0, 1]))
    torch.testing.assert_close(data.node_event_idx, torch.IntTensor([2, 3, 4]))
    torch.testing.assert_close(data.node_ids, node_ids)
    assert data.dynamic_node_feats is None
    assert data.static_node_feats is None
    assert data.edge_type is None
    assert data.node_type is None
    assert data.time_delta == TimeDeltaDG('r')
    assert data.num_nodes == 21


def test_init_dg_data_node_events_and_node_features():
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    edge_feats = torch.rand(2, 5)
    node_ids = torch.IntTensor([1, 2, 3])
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
    torch.testing.assert_close(data.edge_event_idx, torch.IntTensor([0, 1]))
    torch.testing.assert_close(data.node_event_idx, torch.IntTensor([2, 3, 4]))
    torch.testing.assert_close(data.node_ids, node_ids)
    torch.testing.assert_close(data.dynamic_node_feats, dynamic_node_feats)
    torch.testing.assert_close(data.static_node_feats, static_node_feats)
    assert data.time_delta == TimeDeltaDG('r')


def test_init_dg_data_sort_required():
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([5, 1])
    edge_feats = torch.rand(2, 5)
    node_ids = torch.IntTensor([1, 2, 3])
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

    exp_edge_index = torch.IntTensor([[10, 20], [2, 3]])
    exp_node_ids = torch.IntTensor([3, 2, 1])
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
    torch.testing.assert_close(data.edge_event_idx, torch.IntTensor([0, 1]))

    torch.testing.assert_close(
        data.node_event_idx,
        torch.IntTensor(
            [2, 3, 4],
        ),
    )
    torch.testing.assert_close(data.node_ids, exp_node_ids)
    torch.testing.assert_close(data.dynamic_node_feats, exp_dynamic_node_feats)
    torch.testing.assert_close(data.static_node_feats, static_node_feats)
    assert data.time_delta == TimeDeltaDG('r')


def test_init_dg_data_bad_args_invalid_node_id():
    edge_index = torch.IntTensor([[PADDED_NODE_ID, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    with pytest.raises(InvalidNodeIDError):
        _ = DGData.from_raw(edge_timestamps, edge_index)

    edge_index = torch.IntTensor([[1, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    node_ids = torch.IntTensor([PADDED_NODE_ID])
    node_timestamps = torch.LongTensor([1])
    with pytest.raises(InvalidNodeIDError):
        _ = DGData.from_raw(
            edge_timestamps,
            edge_index,
            node_ids=node_ids,
            node_timestamps=node_timestamps,
        )


def test_init_dg_data_with_nan_feats():
    edge_index = torch.IntTensor([[0, 0]])
    edge_timestamps = torch.LongTensor([1])
    node_ids = torch.IntTensor([0])
    node_timestamps = torch.LongTensor([2])

    edge_feats = torch.rand((1, 1)).float()
    edge_feats[0][0] = torch.nan

    with pytest.raises(ValueError):
        DGData.from_raw(
            edge_timestamps,
            edge_index,
            node_ids=node_ids,
            node_timestamps=node_timestamps,
            edge_feats=edge_feats,
        )

    node_feats = torch.rand((1, 1)).float()
    node_feats[0][0] = torch.nan

    with pytest.raises(ValueError):
        DGData.from_raw(
            edge_timestamps,
            edge_index,
            node_ids=node_ids,
            node_timestamps=node_timestamps,
            dynamic_node_feats=node_feats,
        )

    with pytest.raises(ValueError):
        DGData.from_raw(
            edge_timestamps,
            edge_index,
            node_ids=node_ids,
            node_timestamps=node_timestamps,
            static_node_feats=node_feats,
        )


def test_init_dg_data_with_downcast_warning():
    edge_index = torch.IntTensor([[0, 1]])
    edge_timestamps = torch.LongTensor([1])
    node_ids = torch.IntTensor([2])
    node_timestamps = torch.LongTensor([2])

    # Fields to test for downcasting
    test_cases = {
        'edge_index': edge_index.long(),
        'node_ids': node_ids.long(),
        'edge_feats': torch.rand((len(edge_index), 1), dtype=torch.float64),
        'dynamic_node_feats': torch.rand((len(node_ids), 1), dtype=torch.float64),
        'static_node_feats': torch.rand((3, 1), dtype=torch.float64),
    }

    for field_name, tensor in test_cases.items():
        kwargs = {
            'edge_timestamps': edge_timestamps,
            'edge_index': edge_index,
            'node_ids': node_ids,
            'node_timestamps': node_timestamps,
        }
        kwargs[field_name] = tensor

        with pytest.warns(UserWarning):
            dg = DGData.from_raw(**kwargs)

        assert dg.edge_index.dtype == torch.int32
        assert dg.node_ids.dtype == torch.int32
        if dg.edge_feats is not None:
            assert dg.edge_feats.dtype == torch.float32
        if dg.dynamic_node_feats is not None:
            assert dg.dynamic_node_feats.dtype == torch.float32
        if dg.static_node_feats is not None:
            assert dg.static_node_feats.dtype == torch.float32


def test_init_dg_data_invalid_node_id_with_id_overflow():
    max_int32 = torch.iinfo(torch.int32).max

    edge_timestamps = torch.LongTensor([0])
    edge_index = torch.LongTensor([[0, max_int32 + 1]])
    with pytest.raises(InvalidNodeIDError):
        DGData.from_raw(edge_timestamps, edge_index)

    edge_timestamps = torch.LongTensor([0])
    edge_index = torch.IntTensor([[0, 1]])
    node_timestamps = torch.LongTensor([0])
    node_ids = torch.LongTensor([max_int32 + 1])
    with pytest.raises(InvalidNodeIDError):
        DGData.from_raw(
            edge_timestamps,
            edge_index,
            node_timestamps=node_timestamps,
            node_ids=node_ids,
        )


def test_init_dg_data_invalid_time_overflow():
    max_int32 = torch.iinfo(torch.int32).max

    edge_timestamps = torch.LongTensor([max_int32 + 1])
    edge_index = torch.IntTensor([[0, 1]])
    with pytest.raises(ValueError):
        DGData.from_raw(edge_timestamps, edge_index)

    edge_timestamps = torch.LongTensor([0])
    edge_index = torch.IntTensor([[0, 1]])
    node_timestamps = torch.LongTensor([max_int32 + 1])
    node_ids = torch.IntTensor([0])
    with pytest.raises(ValueError):
        DGData.from_raw(
            edge_timestamps,
            edge_index,
            node_timestamps=node_timestamps,
            node_ids=node_ids,
        )


def test_init_dg_data_bad_args_empty_graph():
    # Empty graph not supported
    with pytest.raises(EmptyGraphError):
        _ = DGData.from_raw(
            torch.empty(0, dtype=torch.int), torch.empty((0, 2), dtype=torch.int)
        )


def test_init_dg_data_bad_args_empty_node_data():
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([0, 1])
    node_ids = torch.empty(0, dtype=torch.int32)
    node_timestamps = torch.empty(0, dtype=torch.int64)
    with pytest.raises(ValueError):
        _ = DGData.from_raw(
            edge_timestamps,
            edge_index,
            node_ids=node_ids,
            node_timestamps=node_timestamps,
        )


def test_init_dg_Data_bad_static_node_feats():
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([0, 1])
    static_node_feats = torch.rand(0)  # should be 2d
    with pytest.raises(ValueError):
        _ = DGData.from_raw(
            edge_timestamps, edge_index, static_node_feats=static_node_feats
        )


def test_init_dg_data_bad_args_bad_timestamps():
    # Negative timestamps not supported
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    with pytest.raises(ValueError):
        _ = DGData.from_raw(torch.IntTensor([-1, 5]), edge_index)

    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    with pytest.raises(TypeError):
        _ = DGData.from_raw('foo', edge_index)


def test_init_dg_data_bad_args_bad_edge_index():
    edge_timestamps = torch.LongTensor([1, 5])
    with pytest.raises(TypeError):
        _ = DGData.from_raw(edge_timestamps, 'foo')

    with pytest.raises(ValueError):
        _ = DGData.from_raw(edge_timestamps, torch.IntTensor([1, 2]))


def test_init_dg_data_bad_args_bad_edge_feats():
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    with pytest.raises(TypeError):
        _ = DGData.from_raw(edge_timestamps, edge_index, 'foo')

    with pytest.raises(ValueError):
        _ = DGData.from_raw(edge_timestamps, edge_index, torch.rand(1))


def test_init_dg_data_bad_args_bad_node_ids():
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
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
            node_ids=torch.IntTensor([0]),
        )


def test_init_dg_data_bad_args_non_integral_types():
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    node_timestamps = torch.LongTensor([6, 7, 8])
    node_ids = torch.IntTensor([0, 1, 0])

    with pytest.raises(TypeError):
        _ = DGData.from_raw(
            edge_timestamps.float(),
            edge_index,
            node_timestamps=node_timestamps,
            node_ids=node_ids,
        )
    with pytest.raises(TypeError):
        _ = DGData.from_raw(
            edge_timestamps,
            edge_index.float(),
            node_timestamps=node_timestamps,
            node_ids=node_ids,
        )
    with pytest.raises(TypeError):
        _ = DGData.from_raw(
            edge_timestamps,
            edge_index,
            node_timestamps=node_timestamps.float(),
            node_ids=node_ids,
        )
    with pytest.raises(TypeError):
        _ = DGData.from_raw(
            edge_timestamps,
            edge_index,
            node_timestamps=node_timestamps,
            node_ids=node_ids.float(),
        )


def test_init_dg_data_bad_args_bad_dynamic_node_feats():
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    edge_feats = torch.rand(2, 5)
    node_ids = torch.IntTensor([1, 2, 3])
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
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    node_ids = torch.IntTensor([1, 2, 3])
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
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
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
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    node_ids = torch.IntTensor([1, 2, 100])
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
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    timestamps = torch.LongTensor([1, 5])
    edge_feats = torch.rand(2, 5)
    data = DGData.from_raw(
        edge_timestamps=timestamps, edge_index=edge_index, edge_feats=edge_feats
    )

    edge_feats_col = [f'dim_{i}' for i in range(5)]
    col_names = {'edge_src_col': 'src', 'edge_dst_col': 'dst', 'edge_time_col': 't'}

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, newline='')
    tmp_name = tmp.name
    tmp.close()

    try:
        with open(tmp_name, 'w', newline='') as f:
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
        assert data.time_delta == TimeDeltaDG('r')
    except Exception as e:
        raise e
    finally:
        os.remove(tmp_name)


def test_from_csv_with_node_events():
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 1])
    node_ids = torch.IntTensor([7, 8])
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

    tmp_edge = tempfile.NamedTemporaryFile(mode='w', delete=False, newline='')
    tmp_name_edge = tmp_edge.name
    tmp_edge.close()

    tmp_node = tempfile.NamedTemporaryFile(mode='w', delete=False, newline='')
    tmp_name_node = tmp_node.name
    tmp_node.close()

    try:
        with open(tmp_name_edge, 'w', newline='') as edge_file:
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

        with open(tmp_name_node, 'w', newline='') as node_file:
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
        assert data.time_delta == TimeDeltaDG('r')
    except Exception as e:
        raise e
    finally:
        os.remove(tmp_name_edge)
        os.remove(tmp_name_node)


def test_from_csv_with_node_features():
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 1])
    node_ids = torch.IntTensor([7, 8])
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

    tmp_edge = tempfile.NamedTemporaryFile(mode='w', delete=False, newline='')
    tmp_name_edge = tmp_edge.name
    tmp_edge.close()

    tmp_node = tempfile.NamedTemporaryFile(mode='w', delete=False, newline='')
    tmp_name_node = tmp_node.name
    tmp_node.close()

    tmp_static_node = tempfile.NamedTemporaryFile(mode='w', delete=False, newline='')
    tmp_name_static_node = tmp_static_node.name
    tmp_static_node.close()

    try:
        with open(tmp_name_edge, 'w', newline='') as edge_file:
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

        with open(tmp_name_node, 'w', newline='') as node_file:
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

        with open(tmp_name_static_node, 'w', newline='') as static_node_file:
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
        torch.testing.assert_close(
            data.static_node_feats, recovered_data.static_node_feats
        )
        assert data.time_delta == TimeDeltaDG('r')
    except Exception as e:
        raise e
    finally:
        os.remove(tmp_name_edge)
        os.remove(tmp_name_node)
        os.remove(tmp_name_static_node)


def test_from_csv_bad_node_cols_not_specified():
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 1])
    node_ids = torch.IntTensor([7, 8])
    node_timestamps = torch.LongTensor([3, 6])
    dynamic_node_feats = torch.rand(2, 5)

    edge_col_names = {
        'edge_src_col': 'src',
        'edge_dst_col': 'dst',
        'edge_time_col': 't',
    }
    node_col_names = {'node_id_col': 'node_id', 'node_time_col': 'node_time'}
    node_feats_col = [f'dim_{i}' for i in range(5)]

    tmp_edge = tempfile.NamedTemporaryFile(mode='w', delete=False, newline='')
    tmp_name_edge = tmp_edge.name
    tmp_edge.close()

    tmp_node = tempfile.NamedTemporaryFile(mode='w', delete=False, newline='')
    tmp_name_node = tmp_node.name
    tmp_node.close()

    try:
        with open(tmp_name_edge, 'w', newline='') as edge_file:
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

        with open(tmp_name_node, 'w', newline='') as node_file:
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

        with pytest.raises(ValueError):
            DGData.from_csv(
                edge_file_path=edge_file.name,
                node_file_path=node_file.name,
                dynamic_node_feats_col=node_feats_col,
                **edge_col_names,
            )
        with pytest.raises(ValueError):
            DGData.from_csv(
                edge_file_path=edge_file.name,
                static_node_feats_file_path=node_file.name,
                **edge_col_names,
            )

    except Exception as e:
        raise e
    finally:
        os.remove(tmp_name_edge)
        os.remove(tmp_name_node)


def test_from_pandas_with_edge_features():
    events_dict = {
        'src': [2, 10],
        'dst': [3, 20],
        't': [1337, 1338],
        'edge_features': [torch.rand(5).tolist(), torch.rand(5).tolist()],
    }
    events_df = pd.DataFrame(events_dict)
    events_df[['src', 'dst', 't']] = events_df[['src', 'dst', 't']].astype('int32')

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
    assert data.time_delta == TimeDeltaDG('r')


def test_from_pandas_with_node_events():
    edge_dict = {'src': [2, 10], 'dst': [3, 20], 't': [1337, 1338]}
    edge_df = pd.DataFrame(edge_dict)
    edge_df[['src', 'dst', 't']] = edge_df[['src', 'dst', 't']].astype('int32')

    node_dict = {'node': [7, 8], 't': [3, 6]}
    node_df = pd.DataFrame(node_dict)
    node_df[['node', 't']] = node_df[['node', 't']].astype('int32')

    data = DGData.from_pandas(
        edge_df=edge_df,
        edge_src_col='src',
        edge_dst_col='dst',
        edge_time_col='t',
        node_df=node_df,
        node_id_col='node',
        node_time_col='t',
    )
    assert isinstance(data, DGData)
    assert data.edge_index.tolist() == [[2, 3], [10, 20]]
    assert data.timestamps.tolist() == [3, 6, 1337, 1338]
    assert data.node_ids.tolist() == [7, 8]
    assert data.dynamic_node_feats is None
    assert data.time_delta == TimeDeltaDG('r')


def test_from_pandas_with_node_features():
    edge_dict = {'src': [2, 10], 'dst': [3, 20], 't': [1337, 1338]}
    edge_df = pd.DataFrame(edge_dict)
    edge_df[['src', 'dst', 't']] = edge_df[['src', 'dst', 't']].astype('int32')

    node_dict = {
        'node': [7, 8],
        't': [3, 6],
        'node_features': [torch.rand(5).tolist(), torch.rand(5).tolist()],
    }
    node_df = pd.DataFrame(node_dict)
    node_df[['node', 't']] = node_df[['node', 't']].astype('int32')

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
    assert data.time_delta == TimeDeltaDG('r')


def test_from_pandas_with_static_node_features():
    edge_dict = {'src': [2, 10], 'dst': [3, 20], 't': [1337, 1338]}
    edge_df = pd.DataFrame(edge_dict)
    edge_df[['src', 'dst', 't']] = edge_df[['src', 'dst', 't']].astype('int32')

    node_dict = {
        'node_features': [torch.rand(5).tolist() for _ in range(21)],
    }
    node_df = pd.DataFrame(node_dict)

    data = DGData.from_pandas(
        edge_df=edge_df,
        edge_src_col='src',
        edge_dst_col='dst',
        edge_time_col='t',
        static_node_feats_df=node_df,
        static_node_feats_col='node_features',
    )
    assert isinstance(data, DGData)
    torch.testing.assert_close(
        data.static_node_feats.tolist(), node_df.node_features.tolist()
    )


def test_from_pandas_bad_static_node_features_col_no_specified():
    edge_dict = {'src': [2, 10], 'dst': [3, 20], 't': [1337, 1338]}
    edge_df = pd.DataFrame(edge_dict)
    edge_df[['src', 'dst', 't']] = edge_df[['src', 'dst', 't']].astype('int32')

    node_dict = {
        'node_features': [torch.rand(5).tolist() for _ in range(21)],
    }
    node_df = pd.DataFrame(node_dict)

    with pytest.raises(ValueError):
        DGData.from_pandas(
            edge_df=edge_df,
            edge_src_col='src',
            edge_dst_col='dst',
            edge_time_col='t',
            static_node_feats_df=node_df,
        )


def test_from_pandas_bad_node_cols_not_specified():
    edge_dict = {'src': [2, 10], 'dst': [3, 20], 't': [1337, 1338]}
    edge_df = pd.DataFrame(edge_dict)
    edge_df[['src', 'dst', 't']] = edge_df[['src', 'dst', 't']].astype('int32')

    node_dict = {
        'node': [7, 8],
        't': [3, 6],
        'node_features': [torch.rand(5).tolist(), torch.rand(5).tolist()],
    }
    node_df = pd.DataFrame(node_dict)
    node_df[['node', 't']] = node_df[['node', 't']].astype('int32')

    with pytest.raises(ValueError):
        DGData.from_pandas(
            edge_df=edge_df,
            edge_src_col='src',
            edge_dst_col='dst',
            edge_time_col='t',
            node_df=node_df,
        )


@pytest.fixture
def tgb_dataset_factory():
    def _make_dataset(
        split: str = 'all', with_node_feats: bool = False, thgl=False, tkgl=False
    ):
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

        if thgl:
            mock_dataset.full_data['edge_type'] = np.arange(num_events)
            mock_dataset.node_type = np.arange(
                max(sources.max(), destinations.max()) + 1
            )

        if tkgl:
            mock_dataset.full_data['edge_type'] = np.arange(num_events)
            mock_dataset.full_data['w'] = np.random.rand(num_events)

        return mock_dataset

    return _make_dataset


@pytest.fixture
def bad_thgl_dataset_factory():  # Missing edge_type or node_type
    def _make_dataset(split: str = 'all', with_edge_type=True, with_node_type=False):
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
            1 + max(np.max(sources), np.max(destinations))
        else:
            mask = {'train': train_mask, 'val': val_mask, 'test': test_mask}[split]
            valid_src, valid_dst = sources[mask], destinations[mask]
            1 + max(np.max(valid_src), np.max(valid_dst))

        mock_dataset.node_feat = None

        mock_dataset.full_data['node_label_dict'] = {}
        for i in range(5):
            mock_dataset.full_data['node_label_dict'][i] = {i: np.zeros(10)}

        if with_edge_type:
            mock_dataset.full_data['edge_type'] = np.arange(num_events)

        if with_node_type:
            mock_dataset.node_type = np.arange(
                max(sources.max(), destinations.max()) + 1
            )
        else:
            if hasattr(mock_dataset, 'node_type'):
                del mock_dataset.node_type

        return mock_dataset

    return _make_dataset


@pytest.fixture
def bad_tkgl_dataset_factory():  # Missing edge_type
    def _make_dataset(split: str = 'all'):
        num_events, num_train, num_val = 10, 7, 2
        train_indices = np.arange(0, num_train)
        val_indices = np.arange(num_train, num_train + num_val)
        test_indices = np.arange(num_train + num_val, num_events)

        sources = np.random.randint(0, 1000, size=num_events)
        destinations = np.random.randint(0, 1000, size=num_events)
        timestamps = np.arange(num_events)
        edge_feat = None
        w = np.random.rand(num_events, 10)

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
            'w': w,
        }

        if split == 'all':
            1 + max(np.max(sources), np.max(destinations))
        else:
            mask = {'train': train_mask, 'val': val_mask, 'test': test_mask}[split]
            valid_src, valid_dst = sources[mask], destinations[mask]
            1 + max(np.max(valid_src), np.max(valid_dst))

        mock_dataset.node_feat = None

        mock_dataset.full_data['node_label_dict'] = {}
        for i in range(5):
            mock_dataset.full_data['node_label_dict'][i] = {i: np.zeros(10)}

        return mock_dataset

    return _make_dataset


@pytest.fixture
def tgb_seq_dataset_factory():
    def _make_dataset(
        split: str = 'all', with_node_feats: bool = False, with_edge_feats: bool = True
    ):
        num_events, num_train, num_val = 10, 7, 2
        train_indices = np.arange(0, num_train)
        val_indices = np.arange(num_train, num_train + num_val)
        test_indices = np.arange(num_train + num_val, num_events)

        sources = np.random.randint(0, 1000, size=num_events)
        destinations = np.random.randint(0, 1000, size=num_events)
        timestamps = np.arange(num_events)

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
        mock_dataset.src_node_ids = sources
        mock_dataset.dst_node_ids = destinations
        mock_dataset.node_interact_times = timestamps

        if split == 'all':
            num_nodes = 1 + max(np.max(sources), np.max(destinations))
        else:
            mask = {'train': train_mask, 'val': val_mask, 'test': test_mask}[split]
            valid_src, valid_dst = sources[mask], destinations[mask]
            num_nodes = 1 + max(np.max(valid_src), np.max(valid_dst))

        if with_node_feats:
            mock_dataset.node_features = np.random.rand(num_nodes, 10)
        else:
            mock_dataset.node_features = None

        if with_edge_feats:
            mock_dataset.edge_features = np.random.rand(num_events, 10)
        else:
            mock_dataset.edge_features = None

        return mock_dataset

    return _make_dataset


def test_from_bad_tgb_name():
    with pytest.raises(ValueError):
        DGData.from_tgb('foo')


@pytest.mark.parametrize('with_node_feats', [True, False])
@patch('tgb.linkproppred.dataset.LinkPropPredDataset')
@patch.dict('tgm.core.timedelta.TGB_TIME_DELTAS', {'tgbl-wiki': TimeDeltaDG('D')})
def test_from_tgbl(mock_dataset_cls, tgb_dataset_factory, with_node_feats):
    dataset = tgb_dataset_factory(with_node_feats=with_node_feats)
    mock_dataset_cls.return_value = dataset

    mock_native_time_delta = TimeDeltaDG('D')  # Patched value

    def _get_exp_edges():
        src, dst = dataset.full_data['sources'], dataset.full_data['destinations']
        return np.stack([src, dst], axis=1)

    def _get_exp_times():
        return dataset.full_data['timestamps']

    data = DGData.from_tgb(name='tgbl-wiki')
    assert isinstance(data, DGData)
    assert data.time_delta == mock_native_time_delta
    np.testing.assert_allclose(data.edge_index.numpy(), _get_exp_edges())
    np.testing.assert_allclose(data.timestamps.numpy(), _get_exp_times())

    # Confirm correct dataset instantiation
    mock_dataset_cls.assert_called_once_with(name='tgbl-wiki')

    if with_node_feats:
        torch.testing.assert_close(
            data.static_node_feats, torch.Tensor(dataset.node_feat)
        )
    else:
        assert data.static_node_feats is None


@patch('tgb.nodeproppred.dataset.NodePropPredDataset')
@patch.dict('tgm.core.timedelta.TGB_TIME_DELTAS', {'tgbn-trade': TimeDeltaDG('D')})
def test_from_tgbn(mock_dataset_cls, tgb_dataset_factory):
    dataset = tgb_dataset_factory()
    mock_dataset_cls.return_value = dataset

    mock_native_time_delta = TimeDeltaDG('D')  # Patched value

    def _get_exp_edges():
        src, dst = dataset.full_data['sources'], dataset.full_data['destinations']
        return np.stack([src, dst], axis=1)

    def _get_exp_times():
        times = dataset.full_data['timestamps']
        edge_times = times

        # Node times get integrated into the global timestamp array
        node_times = list(dataset.full_data['node_label_dict'].keys())
        node_times = [t for t in node_times if edge_times.min() <= t < edge_times.max()]
        exp_times = np.concatenate([edge_times, node_times])
        exp_times.sort()
        return exp_times

    data = DGData.from_tgb(name='tgbn-trade')
    assert isinstance(data, DGData)
    assert data.time_delta == mock_native_time_delta
    np.testing.assert_allclose(data.edge_index.numpy(), _get_exp_edges())
    np.testing.assert_allclose(data.timestamps.numpy(), _get_exp_times())

    # Assert valid node-centric data
    times = dataset.full_data['timestamps']
    edge_times = times

    full_node_dict = dataset.full_data['node_label_dict']
    split_node_dict = {
        t: v for t, v in full_node_dict.items() if edge_times[0] <= t < edge_times[-1]
    }
    if not len(split_node_dict):
        assert data.node_ids is None
        assert data.dynamic_node_feats is None
    else:
        exp_node_ids, exp_node_feats = [], []
        for node_dict in split_node_dict.values():
            nodes = list(node_dict.keys())[0]
            feats = list(node_dict.values())[0].tolist()
            exp_node_ids.append(nodes)
            exp_node_feats.append(feats)
        assert data.node_ids.tolist() == exp_node_ids
        assert data.dynamic_node_feats.tolist() == exp_node_feats

    # Confirm correct dataset instantiation
    mock_dataset_cls.assert_called_once_with(name='tgbn-trade')


def test_from_bad_tgb_seq_name():
    with pytest.raises(ValueError):
        DGData.from_tgb_seq('foo', root='/tmp')


@pytest.mark.parametrize('with_node_feats', [True, False])
@pytest.mark.parametrize('with_edge_feats', [True, False])
@patch.dict(
    'tgm.core.timedelta.TGB_SEQ_TIME_DELTAS', {'tgb-seq-mock': TimeDeltaDG('s')}
)
@patch('tgb_seq.LinkPred.dataloader.TGBSeqLoader')
def test_from_tgb_seq(
    mock_dataset_cls, tgb_seq_dataset_factory, with_node_feats, with_edge_feats
):
    dataset = tgb_seq_dataset_factory(
        with_node_feats=with_node_feats, with_edge_feats=with_edge_feats
    )
    mock_dataset_cls.return_value = dataset

    mock_native_time_delta = TimeDeltaDG('s')  # Patched value

    def _get_exp_edges():
        src, dst = dataset.src_node_ids, dataset.dst_node_ids
        return np.stack([src, dst], axis=1)

    def _get_exp_times():
        return dataset.node_interact_times

    data = DGData.from_tgb_seq(name='tgb-seq-mock')
    assert isinstance(data, DGData)
    assert data.time_delta == mock_native_time_delta
    np.testing.assert_allclose(data.edge_index.numpy(), _get_exp_edges())
    np.testing.assert_allclose(data.timestamps.numpy(), _get_exp_times())

    # Confirm correct dataset instantiation
    mock_dataset_cls.assert_called_once_with(name='tgb-seq-mock', root='./data')

    if with_node_feats:
        torch.testing.assert_close(
            data.static_node_feats, torch.Tensor(dataset.node_features)
        )
    else:
        assert data.static_node_feats is None

    if with_edge_feats:
        torch.testing.assert_close(data.edge_feats, torch.Tensor(dataset.edge_features))
    else:
        assert data.edge_feats is None


@pytest.mark.parametrize('with_node_feats', [True, False])
@pytest.mark.parametrize('thgl', [True])
@patch('tgb.linkproppred.dataset.LinkPropPredDataset')
@patch.dict('tgm.core.timedelta.TGB_TIME_DELTAS', {'thgl-software': TimeDeltaDG('D')})
def test_from_thgl(mock_dataset_cls, tgb_dataset_factory, with_node_feats, thgl):
    dataset = tgb_dataset_factory(with_node_feats=with_node_feats, thgl=thgl)
    mock_dataset_cls.return_value = dataset

    mock_native_time_delta = TimeDeltaDG('D')  # Patched value

    def _get_exp_edges():
        src, dst = dataset.full_data['sources'], dataset.full_data['destinations']
        return np.stack([src, dst], axis=1)

    def _get_exp_times():
        return dataset.full_data['timestamps']

    def _get_exp_edge_type():
        return dataset.full_data['edge_type']

    def _get_exp_node_type():
        return dataset.node_type

    data = DGData.from_tgb(name='thgl-software')
    assert isinstance(data, DGData)
    assert data.time_delta == mock_native_time_delta
    np.testing.assert_allclose(data.edge_index.numpy(), _get_exp_edges())
    np.testing.assert_allclose(data.timestamps.numpy(), _get_exp_times())
    np.testing.assert_allclose(data.edge_type.numpy(), _get_exp_edge_type())
    np.testing.assert_allclose(data.node_type.numpy(), _get_exp_node_type())

    # Confirm correct dataset instantiation
    mock_dataset_cls.assert_called_once_with(name='thgl-software')

    if with_node_feats:
        torch.testing.assert_close(
            data.static_node_feats, torch.Tensor(dataset.node_feat)
        )
    else:
        assert data.static_node_feats is None


@pytest.mark.parametrize(
    'with_node_type, with_edge_type',
    [(True, False), (False, True)],
)
@patch('tgb.linkproppred.dataset.LinkPropPredDataset')
def test_from_bad_thgl(
    mock_dataset_cls, bad_thgl_dataset_factory, with_node_type, with_edge_type
):
    dataset = bad_thgl_dataset_factory(
        with_node_type=with_node_type, with_edge_type=with_edge_type
    )
    mock_dataset_cls.return_value = dataset
    with pytest.raises(ValueError):
        data = DGData.from_tgb(name='thgl-software')


def test_discretize_reduce_op_bad():
    edge_index = torch.IntTensor([[1, 2], [1, 2], [2, 3], [1, 2], [4, 5]])
    edge_timestamps = torch.LongTensor([1, 2, 3, 63, 65])
    data = DGData.from_raw(
        edge_timestamps,
        edge_index,
        time_delta='m',
    )
    with pytest.raises(ValueError):
        data.discretize('h', reduce_op='foo')


def test_discretize_reduce_op_first():
    edge_index = torch.IntTensor([[1, 2], [1, 2], [2, 3], [1, 2], [4, 5]])
    edge_timestamps = torch.LongTensor([1, 2, 3, 63, 65])
    edge_type = torch.IntTensor([0, 0, 1, 0, 2])
    node_type = torch.arange(6, dtype=torch.int32)
    edge_feats = torch.rand(5, 5)
    static_node_feats = torch.rand(6, 11)
    data = DGData.from_raw(
        edge_timestamps,
        edge_index,
        edge_feats,
        static_node_feats=static_node_feats,
        time_delta='m',
        edge_type=edge_type,
        node_type=node_type,
    )
    new_granularity = TimeDeltaDG('h')
    coarse_data = data.discretize(new_granularity, reduce_op='first')

    assert coarse_data.time_delta == new_granularity
    assert data.time_delta == TimeDeltaDG('m')
    assert id(coarse_data) != id(data)

    exp_timestamps = torch.LongTensor([0, 0, 1, 1])
    exp_edge_event_idx = torch.IntTensor([0, 1, 2, 3])
    exp_edge_index = torch.IntTensor([[1, 2], [2, 3], [1, 2], [4, 5]])
    exp_edge_feats = torch.stack(
        [edge_feats[0], edge_feats[2], edge_feats[3], edge_feats[4]]
    )
    exp_edge_type = torch.IntTensor([0, 1, 0, 2])
    exp_static_node_feats = static_node_feats
    exp_node_type = node_type

    torch.testing.assert_close(coarse_data.timestamps, exp_timestamps)
    torch.testing.assert_close(coarse_data.edge_event_idx, exp_edge_event_idx)
    torch.testing.assert_close(coarse_data.edge_index, exp_edge_index)
    torch.testing.assert_close(coarse_data.edge_feats, exp_edge_feats)
    torch.testing.assert_close(coarse_data.static_node_feats, exp_static_node_feats)
    torch.testing.assert_close(coarse_data.edge_type, exp_edge_type)
    torch.testing.assert_close(coarse_data.node_type, exp_node_type)

    assert coarse_data.node_event_idx is None
    assert coarse_data.node_ids is None
    assert coarse_data.dynamic_node_feats is None


def test_discretize_with_node_events_reduce_op_first():
    edge_index = torch.IntTensor([[1, 2], [1, 2], [2, 3], [1, 2], [4, 5]])
    edge_timestamps = torch.LongTensor([1, 2, 3, 63, 65])
    edge_feats = torch.rand(5, 5)
    edge_type = torch.IntTensor([0, 0, 1, 0, 2])
    node_type = torch.arange(8, dtype=torch.int32)

    node_ids = torch.IntTensor([6, 6, 7, 6, 6, 7])
    node_timestamps = torch.LongTensor([10, 20, 30, 70, 80, 90])
    dynamic_node_feats = torch.rand(6, 5)
    static_node_feats = torch.rand(8, 11)
    data = DGData.from_raw(
        edge_timestamps,
        edge_index,
        edge_feats,
        node_timestamps,
        node_ids,
        dynamic_node_feats,
        static_node_feats,
        time_delta='m',
        edge_type=edge_type,
        node_type=node_type,
    )

    new_granularity = TimeDeltaDG('h')
    coarse_data = data.discretize(new_granularity, reduce_op='first')

    assert coarse_data.time_delta == new_granularity
    assert data.time_delta == TimeDeltaDG('m')
    assert id(coarse_data) != id(data)

    exp_timestamps = torch.LongTensor([0, 0, 0, 0, 1, 1, 1, 1])
    exp_edge_event_idx = torch.IntTensor([0, 1, 4, 5])
    exp_edge_index = torch.IntTensor([[1, 2], [2, 3], [1, 2], [4, 5]])
    exp_edge_feats = torch.stack(
        [edge_feats[0], edge_feats[2], edge_feats[3], edge_feats[4]]
    )
    exp_edge_type = torch.IntTensor([0, 1, 0, 2])
    exp_node_type = node_type
    exp_static_node_feats = static_node_feats

    exp_node_event_idx = torch.IntTensor([2, 3, 6, 7])
    exp_node_ids = torch.IntTensor([6, 7, 6, 7])

    exp_dynamic_node_feats = torch.stack(
        [
            dynamic_node_feats[0],
            dynamic_node_feats[2],
            dynamic_node_feats[3],
            dynamic_node_feats[5],
        ]
    )

    torch.testing.assert_close(coarse_data.timestamps, exp_timestamps)
    torch.testing.assert_close(coarse_data.edge_event_idx, exp_edge_event_idx)
    torch.testing.assert_close(coarse_data.edge_index, exp_edge_index)
    torch.testing.assert_close(coarse_data.edge_feats, exp_edge_feats)
    torch.testing.assert_close(coarse_data.static_node_feats, exp_static_node_feats)

    torch.testing.assert_close(coarse_data.node_event_idx, exp_node_event_idx)
    torch.testing.assert_close(coarse_data.node_ids, exp_node_ids)
    torch.testing.assert_close(coarse_data.dynamic_node_feats, exp_dynamic_node_feats)
    torch.testing.assert_close(coarse_data.edge_type, exp_edge_type)
    torch.testing.assert_close(coarse_data.node_type, exp_node_type)


def test_discretize_no_op():
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    data = DGData.from_raw(edge_timestamps, edge_index)
    coarse_data = data.discretize('r')
    assert id(data) != id(coarse_data)  # No Shared memory

    data = DGData.from_raw(edge_timestamps, edge_index, time_delta='s')
    coarse_data = data.discretize('s')
    assert id(data) != id(coarse_data)  # No Shared memory


def test_discretize_with_huge_ids_no_overflow():
    max_int32 = torch.iinfo(torch.int32).max

    edge_index = torch.IntTensor(
        [[1, 2], [1, 2], [2, 3], [1, 2], [max_int32 - 1, max_int32 - 1]]
    )
    edge_type = torch.IntTensor([0, 0, 1, 0, 2])
    edge_timestamps = torch.LongTensor([1, 2, 3, max_int32 - 1, max_int32 - 1])
    edge_feats = torch.rand(5, 5)
    data = DGData.from_raw(
        edge_timestamps,
        edge_index,
        edge_feats,
        time_delta='m',
        edge_type=edge_type,
    )
    new_granularity = TimeDeltaDG('h')
    coarse_data = data.discretize(new_granularity, reduce_op='first')

    assert coarse_data.time_delta == new_granularity
    assert data.time_delta == TimeDeltaDG('m')
    assert id(coarse_data) != id(data)

    exp_timestamps = torch.LongTensor(
        [0, 0, (max_int32 - 1) // 60, (max_int32 - 1) // 60]
    )
    exp_edge_event_idx = torch.IntTensor([0, 1, 2, 3])
    exp_edge_index = torch.IntTensor(
        [[1, 2], [2, 3], [1, 2], [max_int32 - 1, max_int32 - 1]]
    )
    exp_edge_feats = torch.stack(
        [edge_feats[0], edge_feats[2], edge_feats[3], edge_feats[4]]
    )
    exp_edge_type = torch.stack(
        [edge_type[0], edge_type[2], edge_type[3], edge_type[4]]
    )

    torch.testing.assert_close(coarse_data.timestamps, exp_timestamps)
    torch.testing.assert_close(coarse_data.edge_event_idx, exp_edge_event_idx)
    torch.testing.assert_close(coarse_data.edge_index, exp_edge_index)
    torch.testing.assert_close(coarse_data.edge_feats, exp_edge_feats)
    torch.testing.assert_close(coarse_data.edge_type, exp_edge_type)


def test_discretize_bad_args():
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])

    with pytest.raises(EventOrderedConversionError):
        data = DGData.from_raw(edge_timestamps, edge_index, time_delta='r')
        data.discretize('s')  # Cannot reduce from event-ordered
    with pytest.raises(InvalidDiscretizationError):
        data = DGData.from_raw(edge_timestamps, edge_index, time_delta='h')
        data.discretize('s')  # Cannot reduce into more granular time


def test_split_default_calls_ratio_split(monkeypatch):
    data = DGData.__new__(DGData)
    data._split_strategy = None

    expected = (MagicMock(spec=DGData), MagicMock(spec=DGData), MagicMock(spec=DGData))
    mock_strategy = MagicMock()
    mock_strategy.apply.return_value = expected
    monkeypatch.setattr('tgm.data.dg_data.TemporalRatioSplit', lambda: mock_strategy)

    result = data.split()
    mock_strategy.apply.assert_called_once_with(data)

    assert isinstance(result, tuple)
    assert len(result) == 3
    assert all(isinstance(x, DGData) for x in result)


def test_split_with_explicit_ratio_split():
    data = DGData.__new__(DGData)
    data._split_strategy = None

    strategy = TemporalRatioSplit(0.5, 0.25, 0.25)
    expected = (MagicMock(spec=DGData), MagicMock(spec=DGData), MagicMock(spec=DGData))
    strategy.apply = MagicMock(return_value=expected)

    result = data.split(strategy=strategy)
    strategy.apply.assert_called_once_with(data)

    assert isinstance(result, tuple)
    assert len(result) == 3
    assert all(isinstance(x, DGData) for x in result)


def test_split_uses_tgb_split_when_present():
    data = DGData.__new__(DGData)
    data._split_strategy = None

    expected = (MagicMock(spec=DGData), MagicMock(spec=DGData), MagicMock(spec=DGData))
    mock_strategy = MagicMock(spec=TGBSplit)
    mock_strategy.apply.return_value = expected
    data._split_strategy = mock_strategy

    result = data.split()
    mock_strategy.apply.assert_called_once_with(data)

    assert isinstance(result, tuple)
    assert len(result) == 3
    assert all(isinstance(x, DGData) for x in result)


def test_split_cannot_override_tgb_split():
    data = DGData.__new__(DGData)
    data._split_strategy = TGBSplit({})

    with pytest.raises(ValueError):
        data.split(strategy=TemporalRatioSplit())


def test_clone():
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])

    dg1 = DGData.from_raw(edge_timestamps, edge_index)
    dg2 = dg1.clone()

    assert dg1 is not dg2

    for name, val in dg1.__dict__.items():
        if isinstance(val, torch.Tensor):
            val2 = getattr(dg2, name)
            assert val is not val2
            assert torch.equal(val, val2)

    assert dg1.time_delta == dg2.time_delta
    assert dg1.time_delta is not dg2.time_delta


def test_init_edge_type():
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    edge_type = torch.arange(2, dtype=torch.int32)
    data = DGData.from_raw(
        edge_timestamps,
        edge_index,
        edge_type=edge_type,
    )
    torch.testing.assert_close(data.edge_index, edge_index)
    torch.testing.assert_close(data.timestamps, edge_timestamps)
    torch.testing.assert_close(data.edge_event_idx, torch.IntTensor([0, 1]))
    torch.testing.assert_close(data.edge_type, edge_type)
    assert data.edge_feats is None
    assert data.node_event_idx is None
    assert data.node_ids is None
    assert data.dynamic_node_feats is None
    assert data.static_node_feats is None
    assert data.node_type is None
    assert data.time_delta == TimeDeltaDG('r')


def test_init_bad_edge_type():
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    edge_type = torch.arange(10, dtype=torch.int32)
    with pytest.raises(ValueError):
        data = DGData.from_raw(
            edge_timestamps,
            edge_index,
            edge_type=edge_type,
        )

    edge_type = torch.arange(1, dtype=torch.int32)
    with pytest.raises(ValueError):
        data = DGData.from_raw(
            edge_timestamps,
            edge_index,
            edge_type=edge_type,
        )

    edge_type = torch.arange(2, dtype=torch.float16)
    with pytest.raises(TypeError):
        data = DGData.from_raw(
            edge_timestamps,
            edge_index,
            edge_type=edge_type,
        )

    edge_type = torch.arange(2, dtype=torch.int32)
    with pytest.raises(TypeError):
        data = DGData.from_raw(
            edge_timestamps,
            edge_index,
            edge_type=edge_type.numpy(),
        )

    with pytest.raises(ValueError):
        data = DGData.from_raw(
            edge_timestamps,
            edge_index,
            edge_type=torch.stack([edge_type, edge_type]),
        )

    edge_type = torch.empty(2).fill_(float('nan'))
    with pytest.raises(ValueError):
        data = DGData.from_raw(
            edge_timestamps,
            edge_index,
            edge_type=edge_type,
        )


def test_init_node_type():
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    node_type = torch.arange(21, dtype=torch.int32)
    data = DGData.from_raw(
        edge_timestamps,
        edge_index,
        node_type=node_type,
    )
    torch.testing.assert_close(data.edge_index, edge_index)
    torch.testing.assert_close(data.timestamps, edge_timestamps)
    torch.testing.assert_close(data.edge_event_idx, torch.IntTensor([0, 1]))
    torch.testing.assert_close(data.node_type, node_type)
    assert data.edge_feats is None
    assert data.node_event_idx is None
    assert data.node_ids is None
    assert data.dynamic_node_feats is None
    assert data.static_node_feats is None
    assert data.edge_type is None
    assert data.time_delta == TimeDeltaDG('r')


def test_init_bad_node_type():
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    node_type = torch.arange(2, dtype=torch.int32)
    with pytest.raises(ValueError):
        data = DGData.from_raw(
            edge_timestamps,
            edge_index,
            node_type=node_type,
        )

    node_type = torch.arange(21, dtype=torch.float32)
    with pytest.raises(TypeError):
        data = DGData.from_raw(
            edge_timestamps,
            edge_index,
            node_type=node_type,
        )

    node_type = torch.arange(21, dtype=torch.int32)
    with pytest.raises(TypeError):
        data = DGData.from_raw(
            edge_timestamps,
            edge_index,
            node_type=node_type.numpy(),
        )

    node_type = torch.arange(21, dtype=torch.int32)
    with pytest.raises(ValueError):
        data = DGData.from_raw(
            edge_timestamps,
            edge_index,
            node_type=torch.stack([node_type, node_type]),
        )

    node_type = torch.empty(21).fill_(float('nan'))
    with pytest.raises(ValueError):
        data = DGData.from_raw(
            edge_timestamps,
            edge_index,
            node_type=node_type,
        )


def test_init_edge_node_types():
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    edge_type = torch.arange(2, dtype=torch.int32)
    node_type = torch.arange(21, dtype=torch.int32)
    data = DGData.from_raw(
        edge_timestamps,
        edge_index,
        edge_type=edge_type,
        node_type=node_type,
    )
    torch.testing.assert_close(data.edge_index, edge_index)
    torch.testing.assert_close(data.timestamps, edge_timestamps)
    torch.testing.assert_close(data.edge_event_idx, torch.IntTensor([0, 1]))
    torch.testing.assert_close(data.edge_type, edge_type)
    torch.testing.assert_close(data.node_type, node_type)
    assert data.edge_feats is None
    assert data.node_event_idx is None
    assert data.node_ids is None
    assert data.dynamic_node_feats is None
    assert data.static_node_feats is None
    assert data.time_delta == TimeDeltaDG('r')


def test_init_node_edge_types_with_edge_features():
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    edge_feats = torch.rand(2, 5)
    edge_type = torch.arange(2, dtype=torch.int32)
    node_type = torch.arange(21, dtype=torch.int32)
    data = DGData.from_raw(
        edge_timestamps,
        edge_index,
        edge_feats,
        edge_type=edge_type,
        node_type=node_type,
    )
    torch.testing.assert_close(data.edge_index, edge_index)
    torch.testing.assert_close(data.timestamps, edge_timestamps)
    torch.testing.assert_close(data.edge_feats, edge_feats)
    torch.testing.assert_close(data.edge_event_idx, torch.IntTensor([0, 1]))
    torch.testing.assert_close(data.edge_type, edge_type)
    torch.testing.assert_close(data.node_type, node_type)
    assert data.node_event_idx is None
    assert data.node_ids is None
    assert data.dynamic_node_feats is None
    assert data.static_node_feats is None
    assert data.time_delta == TimeDeltaDG('r')


def test_init_node_edge_types_with_node_features():
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 5])
    static_node_feats = torch.rand(21, 11)
    edge_type = torch.arange(2, dtype=torch.int32)
    node_type = torch.arange(21, dtype=torch.int32)
    data = DGData.from_raw(
        edge_timestamps,
        edge_index,
        static_node_feats=static_node_feats,
        edge_type=edge_type,
        node_type=node_type,
    )
    torch.testing.assert_close(data.edge_index, edge_index)
    torch.testing.assert_close(data.timestamps, edge_timestamps)
    torch.testing.assert_close(data.static_node_feats, static_node_feats)
    torch.testing.assert_close(data.edge_event_idx, torch.IntTensor([0, 1]))
    torch.testing.assert_close(data.edge_type, edge_type)
    torch.testing.assert_close(data.node_type, node_type)
    assert data.node_event_idx is None
    assert data.node_ids is None
    assert data.dynamic_node_feats is None
    assert data.edge_feats is None
    assert data.time_delta == TimeDeltaDG('r')


def test_from_csv_with_edge_type():
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    timestamps = torch.LongTensor([1, 5])
    edge_type = torch.arange(2, dtype=torch.int32)
    data = DGData.from_raw(
        edge_timestamps=timestamps, edge_index=edge_index, edge_type=edge_type
    )

    edge_type_col = 'edge_type'
    col_names = {'edge_src_col': 'src', 'edge_dst_col': 'dst', 'edge_time_col': 't'}

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, newline='')
    tmp_name = tmp.name
    tmp.close()

    try:
        with open(tmp_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(list(col_names.values()) + [edge_type_col])
            writer.writerows(
                zip(
                    edge_index[:, 0].tolist(),
                    edge_index[:, 1].tolist(),
                    timestamps.tolist(),
                    edge_type.tolist(),
                )
            )
            f.flush()

        recovered_data = DGData.from_csv(
            f.name, edge_type_col=edge_type_col, **col_names
        )

        torch.testing.assert_close(data.edge_index, recovered_data.edge_index)
        torch.testing.assert_close(data.timestamps, recovered_data.timestamps)
        torch.testing.assert_close(data.edge_feats, recovered_data.edge_feats)
        torch.testing.assert_close(data.edge_type, recovered_data.edge_type)
        assert data.time_delta == TimeDeltaDG('r')
    except Exception as e:
        raise e
    finally:
        os.remove(tmp_name)


def test_from_csv_with_node_type():
    edge_index = torch.IntTensor([[2, 3], [10, 20]])
    edge_timestamps = torch.LongTensor([1, 1])
    node_ids = torch.IntTensor([7, 8])
    node_type = torch.arange(21, dtype=torch.int32)

    data = DGData.from_raw(
        edge_timestamps=edge_timestamps,
        edge_index=edge_index,
        node_ids=node_ids,
        node_type=node_type,
    )

    torch.testing.assert_close(data.node_type, node_type)

    edge_col_names = {
        'edge_src_col': 'src',
        'edge_dst_col': 'dst',
        'edge_time_col': 't',
    }
    node_type_col = 'node_type'

    tmp_edge = tempfile.NamedTemporaryFile(mode='w', delete=False, newline='')
    tmp_name_edge = tmp_edge.name
    tmp_edge.close()

    tmp_static_node = tempfile.NamedTemporaryFile(mode='w', delete=False, newline='')
    tmp_name_static_node = tmp_static_node.name
    tmp_static_node.close()

    try:
        with open(tmp_name_edge, 'w', newline='') as edge_file:
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

        with open(tmp_name_static_node, 'w', newline='') as static_node_file:
            writer = csv.writer(static_node_file)
            writer.writerow([node_type_col])
            writer.writerows([[i] for i in node_type.tolist()])
            static_node_file.flush()

        recovered_data = DGData.from_csv(
            edge_file_path=edge_file.name,
            static_node_feats_file_path=static_node_file.name,
            node_type_col=node_type_col,
            **edge_col_names,
        )

        torch.testing.assert_close(data.edge_index, recovered_data.edge_index)
        torch.testing.assert_close(data.timestamps, recovered_data.timestamps)
        torch.testing.assert_close(data.node_type, recovered_data.node_type)
        assert data.time_delta == TimeDeltaDG('r')
    except Exception as e:
        raise e
    finally:
        os.remove(tmp_name_edge)
        os.remove(tmp_name_static_node)


def test_from_pandas_with_edge_type():
    events_dict = {
        'src': [2, 10],
        'dst': [3, 20],
        't': [1337, 1338],
        'edge_type': [0, 1],
    }
    events_df = pd.DataFrame(events_dict)
    events_df[['src', 'dst', 't', 'edge_type']] = events_df[
        ['src', 'dst', 't', 'edge_type']
    ].astype('int32')

    data = DGData.from_pandas(
        events_df,
        edge_src_col='src',
        edge_dst_col='dst',
        edge_time_col='t',
        edge_type_col='edge_type',
    )
    assert isinstance(data, DGData)
    assert data.edge_index.tolist() == [[2, 3], [10, 20]]
    assert data.timestamps.tolist() == [1337, 1338]
    assert data.edge_type.tolist() == [0, 1]
    assert data.time_delta == TimeDeltaDG('r')


def test_from_pandas_with_static_node_type():
    edge_dict = {'src': [2, 10], 'dst': [3, 20], 't': [1337, 1338]}
    edge_df = pd.DataFrame(edge_dict)
    edge_df[['src', 'dst', 't']] = edge_df[['src', 'dst', 't']].astype('int32')

    node_dict = {
        'node_type': list(range(21)),
    }
    node_df = pd.DataFrame(node_dict)
    node_df[['node_type']] = node_df[['node_type']].astype('int32')

    data = DGData.from_pandas(
        edge_df=edge_df,
        edge_src_col='src',
        edge_dst_col='dst',
        edge_time_col='t',
        static_node_feats_df=node_df,
        node_type_col='node_type',
    )
    assert isinstance(data, DGData)
    torch.testing.assert_close(data.node_type.tolist(), node_dict['node_type'])


@pytest.mark.parametrize('with_node_feats', [True, False])
@pytest.mark.parametrize('tkgl', [True])
@patch('tgb.linkproppred.dataset.LinkPropPredDataset')
@patch.dict('tgm.core.timedelta.TGB_TIME_DELTAS', {'tkgl-smallpedia': TimeDeltaDG('D')})
def test_from_tkgl(mock_dataset_cls, tgb_dataset_factory, with_node_feats, tkgl):
    dataset = tgb_dataset_factory(with_node_feats=with_node_feats, tkgl=tkgl)
    mock_dataset_cls.return_value = dataset

    mock_native_time_delta = TimeDeltaDG('D')  # Patched value

    def _get_exp_edges():
        src, dst = dataset.full_data['sources'], dataset.full_data['destinations']
        return np.stack([src, dst], axis=1)

    def _get_exp_times():
        return dataset.full_data['timestamps']

    def _get_exp_edge_type():
        return dataset.full_data['edge_type']

    def _get_exp_edge_feat():
        return dataset.full_data['w']

    data = DGData.from_tgb(name='tkgl-smallpedia')
    assert isinstance(data, DGData)
    assert data.time_delta == mock_native_time_delta
    np.testing.assert_allclose(data.edge_index.numpy(), _get_exp_edges())
    np.testing.assert_allclose(data.timestamps.numpy(), _get_exp_times())
    np.testing.assert_allclose(data.edge_type.numpy(), _get_exp_edge_type())
    np.testing.assert_allclose(data.edge_feats.numpy(), _get_exp_edge_feat()[:, None])

    # Confirm correct dataset instantiation
    mock_dataset_cls.assert_called_once_with(name='tkgl-smallpedia')

    if with_node_feats:
        torch.testing.assert_close(
            data.static_node_feats, torch.Tensor(dataset.node_feat)
        )
    else:
        assert data.static_node_feats is None


@patch('tgb.linkproppred.dataset.LinkPropPredDataset')
def test_from_bad_thgl(mock_dataset_cls, bad_tkgl_dataset_factory):
    dataset = bad_tkgl_dataset_factory()
    mock_dataset_cls.return_value = dataset
    with pytest.raises(ValueError):
        data = DGData.from_tgb(name='tkgl-smallpedia')
