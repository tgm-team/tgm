import pytest
import torch

from tgm import DGBatch, DGraph, TimeDeltaDG
from tgm.data import DGData, DGDataLoader
from tgm.exceptions import (
    EmptyBatchError,
    EventOrderedConversionError,
    InvalidDiscretizationError,
)
from tgm.util.seed import seed_everything


@pytest.fixture(autouse=True)
def run_seed_before_tests():
    seed_everything(1337)
    yield


def test_init_ordered_dg_ordered_batch():
    edge_index = torch.IntTensor([[2, 3]])
    edge_timestamps = torch.LongTensor([1])
    data = DGData.from_raw(edge_timestamps, edge_index)
    dg = DGraph(data)
    loader = DGDataLoader(dg)
    assert loader._batch_size == 1
    assert id(dg) == id(loader.dgraph)


@pytest.mark.parametrize('batch_unit', ['Y', 'M', 'W', 'D', 'h', 's', 'ms', 'us', 'ns'])
def test_init_ordered_dg_non_ordered_batch(batch_unit):
    edge_index = torch.IntTensor([[2, 3]])
    edge_timestamps = torch.LongTensor([1])
    data = DGData.from_raw(edge_timestamps, edge_index)
    dg = DGraph(data)
    with pytest.raises(EventOrderedConversionError):
        _ = DGDataLoader(dg, batch_unit=batch_unit)


def test_init_bad_batch_size():
    edge_index = torch.IntTensor([[2, 3]])
    edge_timestamps = torch.LongTensor([1])
    data = DGData.from_raw(edge_timestamps, edge_index)
    dg = DGraph(data)
    with pytest.raises(ValueError):
        _ = DGDataLoader(dg, batch_size=0)


def test_init_bad_on_empty_arg():
    edge_index = torch.IntTensor([[2, 3]])
    edge_timestamps = torch.LongTensor([1])
    data = DGData.from_raw(edge_timestamps, edge_index)
    dg = DGraph(data)
    with pytest.raises(ValueError):
        _ = DGDataLoader(dg, on_empty='foo')


@pytest.mark.parametrize('drop_last', [True, False])
@pytest.mark.parametrize('time_delta', ['r', 's'])
def test_iteration_ordered(drop_last, time_delta):
    edge_index = torch.IntTensor(
        [
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 8],
            [8, 9],
            [9, 10],
            [10, 11],
        ]
    )
    edge_timestamps = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    data = DGData.from_raw(edge_timestamps, edge_index, time_delta=time_delta)
    dg = DGraph(data)
    loader = DGDataLoader(dg, batch_size=3, batch_unit='r', drop_last=drop_last)

    src, dst, t = dg.edges
    batch_num = 0
    for batch in loader:
        assert isinstance(batch, DGBatch)
        torch.testing.assert_close(batch.src, src[3 * batch_num : 3 * (batch_num + 1)])
        torch.testing.assert_close(batch.dst, dst[3 * batch_num : 3 * (batch_num + 1)])
        torch.testing.assert_close(batch.time, t[3 * batch_num : 3 * (batch_num + 1)])
        assert batch_num < 4
        batch_num += 1
    if drop_last:
        assert batch_num == 3
    else:
        assert batch_num == 4


@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_by_time_equal_unit(drop_last):
    edge_index = torch.IntTensor(
        [
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 8],
            [8, 9],
            [9, 10],
            [10, 11],
        ]
    )
    edge_timestamps = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    data = DGData.from_raw(edge_timestamps, edge_index, time_delta='s')
    dg = DGraph(data)
    loader = DGDataLoader(
        dg,
        batch_size=3,
        batch_unit='s',
        drop_last=drop_last,
    )

    src, dst, t = dg.edges
    batch_num = 0
    for batch in loader:
        assert isinstance(batch, DGBatch)
        torch.testing.assert_close(batch.src, src[3 * batch_num : 3 * (batch_num + 1)])
        torch.testing.assert_close(batch.dst, dst[3 * batch_num : 3 * (batch_num + 1)])
        torch.testing.assert_close(batch.time, t[3 * batch_num : 3 * (batch_num + 1)])
        assert batch_num < 4
        batch_num += 1
    if drop_last:
        assert batch_num == 3
    else:
        assert batch_num == 4


@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_by_time_with_conversion_time_delta_value(drop_last):
    edge_index = torch.IntTensor(
        [
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 8],
            [8, 9],
            [9, 10],
        ]
    )
    edge_timestamps = torch.LongTensor([0, 0, 1, 1, 1, 12, 18, 24, 24])
    data = DGData.from_raw(
        edge_timestamps, edge_index, time_delta=TimeDeltaDG('s', value=10)
    )
    dg = DGraph(data)
    loader = DGDataLoader(dg, batch_size=2, batch_unit='m', drop_last=drop_last)

    src, _, _ = dg.edges
    batch_num = 0
    for batch in loader:
        assert isinstance(batch, DGBatch)
        if batch_num == 0:
            torch.testing.assert_close(batch.src, src[:5])
        elif batch_num == 1:
            torch.testing.assert_close(batch.src, src[5:7])
        elif batch_num == 2:
            torch.testing.assert_close(batch.src, src[7:])
        else:
            assert False
        batch_num += 1
    if drop_last:
        assert batch_num == 2
    else:
        assert batch_num == 3


def test_iteration_non_ordered_dg_non_ordered_batch_unit_too_granular():
    edge_index = torch.IntTensor([[2, 3]])
    edge_timestamps = torch.LongTensor([1])
    data = DGData.from_raw(edge_timestamps, edge_index, time_delta='m')
    dg = DGraph(data)
    with pytest.raises(InvalidDiscretizationError):
        # Seconds are too granular of an iteration unit for DG with minute time granularity
        _ = DGDataLoader(dg, batch_unit='s')

    data = DGData.from_raw(
        edge_timestamps, edge_index, time_delta=TimeDeltaDG('s', value=30)
    )
    dg = DGraph(data)
    with pytest.raises(InvalidDiscretizationError):
        # Seconds are too granular of an iteration unit for DG with 'every 30 seconds' time granularity
        _ = DGDataLoader(dg, batch_unit='s')


def test_iteration_with_only_node_events_is_non_empty():
    edge_index = torch.IntTensor([[2, 3], [2, 3]])
    edge_timestamps = torch.LongTensor([1, 5])
    node_timestamps = torch.LongTensor([2, 3, 4])
    node_ids = torch.IntTensor([2, 2, 2])

    # Can't actually get node events without dynamic node feats
    node_x = torch.rand(3, 3)
    data = DGData.from_raw(
        edge_timestamps,
        edge_index,
        node_timestamps=node_timestamps,
        node_ids=node_ids,
        node_x=node_x,
        time_delta='s',
    )
    dg = DGraph(data)

    loader = DGDataLoader(dg, batch_unit='s')
    assert len(loader) == 5

    num_yielded = 0
    for _ in loader:
        num_yielded += 1
        continue
    assert num_yielded == 5


def test_iteration_with_empty_batch():
    edge_index = torch.IntTensor([[2, 3], [2, 3]])
    edge_timestamps = torch.LongTensor([1, 5])
    data = DGData.from_raw(edge_timestamps, edge_index, time_delta='s')
    dg = DGraph(data)

    loader = DGDataLoader(dg, batch_unit='s')
    assert len(loader) == 5  # Includes skipped batches

    num_yielded = 0
    for _ in loader:
        num_yielded += 1
        continue
    assert num_yielded == 2


def test_iteration_with_empty_batch_process_empty():
    edge_index = torch.IntTensor([[2, 3], [2, 3]])
    edge_timestamps = torch.LongTensor([1, 5])
    data = DGData.from_raw(edge_timestamps, edge_index, time_delta='s')
    dg = DGraph(data)

    loader = DGDataLoader(dg, batch_unit='s', on_empty=None)
    assert len(loader) == 5  # Includes skipped batches

    num_yielded = 0
    for _ in loader:
        num_yielded += 1
        continue
    assert num_yielded == 5


def test_iteration_with_empty_batch_raise():
    edge_index = torch.IntTensor([[2, 3], [2, 3]])
    edge_timestamps = torch.LongTensor([1, 5])
    data = DGData.from_raw(edge_timestamps, edge_index, time_delta='s')
    dg = DGraph(data)

    loader = DGDataLoader(dg, batch_unit='s', on_empty='raise')
    assert len(loader) == 5  # Includes skipped batches

    it = iter(loader)
    next(it)  # First batch should yield correctly

    with pytest.raises(EmptyBatchError):
        next(it)
