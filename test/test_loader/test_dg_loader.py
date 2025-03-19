import pytest

from opendg.events import EdgeEvent
from opendg.graph import DGraph
from opendg.loader import DGDataLoader
from opendg.timedelta import TimeDeltaDG


def test_init_ordered_dg_ordered_batch():
    events = [EdgeEvent(t=1, src=2, dst=3)]
    dg = DGraph(events)
    loader = DGDataLoader(dg)
    assert loader._batch_size == 1
    assert loader._batch_unit == 'r'


@pytest.mark.parametrize('batch_unit', ['Y', 'M', 'W', 'D', 'h', 's', 'ms', 'us', 'ns'])
def test_init_ordered_dg_non_ordered_batch(batch_unit):
    events = [EdgeEvent(t=1, src=2, dst=3)]
    dg = DGraph(events)
    with pytest.raises(ValueError):
        _ = DGDataLoader(dg, batch_unit=batch_unit)


@pytest.mark.parametrize('batch_unit', ['Y', 'M', 'W', 'D', 'h', 's', 'ms', 'us', 'ns'])
def test_init_non_ordered_dg_ordered_batch(batch_unit):
    events = [EdgeEvent(t=1, src=2, dst=3)]
    dg = DGraph(events, time_delta=TimeDeltaDG(batch_unit))
    with pytest.raises(ValueError):
        _ = DGDataLoader(dg)


@pytest.mark.parametrize(
    'batch_unit', ['r', 'Y', 'M', 'W', 'D', 'h', 's', 'ms', 'us', 'ns']
)
@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_equal_unit(batch_unit, drop_last):
    events = [
        EdgeEvent(t=0, src=1, dst=2),
        EdgeEvent(t=1, src=2, dst=3),
        EdgeEvent(t=2, src=3, dst=4),
        EdgeEvent(t=3, src=4, dst=5),
        EdgeEvent(t=4, src=5, dst=6),
        EdgeEvent(t=5, src=6, dst=7),
        EdgeEvent(t=6, src=7, dst=8),
        EdgeEvent(t=7, src=8, dst=9),
        EdgeEvent(t=8, src=9, dst=10),
        EdgeEvent(t=9, src=10, dst=11),
    ]
    dg = DGraph(events, time_delta=TimeDeltaDG(batch_unit))
    loader = DGDataLoader(dg, batch_size=3, batch_unit=batch_unit, drop_last=drop_last)

    batch_num = 0
    for batch in loader:
        assert isinstance(batch, DGraph)
        assert batch_num < 4
        if batch_num == 3:
            assert not drop_last
            assert len(batch) == 1
        else:
            assert len(batch) == 3
        assert batch.to_events() == [e for e in events if e.t // 3 == batch_num]
        batch_num += 1

    assert batch_num == 3 if drop_last else 4


@pytest.mark.parametrize(
    'batch_unit', ['r', 'Y', 'M', 'W', 'D', 'h', 's', 'ms', 'us', 'ns']
)
@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_equal_unit_batch_1(batch_unit, drop_last):
    events = [
        EdgeEvent(t=0, src=1, dst=2),
        EdgeEvent(t=0, src=2, dst=3),
        EdgeEvent(t=0, src=3, dst=4),
        EdgeEvent(t=1, src=4, dst=5),
        EdgeEvent(t=1, src=5, dst=6),
        EdgeEvent(t=2, src=6, dst=7),
        EdgeEvent(t=2, src=7, dst=8),
        EdgeEvent(t=3, src=8, dst=9),
        EdgeEvent(t=4, src=9, dst=10),
    ]
    dg = DGraph(events, time_delta=TimeDeltaDG(batch_unit))
    loader = DGDataLoader(dg, batch_unit=batch_unit, drop_last=drop_last)

    batch_num = 0
    for batch in loader:
        assert isinstance(batch, DGraph)
        assert batch_num < 5
        assert len(batch) == 1
        assert batch.to_events() == [e for e in events if e.t == batch_num]
        batch_num += 1
    assert batch_num == 5


@pytest.mark.parametrize(
    'batch_unit', ['r', 'Y', 'M', 'W', 'D', 'h', 's', 'ms', 'us', 'ns']
)
@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_equal_unit_batch_larger_than_dg(batch_unit, drop_last):
    events = [
        EdgeEvent(t=0, src=1, dst=2),
        EdgeEvent(t=0, src=2, dst=3),
        EdgeEvent(t=0, src=3, dst=4),
        EdgeEvent(t=1, src=4, dst=5),
        EdgeEvent(t=1, src=5, dst=6),
        EdgeEvent(t=2, src=6, dst=7),
        EdgeEvent(t=2, src=7, dst=8),
        EdgeEvent(t=3, src=8, dst=9),
        EdgeEvent(t=4, src=9, dst=10),
    ]
    dg = DGraph(events, time_delta=TimeDeltaDG(batch_unit))
    loader = DGDataLoader(dg, batch_size=10, batch_unit=batch_unit, drop_last=drop_last)

    batch_num = 0
    for batch in loader:
        assert not drop_last
        assert isinstance(batch, DGraph)
        assert batch_num < 1
        assert len(batch) == 5
        assert batch.to_events() == events
        batch_num += 1

    assert batch_num == 0 if drop_last else 1


@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_non_ordered_dg_non_ordered_batch_with_conversion(drop_last):
    events = [
        EdgeEvent(t=0, src=1, dst=2),  # Batch 1
        EdgeEvent(t=0, src=2, dst=3),  # Batch 1
        EdgeEvent(t=100, src=3, dst=4),  # Batch 1
        EdgeEvent(t=100, src=4, dst=5),  # Batch 1
        EdgeEvent(t=100, src=5, dst=6),  # Batch 1
        EdgeEvent(t=120, src=6, dst=7),  # Batch 2
        EdgeEvent(t=180, src=7, dst=8),  # Batch 2
        EdgeEvent(t=240, src=8, dst=9),  # Batch 3
        EdgeEvent(t=240, src=9, dst=10),  # Batch 3
    ]
    dg = DGraph(events, time_delta=TimeDeltaDG('s'))
    loader = DGDataLoader(dg, batch_size=2, batch_unit='m', drop_last=drop_last)

    batch_num = 0
    for batch in loader:
        assert isinstance(batch, DGraph)
        if batch_num == 0:
            expected_events = events[:5]
        elif batch_num == 1:
            expected_events = events[5:7]
        elif batch_num == 2:
            expected_events = events[7:]
        else:
            assert False

        assert isinstance(batch, DGraph)
        assert batch.to_events() == expected_events
        batch_num += 1

    assert batch_num == 2 if drop_last else 3


@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_non_ordered_dg_non_ordered_batch_with_conversion_and_time_delta_values_greater_than_1(
    drop_last,
):
    events = [
        EdgeEvent(t=0, src=1, dst=2),  # Batch 1
        EdgeEvent(t=0, src=2, dst=3),  # Batch 1
        EdgeEvent(t=1, src=3, dst=4),  # Batch 1
        EdgeEvent(t=1, src=4, dst=5),  # Batch 1
        EdgeEvent(t=1, src=5, dst=6),  # Batch 1
        EdgeEvent(t=12, src=6, dst=7),  # Batch 2
        EdgeEvent(t=18, src=7, dst=8),  # Batch 2
        EdgeEvent(t=24, src=8, dst=9),  # Batch 3
        EdgeEvent(t=24, src=9, dst=10),  # Batch 3
    ]
    dg = DGraph(events, time_delta=TimeDeltaDG('s', value=10))
    loader = DGDataLoader(dg, batch_size=2, batch_unit='m', drop_last=drop_last)

    batch_num = 0
    for batch in loader:
        assert isinstance(batch, DGraph)
        if batch_num == 0:
            expected_events = events[:5]
        elif batch_num == 1:
            expected_events = events[5:7]
        elif batch_num == 2:
            expected_events = events[7:]
        else:
            assert False

        assert isinstance(batch, DGraph)
        assert batch.to_events() == expected_events
        batch_num += 1

    assert batch_num == 2 if drop_last else 3


def test_iteration_non_ordered_dg_non_ordered_batch_unit_too_granular():
    events = [EdgeEvent(t=1, src=2, dst=3)]
    dg = DGraph(events, time_delta=TimeDeltaDG('m'))
    with pytest.raises(ValueError):
        # Seconds are too granular of an iteration unit for DG with minute time granularity
        _ = DGDataLoader(dg, batch_unit='s')

    dg = DGraph(events, time_delta=TimeDeltaDG('s', value=30))
    with pytest.raises(ValueError):
        # Seconds are too granular of an iteration unit for DG with 'every 30 seconds' time granularity
        _ = DGDataLoader(dg, batch_unit='s')
