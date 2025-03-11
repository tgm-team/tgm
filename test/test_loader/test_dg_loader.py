import pytest

from opendg.events import EdgeEvent
from opendg.graph import DGraph
from opendg.loader import DGDataLoader
from opendg.timedelta import TimeDeltaDG, TimeDeltaUnit


def test_init_ordered_dg_ordered_batch():
    events = [EdgeEvent(time=1, edge=(2, 3))]
    dg = DGraph(events)
    loader = DGDataLoader(dg)
    assert loader._batch_size == 1
    assert loader._batch_unit == TimeDeltaUnit.ORDERED


@pytest.mark.parametrize('batch_unit', ['Y', 'M', 'W', 'D', 'h', 's', 'ms', 'us', 'ns'])
def test_init_ordered_dg_non_ordered_batch(batch_unit):
    events = [EdgeEvent(time=1, edge=(2, 3))]
    dg = DGraph(events)
    with pytest.raises(ValueError):
        _ = DGDataLoader(dg, batch_unit=batch_unit)


@pytest.mark.parametrize(
    'batch_unit', ['r', 'Y', 'M', 'W', 'D', 'h', 's', 'ms', 'us', 'ns']
)
@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_equal_unit(batch_unit, drop_last):
    events = [
        EdgeEvent(time=0, edge=(1, 2)),
        EdgeEvent(time=1, edge=(2, 3)),
        EdgeEvent(time=2, edge=(3, 4)),
        EdgeEvent(time=3, edge=(4, 5)),
        EdgeEvent(time=4, edge=(5, 6)),
        EdgeEvent(time=5, edge=(6, 7)),
        EdgeEvent(time=6, edge=(7, 8)),
        EdgeEvent(time=7, edge=(8, 9)),
        EdgeEvent(time=8, edge=(9, 10)),
        EdgeEvent(time=9, edge=(10, 11)),
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
        # assert batch.to_events() == [e for e in events if e.time // 3 == batch_num]
        batch_num += 1

    assert batch_num == 3 if drop_last else 4


@pytest.mark.parametrize(
    'batch_unit', ['r', 'Y', 'M', 'W', 'D', 'h', 's', 'ms', 'us', 'ns']
)
@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_equal_unit_batch_1(batch_unit, drop_last):
    events = [
        EdgeEvent(time=0, edge=(1, 2)),
        EdgeEvent(time=0, edge=(2, 3)),
        EdgeEvent(time=0, edge=(3, 4)),
        EdgeEvent(time=1, edge=(4, 5)),
        EdgeEvent(time=1, edge=(5, 6)),
        EdgeEvent(time=2, edge=(6, 7)),
        EdgeEvent(time=2, edge=(7, 8)),
        EdgeEvent(time=3, edge=(8, 9)),
        EdgeEvent(time=4, edge=(9, 10)),
    ]
    dg = DGraph(events, time_delta=TimeDeltaDG(batch_unit))
    loader = DGDataLoader(dg, batch_unit=batch_unit, drop_last=drop_last)

    batch_num = 0
    for batch in loader:
        assert isinstance(batch, DGraph)
        assert batch_num < 5
        assert len(batch) == 1
        # assert batch.to_events() == [e for e in events if e.time == batch_num + 1]
        batch_num += 1
    assert batch_num == 5


@pytest.mark.parametrize(
    'batch_unit', ['r', 'Y', 'M', 'W', 'D', 'h', 's', 'ms', 'us', 'ns']
)
@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_equal_unit_batch_larger_than_dg(batch_unit, drop_last):
    events = [
        EdgeEvent(time=0, edge=(1, 2)),
        EdgeEvent(time=0, edge=(2, 3)),
        EdgeEvent(time=0, edge=(3, 4)),
        EdgeEvent(time=1, edge=(4, 5)),
        EdgeEvent(time=1, edge=(5, 6)),
        EdgeEvent(time=2, edge=(6, 7)),
        EdgeEvent(time=2, edge=(7, 8)),
        EdgeEvent(time=3, edge=(8, 9)),
        EdgeEvent(time=4, edge=(9, 10)),
    ]
    dg = DGraph(events, time_delta=TimeDeltaDG(batch_unit))
    loader = DGDataLoader(dg, batch_size=10, batch_unit=batch_unit, drop_last=drop_last)

    batch_num = 0
    for batch in loader:
        assert not drop_last
        assert isinstance(batch, DGraph)
        assert len(batch) == 5
        # assert batch.to_events() == events
        batch_num += 1

    assert batch_num == 0 if drop_last else 1


@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_non_ordered_dg_non_ordered_batch_with_conversion(drop_last):
    events = [
        EdgeEvent(time=0, edge=(1, 2)),  # Batch 1
        EdgeEvent(time=0, edge=(2, 3)),  # Batch 1
        EdgeEvent(time=100, edge=(3, 4)),  # Batch 1
        EdgeEvent(time=100, edge=(4, 5)),  # Batch 1
        EdgeEvent(time=100, edge=(5, 6)),  # Batch 1
        EdgeEvent(time=120, edge=(6, 7)),  # Batch 2
        EdgeEvent(time=180, edge=(7, 8)),  # Batch 2
        EdgeEvent(time=240, edge=(8, 9)),  # Batch 3
        EdgeEvent(time=240, edge=(9, 10)),  # Batch 3
    ]
    dg = DGraph(events, time_delta=TimeDeltaDG(TimeDeltaUnit.SECOND))
    loader = DGDataLoader(
        dg, batch_size=2, batch_unit=TimeDeltaUnit.MINUTE, drop_last=drop_last
    )

    batch_num = 0
    for batch in loader:
        assert isinstance(batch, DGraph)
        if batch_num == 0:
            expected_events = events[:5]
        elif batch_num == 1:
            expected_events = events[5:6]
        elif batch_num == 2:
            expected_events = events[6:]
        else:
            assert False
        # assert batch.to_events() == expected_events
        batch_num += 1

    # assert batch_num == 3


@pytest.mark.skip('Iterating DG that has time delta values > 1 not implemented')
@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_non_ordered_dg_non_ordered_batch_with_conversion_and_time_delta_values_greater_than_1(
    drop_last,
):
    pass


@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_non_ordered_dg_non_ordered_batch_with_conversion_units_mismatch(
    drop_last,
):
    events = [
        EdgeEvent(time=60, edge=(1, 2)),  # Batch 1
        EdgeEvent(time=60, edge=(2, 3)),  # Batch 1
        EdgeEvent(time=100, edge=(3, 4)),  # Batch 1
        EdgeEvent(time=100, edge=(4, 5)),  # Batch 1
        EdgeEvent(time=100, edge=(5, 6)),  # Batch 1
        EdgeEvent(time=120, edge=(6, 7)),  # Batch 2
        EdgeEvent(time=180, edge=(7, 8)),  # Batch 2
        EdgeEvent(time=240, edge=(8, 9)),  # Batch 3
        EdgeEvent(time=240, edge=(9, 10)),  # Batch 3
    ]
    dg = DGraph(events, time_delta=TimeDeltaDG(TimeDeltaUnit.SECOND))
    loader = DGDataLoader(
        dg, batch_size=2, batch_unit=TimeDeltaUnit.MINUTE, drop_last=drop_last
    )

    batch_num = 0
    for batch in loader:
        assert isinstance(batch, DGraph)
        # input()
        # assert len(batch) == 5
        # assert batch.to_events() == events
        batch_num += 1

    # assert batch_num == 3


def test_iteration_non_ordered_dg_non_ordered_batch_unit_too_granular():
    events = [EdgeEvent(time=1, edge=(2, 3))]
    dg = DGraph(events, time_delta=TimeDeltaDG(TimeDeltaUnit.MINUTE))
    with pytest.raises(ValueError):
        # Seconds are too granular of an iteration unit for DG with minute time granularity
        _ = DGDataLoader(dg, batch_unit=TimeDeltaUnit.SECOND)


@pytest.mark.parametrize('batch_unit', ['Y', 'M', 'W', 'D', 'h', 's', 'ms', 'us', 'ns'])
@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_non_ordered_dg_ordered_batch(batch_unit, drop_last):
    events = [
        EdgeEvent(time=0, edge=(1, 2)),
        EdgeEvent(time=0, edge=(2, 3)),
        EdgeEvent(time=0, edge=(3, 4)),
        EdgeEvent(time=1, edge=(4, 5)),
        EdgeEvent(time=1, edge=(5, 6)),
        EdgeEvent(time=2, edge=(6, 7)),
        EdgeEvent(time=2, edge=(7, 8)),
        EdgeEvent(time=3, edge=(8, 9)),
        EdgeEvent(time=4, edge=(9, 10)),
    ]
    dg = DGraph(events, time_delta=TimeDeltaDG(batch_unit))
    loader = DGDataLoader(dg, batch_size=3, drop_last=drop_last)

    batch_num = 0
    for batch in loader:
        assert isinstance(batch, DGraph)
        # assert batch.to_events() == events[3 * batch_num: 3 * (batch_num + 1)]
        batch_num += 1

    assert batch_num == 3


@pytest.mark.parametrize('batch_unit', ['Y', 'M', 'W', 'D', 'h', 's', 'ms', 'us', 'ns'])
@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_non_ordered_dg_ordered_batch_size_1(batch_unit, drop_last):
    events = [
        EdgeEvent(time=0, edge=(1, 2)),
        EdgeEvent(time=0, edge=(2, 3)),
        EdgeEvent(time=0, edge=(3, 4)),
        EdgeEvent(time=1, edge=(4, 5)),
        EdgeEvent(time=1, edge=(5, 6)),
        EdgeEvent(time=2, edge=(6, 7)),
        EdgeEvent(time=2, edge=(7, 8)),
        EdgeEvent(time=3, edge=(8, 9)),
        EdgeEvent(time=4, edge=(9, 10)),
    ]
    dg = DGraph(events, time_delta=TimeDeltaDG(batch_unit))
    loader = DGDataLoader(dg, batch_size=1, drop_last=drop_last)

    batch_num = 0
    for batch in loader:
        assert isinstance(batch, DGraph)
        assert len(batch) == 1
        # assert batch.to_events() == [events[i]]
        batch_num += 1

    assert batch_num == len(events)
