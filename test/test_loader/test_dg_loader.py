import pytest

from opendg.events import EdgeEvent
from opendg.graph import DGraph
from opendg.loader import DGDataLoader
from opendg.timedelta import TimeDeltaDG, TimeDeltaUnit


def test_init_default_args():
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


@pytest.mark.skip(reason='Not implemented')
@pytest.mark.parametrize(
    'batch_unit', ['r', 'Y', 'M', 'W', 'D', 'h', 's', 'ms', 'us', 'ns']
)
@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_equal_unit(batch_unit, drop_last):
    events = [
        EdgeEvent(time=1, edge=(1, 2)),
        EdgeEvent(time=1, edge=(2, 3)),
        EdgeEvent(time=1, edge=(3, 4)),
        EdgeEvent(time=2, edge=(4, 5)),
        EdgeEvent(time=2, edge=(5, 6)),
        EdgeEvent(time=3, edge=(6, 7)),
        EdgeEvent(time=3, edge=(7, 8)),
        EdgeEvent(time=4, edge=(8, 9)),
        EdgeEvent(time=5, edge=(9, 10)),
    ]
    dg = DGraph(events, time_delta=TimeDeltaDG(batch_unit))
    loader = DGDataLoader(dg, batch_size=3, batch_unit=batch_unit, drop_last=drop_last)
    for i, batch in enumerate(loader):
        assert isinstance(batch, DGraph)
        if i == 4:
            if drop_last:
                assert False
            else:
                assert len(batch) == 1
                assert batch.to_events() == events[-1]

        assert len(batch) == 3
        assert batch.to_events() == events[3 * i : 3 * (i + 1)]


@pytest.mark.skip(reason='Not implemented')
@pytest.mark.parametrize(
    'batch_unit', ['r', 'Y', 'M', 'W', 'D', 'h', 's', 'ms', 'us', 'ns']
)
@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_equal_unit_batch_1(batch_unit, drop_last):
    events = [
        EdgeEvent(time=1, edge=(1, 2)),
        EdgeEvent(time=1, edge=(2, 3)),
        EdgeEvent(time=1, edge=(3, 4)),
        EdgeEvent(time=2, edge=(4, 5)),
        EdgeEvent(time=2, edge=(5, 6)),
        EdgeEvent(time=3, edge=(6, 7)),
        EdgeEvent(time=3, edge=(7, 8)),
        EdgeEvent(time=4, edge=(8, 9)),
        EdgeEvent(time=5, edge=(9, 10)),
    ]
    dg = DGraph(events, time_delta=TimeDeltaDG(batch_unit))
    loader = DGDataLoader(dg, batch_size=1, batch_unit=batch_unit, drop_last=drop_last)

    loader = DGDataLoader(dg, drop_last=drop_last)
    for i, batch in enumerate(loader):
        assert isinstance(batch, DGraph)
        assert len(batch) == 1
        assert batch.to_events() == events[i]


@pytest.mark.skip(reason='Not implemented')
@pytest.mark.parametrize(
    'batch_unit', ['r', 'Y', 'M', 'W', 'D', 'h', 's', 'ms', 'us', 'ns']
)
@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_equal_unit_batch_larger_than_dg(batch_unit, drop_last):
    events = [
        EdgeEvent(time=1, edge=(1, 2)),
        EdgeEvent(time=1, edge=(2, 3)),
        EdgeEvent(time=1, edge=(3, 4)),
        EdgeEvent(time=2, edge=(4, 5)),
        EdgeEvent(time=2, edge=(5, 6)),
        EdgeEvent(time=3, edge=(6, 7)),
        EdgeEvent(time=3, edge=(7, 8)),
        EdgeEvent(time=4, edge=(8, 9)),
        EdgeEvent(time=5, edge=(9, 10)),
    ]
    dg = DGraph(events, time_delta=TimeDeltaDG(batch_unit))
    loader = DGDataLoader(dg, batch_size=10, batch_unit=batch_unit, drop_last=drop_last)
    for batch in loader:
        if drop_last:
            assert False
        assert isinstance(batch, DGraph)
        assert len(batch) == len(events)
        assert batch.to_events() == events


@pytest.mark.skip(reason='Not implemented')
@pytest.mark.parametrize('batch_size', [1, 3, 10])
@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_non_ordered_dg_non_ordered_batch_with_conversion_up(
    batch_size, drop_last
):
    pass


@pytest.mark.skip(reason='Not implemented')
@pytest.mark.parametrize('batch_size', [1, 3, 10])
@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_non_ordered_dg_non_ordered_batch_with_conversion_down(
    batch_size, drop_last
):
    pass


@pytest.mark.skip(reason='not implemented')
@pytest.mark.parametrize('batch_size', [1, 3, 10])
@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_non_ordered_dg_ordered_batch(batch_size, drop_last):
    pass


@pytest.mark.skip(reason='not implemented')
@pytest.mark.parametrize('batch_size', [1, 3, 10])
@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_non_ordered_dg_ordered_batch_batch_split_required(
    batch_size, drop_last
):
    pass


@pytest.mark.skip(reason='not implemented')
@pytest.mark.parametrize('batch_size', [1, 3, 10])
@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_non_ordered_dg_ordered_batch_multi_batch_required(
    batch_size, drop_last
):
    pass
