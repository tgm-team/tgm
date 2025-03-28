import pytest

from opendg.events import EdgeEvent
from opendg.graph import DGraph
from opendg.loader import DGNeighborLoader
from opendg.timedelta import TimeDeltaDG


def test_init_bad_num_nbrs():
    events = [EdgeEvent(t=1, src=2, dst=3)]
    dg = DGraph(events)
    with pytest.raises(ValueError):
        _ = DGNeighborLoader(dg, num_nbrs=[])
    with pytest.raises(ValueError):
        _ = DGNeighborLoader(dg, num_nbrs=[-2, 2])


@pytest.mark.parametrize('num_nbrs', [[1], [-1]])
def test_init(num_nbrs):
    events = [EdgeEvent(t=1, src=2, dst=3)]
    dg = DGraph(events)
    loader = DGNeighborLoader(dg, num_nbrs=num_nbrs)
    assert loader.num_hops == len(num_nbrs)
    assert loader.num_nbrs == num_nbrs


@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_with_sampling(drop_last):
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
    loader = DGNeighborLoader(
        dg,
        num_nbrs=[1],
        batch_size=2,
        batch_unit='m',
        drop_last=drop_last,
        materialize=False,
    )

    batch_num = 0
    for batch in loader:
        assert isinstance(batch, DGraph)
        if batch_num == 0:
            assert batch.to_events() == events[:2]
        elif batch_num == 1:
            assert batch.to_events() == events[3:6]
        elif batch_num == 2:
            assert batch.to_events() == events[6:]
        else:
            assert False
        batch_num += 1
    assert batch_num == 2 if drop_last else 3


@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_with_full_neighborhood_sampling(drop_last):
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
    loader = DGNeighborLoader(
        dg,
        num_nbrs=[-1],
        batch_size=2,
        batch_unit='m',
        drop_last=drop_last,
        materialize=False,
    )

    batch_num = 0
    for batch in loader:
        assert isinstance(batch, DGraph)
        if batch_num == 0:
            assert batch.to_events() == events[:2]
        elif batch_num == 1:
            assert batch.to_events() == events[3:6]
        elif batch_num == 2:
            assert batch.to_events() == events[5:]
        else:
            assert False
        batch_num += 1
    assert batch_num == 2 if drop_last else 3


@pytest.mark.skip('Multi-hop not supported')
def test_iteration_multi_hop():
    pass
