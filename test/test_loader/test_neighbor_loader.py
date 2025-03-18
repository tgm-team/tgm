import pytest

from opendg.events import EdgeEvent
from opendg.graph import DGraph
from opendg.loader import DGNeighborLoader


def test_init_bad_num_nbrs():
    events = [EdgeEvent(time=1, edge=(2, 3))]
    dg = DGraph(events)
    with pytest.raises(ValueError):
        _ = DGNeighborLoader(dg, num_nbrs=[])

    with pytest.raises(ValueError):
        _ = DGNeighborLoader(dg, num_nbrs=[-2, 2])


def test_iteration_with_sampling():
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
    dg = DGraph(events)

    loader = DGNeighborLoader(dg, num_nbrs=[1])
    for batch in loader:
        continue
