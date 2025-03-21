import pytest

from opendg.events import EdgeEvent
from opendg.graph import DGraph
from opendg.loader import DGNeighborLoader


def test_init_bad_num_nbrs():
    events = [EdgeEvent(t=1, src=2, dst=3)]
    dg = DGraph(events)
    with pytest.raises(ValueError):
        _ = DGNeighborLoader(dg, num_nbrs=[])
    with pytest.raises(ValueError):
        _ = DGNeighborLoader(dg, num_nbrs=[-2, 2])


def test_iteration_with_sampling():
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
    dg = DGraph(events)
    loader = DGNeighborLoader(dg, num_nbrs=[1])
    for batch in loader:
        continue
