import pytest

from opendg.events import EdgeEvent, NodeEvent
from opendg.graph import DGBatch, DGraph
from opendg.hooks import LastNeighborHook


@pytest.fixture
def events():
    return [
        NodeEvent(t=1, src=2),
        EdgeEvent(t=1, src=2, dst=2),
        NodeEvent(t=5, src=4),
        EdgeEvent(t=5, src=2, dst=4),
        NodeEvent(t=10, src=6),
        EdgeEvent(t=20, src=1, dst=8),
    ]


def test_bad_neighbor_sampler_init():
    with pytest.raises(ValueError):
        LastNeighborHook(num_nodes=0, size=10)
    with pytest.raises(ValueError):
        LastNeighborHook(num_nodes=10, size=0)


def test_neighbor_sampler_hook(events):
    dg = DGraph(events)
    hook = LastNeighborHook(num_nodes=dg.num_nodes, size=3)
    batch = hook(dg)
    assert isinstance(batch, DGBatch)

    # TODO: Add logic for testing
