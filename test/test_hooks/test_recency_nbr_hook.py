import pytest

from opendg.events import EdgeEvent
from opendg.graph import DGBatch, DGraph
from opendg.hooks import RecencyNeighborHook


@pytest.fixture
def events():
    return [
        EdgeEvent(t=1, src=1, dst=10),
        EdgeEvent(t=1, src=1, dst=11),
        EdgeEvent(t=2, src=1, dst=12),
        EdgeEvent(t=2, src=1, dst=13),
    ]


def test_bad_neighbor_sampler_init():
    with pytest.raises(ValueError):
        RecencyNeighborHook(num_nbrs=[0], num_nodes=2)
    with pytest.raises(ValueError):
        RecencyNeighborHook(num_nbrs=[-1], num_nodes=2)
    with pytest.raises(ValueError):
        RecencyNeighborHook(num_nbrs=[1, 2], num_nodes=2)


@pytest.mark.skip('TODO: Add neighbor sampling tests')
def test_neighbor_sampler_hook(events):
    dg = DGraph(events)
    hook = RecencyNeighborHook(num_nbrs=[2], num_nodes=dg.num_nodes)
    batch = hook(dg)
    # TODO: Add logic for testing
    assert isinstance(batch, DGBatch)
