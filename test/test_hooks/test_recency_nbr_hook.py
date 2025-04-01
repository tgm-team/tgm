import pytest

from opendg.events import EdgeEvent
from opendg.graph import DGBatch, DGraph
from opendg.hooks import RecencyNeighborSamplerHook


@pytest.fixture
def events():
    return [
        EdgeEvent(t=1, src=2, dst=2),
        EdgeEvent(t=5, src=2, dst=4),
        EdgeEvent(t=20, src=1, dst=8),
    ]


def test_bad_neighbor_sampler_init():
    with pytest.raises(ValueError):
        RecencyNeighborSamplerHook(num_nbrs=[0])
    with pytest.raises(ValueError):
        RecencyNeighborSamplerHook(num_nbrs=[-1])
    with pytest.raises(ValueError):
        RecencyNeighborSamplerHook(num_nbrs=[1, 2])


def test_neighbor_sampler_hook(events):
    dg = DGraph(events)
    hook = RecencyNeighborSamplerHook(num_nodes=dg.num_nodes, size=3)
    batch = hook(dg)
    assert isinstance(batch, DGBatch)

    # TODO: Add logic for testing
