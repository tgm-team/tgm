import pytest

from opendg.events import EdgeEvent
from opendg.graph import DGBatch, DGraph
from opendg.hooks import RecencyNeighborSamplerHook


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
        RecencyNeighborSamplerHook(num_nbrs=[0])
    with pytest.raises(ValueError):
        RecencyNeighborSamplerHook(num_nbrs=[-1])
    with pytest.raises(ValueError):
        RecencyNeighborSamplerHook(num_nbrs=[1, 2])


def test_neighbor_sampler_hook(events):
    dg = DGraph(events)
    hook = RecencyNeighborSamplerHook(num_nbrs=[2])
    batch = hook(dg)
    assert isinstance(batch, DGBatch)
    print(batch.nbrs)
    # TODO: Add logic for testing
