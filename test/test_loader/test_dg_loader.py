import torch

from opendg.events import EdgeEvent
from opendg.graph import DGraph
from opendg.loader import DGDataLoader


def test_foo():
    events = [
        EdgeEvent(time=1, edge=(2, 3), features=torch.rand(1, 3, 37)),
        EdgeEvent(time=5, edge=(10, 20), features=torch.rand(1, 3, 37)),
    ]
    dg = DGraph(events)

    loader = DGDataLoader(dg, batch_size=4)
    for a in loader:
        print(a)
