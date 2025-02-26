import numpy as np
import pytest
import torch

from opendg._storage import DGStorageBackends
from opendg.events import EdgeEvent, NodeEvent
from opendg.nn import EdgeBankPredictor
from opendg.timedelta import TimeDeltaDG


@pytest.fixture(params=DGStorageBackends.values())
def DGStorageImpl(request):
    return request.param


def test_no_loader(DGStorageImpl):
    events = [
        EdgeEvent(time=1, edge=(2, 3), features=torch.rand(2, 5)),
        EdgeEvent(time=5, edge=(10, 20), features=torch.rand(2, 5)),
        NodeEvent(time=6, node_id=7, features=torch.rand(3, 6)),
    ]
    td = TimeDeltaDG('Y', 1)
    storage = DGStorageImpl(events, td)
    events = storage.to_events()
    edges = []
    for event in events:
        if isinstance(event, EdgeEvent):
            edges.append([event.edge[0], event.edge[1], event.time])
    edge_index = np.array(edges)
    srcs = edge_index[:, 0]
    dsts = edge_index[:, 1]
    ts = edge_index[:, 2]

    MEMORY_MODE = 'unlimited'
    TIME_WINDOW_RATIO = 0.15

    edgebank = EdgeBankPredictor(
        np.asarray(srcs),
        np.asarray(dsts),
        np.asarray(ts),
        memory_mode=MEMORY_MODE,
        time_window_ratio=TIME_WINDOW_RATIO,
    )

    pred = edgebank.predict_link(np.asarray([1]), np.asarray([1]))
    assert pred[0] == 0

    edgebank.update_memory(np.asarray([1]), np.asarray([1]), np.asarray([7]))
    pred = edgebank.predict_link(np.asarray([1]), np.asarray([1]))
    assert pred[0] == 1
