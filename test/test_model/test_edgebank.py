import numpy as np
import torch

from opendg.events import EdgeEvent
from opendg.nn import EdgeBankPredictor


def test_no_loader_unlimited_memory():
    events = [
        EdgeEvent(time=1, edge=(2, 3), features=torch.rand(2, 5)),
        EdgeEvent(time=5, edge=(10, 20), features=torch.rand(2, 5)),
    ]
    edges = []
    for event in events:
        edges.append([event.edge[0], event.edge[1], event.time])
    edge_index = np.array(edges)
    srcs = np.asarray(edge_index[:, 0])
    dsts = np.asarray(edge_index[:, 1])
    ts = np.asarray(edge_index[:, 2])

    MEMORY_MODE = 'unlimited'

    edgebank = EdgeBankPredictor(
        srcs,
        dsts,
        ts,
        memory_mode=MEMORY_MODE,
    )

    pred = edgebank.predict_link(np.asarray([1]), np.asarray([1]))
    assert pred[0] == 0

    edgebank.update_memory(np.asarray([1]), np.asarray([1]), np.asarray([7]))
    pred = edgebank.predict_link(np.asarray([1]), np.asarray([1]))
    assert pred[0] == 1


def test_no_loader_fixed_time_window():
    events = [
        EdgeEvent(time=1, edge=(1, 2), features=torch.rand(2, 5)),
        EdgeEvent(time=2, edge=(2, 3), features=torch.rand(2, 5)),
        EdgeEvent(time=3, edge=(3, 4), features=torch.rand(2, 5)),
        EdgeEvent(time=4, edge=(4, 5), features=torch.rand(2, 5)),
        EdgeEvent(time=5, edge=(5, 6), features=torch.rand(2, 5)),
        EdgeEvent(time=6, edge=(6, 7), features=torch.rand(2, 5)),
    ]
    edges = []
    for event in events:
        edges.append([event.edge[0], event.edge[1], event.time])
    edge_index = np.array(edges)
    srcs = edge_index[:, 0]
    dsts = edge_index[:, 1]
    ts = edge_index[:, 2]

    MEMORY_MODE = 'fixed_time_window'
    TIME_WINDOW_RATIO = 0.5

    edgebank = EdgeBankPredictor(
        np.asarray(srcs),
        np.asarray(dsts),
        np.asarray(ts),
        memory_mode=MEMORY_MODE,
        time_window_ratio=TIME_WINDOW_RATIO,
    )

    pred = edgebank.predict_link(np.array([4]), np.array([5]))
    assert pred[0] == 1
    pred = edgebank.predict_link(np.array([3]), np.array([4]))
    assert pred[0] == 0

    # update but time window doesn't move forward
    edgebank.update_memory(np.array([3]), np.array([4]), np.array([5]))
    pred = edgebank.predict_link(np.array([3]), np.array([4]))
    assert pred[0] == 1

    # update and time window moves forward
    edgebank.update_memory(np.array([7]), np.array([8]), np.array([7]))
    pred = edgebank.predict_link(np.array([7]), np.array([8]))
    assert pred[0] == 1

    pred = edgebank.predict_link(np.array([4]), np.array([5]))
    assert pred[0] == 0
