import numpy as np
import pytest
import torch

from opendg.events import EdgeEvent
from opendg.nn import EdgeBankPredictor


def test_no_loader_unlimited_memory():
    events = [
        EdgeEvent(t=1, src=2, dst=3, features=torch.rand(2, 5)),
        EdgeEvent(t=5, src=10, dst=20, features=torch.rand(2, 5)),
    ]
    edges = []
    for event in events:
        edges.append([event.src, event.dst, event.t])
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
        EdgeEvent(t=1, src=1, dst=2, features=torch.rand(2, 5)),
        EdgeEvent(t=2, src=2, dst=3, features=torch.rand(2, 5)),
        EdgeEvent(t=3, src=3, dst=4, features=torch.rand(2, 5)),
        EdgeEvent(t=4, src=4, dst=5, features=torch.rand(2, 5)),
        EdgeEvent(t=5, src=5, dst=6, features=torch.rand(2, 5)),
        EdgeEvent(t=6, src=6, dst=7, features=torch.rand(2, 5)),
    ]
    edges = []
    for event in events:
        edges.append([event.src, event.dst, event.t])
    edge_index = np.array(edges)
    srcs = edge_index[:, 0]
    dsts = edge_index[:, 1]
    ts = edge_index[:, 2]

    MEMORY_MODE = 'fixed'
    TIME_WINDOW_RATIO = 0.5

    edgebank = EdgeBankPredictor(
        np.asarray(srcs),
        np.asarray(dsts),
        np.asarray(ts),
        memory_mode=MEMORY_MODE,
        window_ratio=TIME_WINDOW_RATIO,
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


def test_edgebank_arguments():
    events = [
        EdgeEvent(t=1, src=1, dst=2, features=torch.rand(2, 5)),
        EdgeEvent(t=2, src=2, dst=3, features=torch.rand(2, 5)),
        EdgeEvent(t=3, src=3, dst=4, features=torch.rand(2, 5)),
        EdgeEvent(t=4, src=4, dst=5, features=torch.rand(2, 5)),
        EdgeEvent(t=5, src=5, dst=6, features=torch.rand(2, 5)),
        EdgeEvent(t=6, src=6, dst=7, features=torch.rand(2, 5)),
    ]
    edges = []
    for event in events:
        edges.append([event.src, event.dst, event.t])
    edge_index = np.array(edges)
    srcs = edge_index[:, 0]
    dsts = edge_index[:, 1]
    ts = edge_index[:, 2]

    MEMORY_MODE = 'fixed'
    pos_prob = 0.01
    window_ratio = 0.15

    edgebank = EdgeBankPredictor(
        srcs,
        dsts,
        ts,
        memory_mode=MEMORY_MODE,
        window_ratio=window_ratio,
        pos_prob=pos_prob,
    )

    assert edgebank.window_ratio == 0.15
    assert edgebank.window_start == 5.25
    assert edgebank.window_end == 6

    pred = edgebank.predict_link(np.array([6]), np.array([7]))
    assert pred[0] == pos_prob

    edgebank = EdgeBankPredictor(
        srcs,
        dsts,
        ts,
    )
    assert edgebank.window_start == 1
    assert edgebank.window_end == 6

    pred = edgebank.predict_link(np.array([1]), np.array([2]))
    assert pred[0] == 1

    with pytest.raises(ValueError):
        edgebank.update_memory(np.array([]), np.array([]), np.array([1]))

    with pytest.raises(ValueError):
        edgebank.update_memory(np.array([]), np.array([]), np.array([]))

    with pytest.raises(ValueError):
        edgebank = EdgeBankPredictor(
            np.array([]),
            np.array([]),
            np.array([]),
        )

    with pytest.raises(TypeError):
        edgebank = EdgeBankPredictor(
            1,
            2,
            3,
        )
