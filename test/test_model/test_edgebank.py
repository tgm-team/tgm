import numpy as np
import pytest

from opendg.events import EdgeEvent
from opendg.nn import EdgeBankPredictor


@pytest.mark.parametrize('pos_prob', [0.7, 1.0])
def test_unlimited_memory(pos_prob):
    events = [EdgeEvent(t=1, src=2, dst=3), EdgeEvent(t=5, src=10, dst=20)]
    src, dst, ts = _events_to_edge_arrays(events)

    bank = EdgeBankPredictor(src, dst, ts, memory_mode='unlimited', pos_prob=pos_prob)
    assert bank.predict_link(np.asarray([1]), np.asarray([1])) == np.array([0])

    bank.update_memory(np.asarray([1]), np.asarray([1]), np.asarray([7]))
    assert bank.predict_link(np.asarray([1]), np.asarray([1])) == np.array([pos_prob])


@pytest.mark.parametrize('pos_prob', [0.7, 1.0])
def test_fixed_time_window(pos_prob):
    events = [
        EdgeEvent(t=1, src=1, dst=2),
        EdgeEvent(t=2, src=2, dst=3),
        EdgeEvent(t=3, src=3, dst=4),
        EdgeEvent(t=4, src=4, dst=5),
        EdgeEvent(t=5, src=5, dst=6),
        EdgeEvent(t=6, src=6, dst=7),
    ]
    src, dst, ts = _events_to_edge_arrays(events)

    MEMORY_MODE = 'fixed'
    TIME_WINDOW_RATIO = 0.5

    bank = EdgeBankPredictor(
        np.asarray(src),
        np.asarray(dst),
        np.asarray(ts),
        memory_mode=MEMORY_MODE,
        window_ratio=TIME_WINDOW_RATIO,
        pos_prob=pos_prob,
    )

    assert bank.predict_link(np.array([4]), np.array([5])) == np.array([pos_prob])
    assert bank.predict_link(np.array([3]), np.array([4])) == np.array([0])

    # update but time window doesn't move forward
    bank.update_memory(np.array([3]), np.array([4]), np.array([5]))
    assert bank.predict_link(np.array([3]), np.array([4])) == np.array([pos_prob])

    # update and time window moves forward
    bank.update_memory(np.array([7]), np.array([8]), np.array([7]))
    assert bank.predict_link(np.array([7]), np.array([8])) == np.array([pos_prob])
    assert bank.predict_link(np.array([4]), np.array([5])) == np.array([0])


def test_bad_init_args():
    with pytest.raises(ValueError):
        EdgeBankPredictor(np.array([]), np.array([]), np.array([]))

    with pytest.raises(TypeError):
        EdgeBankPredictor(1, 2, 3)


def test_bad_update_args():
    events = [EdgeEvent(t=1, src=2, dst=3), EdgeEvent(t=5, src=10, dst=20)]
    src, dst, ts = _events_to_edge_arrays(events)
    bank = EdgeBankPredictor(src, dst, ts, memory_mode='unlimited')

    with pytest.raises(ValueError):
        bank.update_memory(np.array([]), np.array([]), np.array([1]))

    with pytest.raises(ValueError):
        bank.update_memory(np.array([]), np.array([]), np.array([]))


@pytest.mark.parametrize('pos_prob', [0.7, 1.0])
def test_edgebank_arguments(pos_prob):
    events = [
        EdgeEvent(t=1, src=1, dst=2),
        EdgeEvent(t=2, src=2, dst=3),
        EdgeEvent(t=3, src=3, dst=4),
        EdgeEvent(t=4, src=4, dst=5),
        EdgeEvent(t=5, src=5, dst=6),
        EdgeEvent(t=6, src=6, dst=7),
    ]
    src, dst, ts = _events_to_edge_arrays(events)

    WINDOW_RATIO = 0.15

    bank = EdgeBankPredictor(
        src,
        dst,
        ts,
        memory_mode='fixed',
        window_ratio=WINDOW_RATIO,
        pos_prob=pos_prob,
    )
    assert bank.window_start == 5.25
    assert bank.window_end == 6
    assert bank.window_ratio == WINDOW_RATIO
    assert bank.predict_link(np.array([6]), np.array([7])) == np.array([pos_prob])


def _events_to_edge_arrays(events):
    edge_list = []
    for event in events:
        edge_list.append([event.src, event.dst, event.t])
    edges = np.array(edge_list)
    return np.asarray(edges[:, 0]), np.asarray(edges[:, 1]), np.asarray(edges[:, 2])
