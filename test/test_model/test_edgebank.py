import pytest
import torch

from opendg.events import EdgeEvent
from opendg.nn import EdgeBankPredictor


@pytest.mark.parametrize('pos_prob', [0.7, 1.0])
def test_unlimited_memory(pos_prob):
    events = [EdgeEvent(t=1, src=2, dst=3), EdgeEvent(t=5, src=10, dst=20)]
    src, dst, ts = _events_to_edge_arrays(events)

    bank = EdgeBankPredictor(src, dst, ts, memory_mode='unlimited', pos_prob=pos_prob)
    assert bank(torch.Tensor([1]), torch.Tensor([1])) == torch.Tensor([0])

    bank.update(torch.Tensor([1]), torch.Tensor([1]), torch.Tensor([7]))
    assert bank(torch.Tensor([1]), torch.Tensor([1])) == torch.Tensor([pos_prob])


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
        torch.Tensor(src),
        torch.Tensor(dst),
        torch.Tensor(ts),
        memory_mode=MEMORY_MODE,
        window_ratio=TIME_WINDOW_RATIO,
        pos_prob=pos_prob,
    )

    assert bank(torch.Tensor([4]), torch.Tensor([5])) == torch.Tensor([pos_prob])
    assert bank(torch.Tensor([3]), torch.Tensor([4])) == torch.Tensor([0])

    # update but time window doesn't move forward
    bank.update(torch.Tensor([3]), torch.Tensor([4]), torch.Tensor([5]))
    assert bank(torch.Tensor([3]), torch.Tensor([4])) == torch.Tensor([pos_prob])

    # update and time window moves forward
    bank.update(torch.Tensor([7]), torch.Tensor([8]), torch.Tensor([7]))
    assert bank(torch.Tensor([7]), torch.Tensor([8])) == torch.Tensor([pos_prob])
    assert bank(torch.Tensor([4]), torch.Tensor([5])) == torch.Tensor([0])


def test_bad_init_args():
    with pytest.raises(ValueError):
        EdgeBankPredictor(torch.Tensor([]), torch.Tensor([]), torch.Tensor([]))

    with pytest.raises(TypeError):
        EdgeBankPredictor(1, 2, 3)


def test_bad_update_args():
    events = [EdgeEvent(t=1, src=2, dst=3), EdgeEvent(t=5, src=10, dst=20)]
    src, dst, ts = _events_to_edge_arrays(events)
    bank = EdgeBankPredictor(src, dst, ts, memory_mode='unlimited')

    with pytest.raises(ValueError):
        bank.update(torch.Tensor([]), torch.Tensor([]), torch.Tensor([1]))

    with pytest.raises(ValueError):
        bank.update(torch.Tensor([]), torch.Tensor([]), torch.Tensor([]))


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
    assert bank(torch.Tensor([6]), torch.Tensor([7])) == torch.Tensor([pos_prob])


def _events_to_edge_arrays(events):
    edge_list = []
    for event in events:
        edge_list.append([event.src, event.dst, event.t])
    edges = torch.Tensor(edge_list)
    return (
        torch.Tensor(edges[:, 0]),
        torch.Tensor(edges[:, 1]),
        torch.Tensor(edges[:, 2]),
    )
