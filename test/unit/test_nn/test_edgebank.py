import pytest
import torch

from tgm.nn import EdgeBankPredictor


@pytest.mark.parametrize('pos_prob', [0.7, 1.0])
def test_unlimited_memory(pos_prob):
    src = torch.Tensor([2, 10])
    dst = torch.Tensor([3, 20])
    ts = torch.Tensor([1, 5])

    bank = EdgeBankPredictor(src, dst, ts, memory_mode='unlimited', pos_prob=pos_prob)
    assert bank(torch.Tensor([1]), torch.Tensor([1])) == torch.Tensor([0])

    bank.update(torch.Tensor([1]), torch.Tensor([1]), torch.Tensor([7]))
    assert bank(torch.Tensor([1]), torch.Tensor([1])) == torch.Tensor([pos_prob])


@pytest.mark.parametrize('pos_prob', [0.7, 1.0])
def test_fixed_time_window(pos_prob):
    src = torch.Tensor([1, 2, 3, 4, 5, 6])
    dst = torch.Tensor([2, 3, 4, 5, 6, 7])
    ts = torch.Tensor([1, 2, 3, 4, 5, 6])

    WINDOW_RATIO = 0.5

    bank = EdgeBankPredictor(
        src,
        dst,
        ts,
        memory_mode='fixed',
        window_ratio=WINDOW_RATIO,
        pos_prob=pos_prob,
    )
    assert bank.window_start == 3.5
    assert bank.window_end == 6
    assert bank.window_ratio == WINDOW_RATIO

    assert bank(torch.Tensor([4]), torch.Tensor([5])) == torch.Tensor([pos_prob])
    assert bank(torch.Tensor([3]), torch.Tensor([4])) == torch.Tensor([0])

    # update but time window doesn't move forward
    bank.update(torch.Tensor([3]), torch.Tensor([4]), torch.Tensor([5]))
    assert bank(torch.Tensor([3]), torch.Tensor([4])) == torch.Tensor([pos_prob])

    # update and time window moves forward
    bank.update(torch.Tensor([7]), torch.Tensor([8]), torch.Tensor([7]))
    assert bank(torch.Tensor([7]), torch.Tensor([8])) == torch.Tensor([pos_prob])
    assert bank(torch.Tensor([4]), torch.Tensor([5])) == torch.Tensor([0])
    assert (
        not (torch.Tensor([4]), torch.Tensor([5])) in bank.memory
    )  # The edge should be removed from the memory


def test_bad_init_args():
    with pytest.raises(ValueError):
        EdgeBankPredictor(torch.Tensor([]), torch.Tensor([]), torch.Tensor([]))

    with pytest.raises(TypeError):
        EdgeBankPredictor(1, 2, 3)

    src = torch.Tensor([2, 10])
    dst = torch.Tensor([3, 20])
    ts = torch.Tensor([1, 5])
    with pytest.raises(ValueError):
        EdgeBankPredictor(src, dst, ts, memory_mode='foo')

    with pytest.raises(ValueError):
        EdgeBankPredictor(src, dst, ts, window_ratio=0)


def test_bad_update_args():
    src = torch.Tensor([2, 10])
    dst = torch.Tensor([3, 20])
    ts = torch.Tensor([1, 5])
    bank = EdgeBankPredictor(src, dst, ts, memory_mode='unlimited')

    with pytest.raises(ValueError):
        bank.update(torch.Tensor([]), torch.Tensor([]), torch.Tensor([1]))

    with pytest.raises(ValueError):
        bank.update(torch.Tensor([]), torch.Tensor([]), torch.Tensor([]))
