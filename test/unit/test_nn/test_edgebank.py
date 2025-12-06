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


def test_complete_eviction_fixed_time_window():
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
    )
    assert bank.window_start == 3.5
    assert bank.window_end == 6
    assert bank.window_ratio == WINDOW_RATIO

    # Update with edge in the very far future. Evict all existing interactions
    bank.update(torch.Tensor([7]), torch.Tensor([8]), torch.Tensor([100000000]))
    for s, d in zip(src, dst):
        assert not (s.item(), d.item()) in bank.memory

    assert (7, 8) in bank.memory


def test_out_of_order_construct_fixed():
    src = torch.Tensor([1, 2, 3, 4])
    dst = torch.Tensor([2, 3, 4, 5])
    ts = torch.Tensor([1, 4, 2, 3])

    WINDOW_RATIO = 0.5

    bank = EdgeBankPredictor(
        src,
        dst,
        ts,
        memory_mode='fixed',
        window_ratio=WINDOW_RATIO,
    )
    assert bank.window_start == 2.5
    assert bank.window_end == 4
    assert bank.window_ratio == WINDOW_RATIO

    expected_edge_order = [(4.0, 5.0), (2.0, 3.0)]
    expected_ts = range(3, 5)
    assert bank._head is not None and bank._tail is not None
    curr = bank._head
    count = 0
    while curr is not None:
        assert curr.edge == expected_edge_order[count]
        assert curr.ts == expected_ts[count]
        count += 1
        curr = curr.right

    assert count == 2


def test_out_of_order_update_fixed():
    src = torch.Tensor([1, 2, 3, 4])
    dst = torch.Tensor([2, 3, 4, 5])
    ts = torch.Tensor([1, 4, 2, 3])

    WINDOW_RATIO = 0.5

    bank = EdgeBankPredictor(
        src,
        dst,
        ts,
        memory_mode='fixed',
        window_ratio=WINDOW_RATIO,
    )
    bank.update(torch.Tensor([1]), torch.Tensor([1]), torch.Tensor([3]))
    assert bank.window_start == 2.5
    assert bank.window_end == 4
    assert bank.window_ratio == WINDOW_RATIO

    expected_edge_order = [(4.0, 5.0), (1.0, 1.0), (2.0, 3.0)]
    expected_ts = [3, 3, 4]
    assert bank._head is not None and bank._tail is not None
    curr = bank._head
    count = 0
    while curr is not None:
        assert curr.edge == expected_edge_order[count]
        assert curr.ts == expected_ts[count]
        count += 1
        curr = curr.right
    assert count == 3

    curr = bank._tail
    count = 3
    while curr is not None:
        count -= 1
        assert curr.edge == expected_edge_order[count]
        assert curr.ts == expected_ts[count]
        curr = curr.left
    assert count == 0


def test_out_of_order_construct_unlimited():
    src = torch.Tensor([1, 2, 3])
    dst = torch.Tensor([2, 3, 4])
    ts = torch.Tensor([3, 2, 1])
    bank = EdgeBankPredictor(
        src,
        dst,
        ts,
        memory_mode='unlimited',
    )
    assert bank.window_start == 1
    assert bank.window_end == 3

    expected_edge_order = [
        (3.0, 4.0),
        (2.0, 3.0),
        (1.0, 2.0),
    ]
    expected_ts = range(1, 4)
    assert bank._head is not None and bank._tail is not None
    curr = bank._head
    count = 0
    while curr is not None:
        assert curr.edge == expected_edge_order[count]
        assert curr.ts == expected_ts[count]
        count += 1
        curr = curr.right

    assert count == 3


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
