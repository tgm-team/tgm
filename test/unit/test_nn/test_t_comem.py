import pytest
import torch

from tgm.nn import tCoMemPredictor


@pytest.mark.parametrize('window_ratio', [0.2, 0.5])
def test_init_valid_input(window_ratio):
    src = torch.Tensor([1, 2, 3, 4])
    dst = torch.Tensor([2, 3, 4, 5])
    ts = torch.Tensor([1, 2, 3, 4])

    model = tCoMemPredictor(src, dst, ts, num_nodes=10, k=5, window_ratio=window_ratio)

    assert model.window_end == 4
    assert model.window_start == model.window_end - model.window_size == 1
    assert model.window_ratio == window_ratio
    assert model.window_size == 3


def test_update_moves_window_forward():
    src = torch.Tensor([1, 2, 3, 4, 5, 6])
    dst = torch.Tensor([2, 3, 4, 5, 6, 7])
    ts = torch.Tensor([1, 2, 3, 4, 5, 6])

    WINDOW_RATIO = 0.5

    model = tCoMemPredictor(
        src,
        dst,
        ts,
        num_nodes=10,
        k=5,
        window_ratio=WINDOW_RATIO,
    )

    assert model.window_end == 6
    assert model.window_start == model.window_end - model.window_size == 1
    assert model.window_ratio == WINDOW_RATIO

    # recent interaction should have nonzero score
    assert model(torch.Tensor([4]), torch.Tensor([5])) > 0

    # update but time window doesn't move forward
    model.update(torch.Tensor([3]), torch.Tensor([4]), torch.Tensor([5]))
    assert model(torch.Tensor([3]), torch.Tensor([4])) > 0

    # update and time window moves forward
    model.update(torch.Tensor([7]), torch.Tensor([8]), torch.Tensor([7]))
    assert model(torch.Tensor([7]), torch.Tensor([8])) > 0


def test_out_of_order_construct():
    src = torch.Tensor([1, 2, 3])
    dst = torch.Tensor([2, 3, 4])
    ts = torch.Tensor([3, 1, 2])

    model = tCoMemPredictor(src, dst, ts, num_nodes=10, k=5)

    assert model.window_end == 3
    assert model.window_start == model.window_end - model.window_size == 1


def test_recent_buffers_update_correctly():
    src = torch.Tensor([1, 1, 1])
    dst = torch.Tensor([2, 3, 4])
    ts = torch.Tensor([1, 2, 3])

    model = tCoMemPredictor(src, dst, ts, num_nodes=10, k=2)

    assert model.recent_len[1] == 2
    assert model.recent_dst[1].tolist()[:2] == [4, 3]


def test_co_occurrence_accumulates():
    src = torch.Tensor([1, 1, 1])
    dst = torch.Tensor([2, 2, 2])
    ts = torch.Tensor([1, 2, 3])

    model = tCoMemPredictor(src, dst, ts, num_nodes=10, k=5)

    assert model.node_to_co_occurrence[1][2] == 3
    assert model.node_to_co_occurrence[2][1] == 3


def test_popularity_increases():
    src = torch.Tensor([1, 2, 3])
    dst = torch.Tensor([4, 4, 4])
    ts = torch.Tensor([1, 2, 3])

    model = tCoMemPredictor(src, dst, ts, num_nodes=10, k=5)

    assert model.popularity[4].item() == 3


def test_no_history():
    src = torch.Tensor([1])
    dst = torch.Tensor([2])
    ts = torch.Tensor([1])

    model = tCoMemPredictor(src, dst, ts, num_nodes=10, k=5)

    pred = model(torch.Tensor([9]), torch.Tensor([1]))
    assert pred.item() == 0.0


def test_pred_seen_neighbor():
    src = torch.Tensor([1, 1])
    dst = torch.Tensor([2, 3])
    ts = torch.Tensor([4, 5])

    model = tCoMemPredictor(src, dst, ts, num_nodes=10, k=5)

    pred = model(torch.Tensor([1]), torch.Tensor([2]))
    assert pred.item() > 0.0


def test_co_occurrence_increases_score():
    src = torch.Tensor([1, 1])
    dst = torch.Tensor([2, 2])
    ts = torch.Tensor([1, 2])

    model = tCoMemPredictor(src, dst, ts, num_nodes=10, k=5, co_occurrence_weight=1.0)

    pred1 = model(torch.Tensor([1]), torch.Tensor([2]))

    model.update(torch.Tensor([1]), torch.Tensor([2]), torch.Tensor([3]))
    pred2 = model(torch.Tensor([1]), torch.Tensor([2]))

    assert pred2 > pred1


def test_bad_init_args():
    with pytest.raises(ValueError):
        tCoMemPredictor(
            torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), num_nodes=10
        )

    with pytest.raises(TypeError):
        tCoMemPredictor('1', '2', '3', num_nodes=10, k=5)

    empty_src = torch.Tensor([])
    shorter_src = torch.Tensor([1])
    src = torch.Tensor([1, 1])
    dst = torch.Tensor([2, 2])
    ts = torch.Tensor([1, 2])

    with pytest.raises(ValueError):
        tCoMemPredictor(empty_src, dst, ts, num_nodes=10, k=5)

    with pytest.raises(ValueError):
        tCoMemPredictor(shorter_src, dst, ts, num_nodes=10, k=5)

    with pytest.raises(ValueError):
        tCoMemPredictor(src, dst, ts, num_nodes=10, k=5, co_occurrence_weight=1.5)

    with pytest.raises(ValueError):
        tCoMemPredictor(src, dst, ts, num_nodes=10, k=5, window_ratio=0)

    with pytest.raises(ValueError):
        tCoMemPredictor(src, dst, ts, num_nodes=10, k=5, window_ratio=1.1)

    with pytest.raises(ValueError):
        tCoMemPredictor(src, dst, ts, num_nodes=10, k=-5)

    with pytest.raises(ValueError):
        tCoMemPredictor(src, dst, ts, num_nodes=-10, k=5)

    with pytest.raises(ValueError):
        tCoMemPredictor(src, dst, ts, num_nodes=10, k=11)


def test_bad_update_args():
    src = torch.Tensor([1])
    dst = torch.Tensor([2])
    ts = torch.Tensor([1])

    model = tCoMemPredictor(src, dst, ts, num_nodes=10, k=5)

    with pytest.raises(ValueError):
        model.update(torch.Tensor([]), torch.Tensor([]), torch.Tensor([]))

    with pytest.raises(TypeError):
        model.update(1, 2, 3)
