import pytest
import torch

from tgm.nn import PopTrackPredictor

@pytest.mark.parametrize('decay', [0.7, 1.0])
def test_poptrack_update(decay):
    src = torch.tensor([0, 1])
    dst = torch.tensor([2, 3])
    ts = torch.tensor([1, 2])

    model = PopTrackPredictor(src, dst, ts, num_nodes=4, k=2, decay=decay)

    assert torch.allclose(model(torch.tensor([1]), torch.tensor([1])), torch.tensor([0.0]))

    model.update(
        torch.tensor([1]),
        torch.tensor([1]),
        torch.tensor([7.0]),
        decay=decay,
    )

    assert torch.allclose(model(torch.tensor([1]), torch.tensor([1])), torch.tensor([decay]))


def test_init_valid_input():
    src = torch.tensor([0, 1])
    dst = torch.tensor([2, 3])
    ts = torch.tensor([1, 2])

    model = PopTrackPredictor(src, dst, ts, num_nodes=4, k=2)

    assert isinstance(model.popularity, torch.Tensor)
    assert model.popularity.shape == (4,)
    assert model.k == 2
    assert len(model.top_k) == 2
    

def test_bad_init_args():
    with pytest.raises(ValueError):
        PopTrackPredictor(torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), num_nodes=2)

    with pytest.raises(TypeError):
        PopTrackPredictor(1, 2, 3, num_nodes=2)

    src = torch.tensor([0, 1])
    dst = torch.tensor([2, 3])
    ts = torch.tensor([1, 2])
    with pytest.raises(ValueError):
        PopTrackPredictor(src, dst, ts, num_nodes=4, k=-5)

    with pytest.raises(ValueError):
        PopTrackPredictor(src, dst, ts, num_nodes=4, decay=-0.5)

    with pytest.raises(ValueError):
        PopTrackPredictor(src, dst, ts, num_nodes=4, decay=2)
    
    with pytest.raises(ValueError):
        PopTrackPredictor(src, dst, ts, num_nodes=0)


def test_bad_update_args():
    src = torch.tensor([0, 1])
    dst = torch.tensor([2, 3])
    ts = torch.tensor([1, 2])
    model = PopTrackPredictor(src, dst, ts, num_nodes=4)

    with pytest.raises(ValueError):
        model.update(torch.Tensor([]), torch.Tensor([]), torch.Tensor([1]))

    with pytest.raises(ValueError):
        model.update(torch.Tensor([]), torch.Tensor([]), torch.Tensor([]))
