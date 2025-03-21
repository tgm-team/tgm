import pytest
import torch

from opendg.nn import TimeEncoder


@pytest.mark.parametrize('time_dim', [4, 32])
@pytest.mark.parametrize('requires_grad', [True, False])
def test_time_encoder_init(time_dim, requires_grad):
    enc = TimeEncoder(time_dim=time_dim, requires_grad=requires_grad)
    assert enc.w.weight.shape == (time_dim, 1)
    assert enc.w.bias.shape == (time_dim,)
    assert enc.w.weight.requires_grad == requires_grad
    assert enc.w.bias.requires_grad == requires_grad


@pytest.mark.parametrize('time_dim', [4, 32])
def test_time_encoder_init(time_dim):
    enc = TimeEncoder(time_dim=time_dim)

    batch_size, seq_len = 128, 64
    x = torch.rand(batch_size, seq_len)
    y = enc(x)
    assert torch.is_tensor(y)
    assert y.shape == (batch_size, seq_len, time_dim)
