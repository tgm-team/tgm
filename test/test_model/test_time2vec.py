import pytest
import torch

from tgm.nn import Time2Vec


@pytest.mark.parametrize('time_dim', [4, 32])
def test_time_encoder_init(time_dim):
    enc = Time2Vec(time_dim=time_dim)
    assert enc.w.weight.shape == (time_dim, 1)
    assert enc.w.bias.shape == (time_dim,)


@pytest.mark.parametrize('time_dim', [4, 32])
def test_time_encoder_forward(time_dim):
    enc = Time2Vec(time_dim=time_dim)

    batch_size, seq_len = 128, 64
    x = torch.rand(batch_size, seq_len)
    y = enc(x)
    assert torch.is_tensor(y)
    assert y.shape == (batch_size, seq_len, time_dim)
