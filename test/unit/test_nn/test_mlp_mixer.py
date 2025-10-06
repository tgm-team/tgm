import torch

from tgm.nn import MLPMixer


def test_mlp_mixer():
    B, D, Z = 2, 5, 3
    mixer = MLPMixer(num_tokens=D, num_channels=Z)
    x = torch.rand((B, D, Z))
    out = mixer(x)
    assert out.shape == (B, D, Z)
    assert not torch.isnan(out).any()
