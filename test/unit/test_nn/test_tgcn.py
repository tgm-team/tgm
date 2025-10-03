import torch

from tgm.nn import TGCN


def test_tgcn():
    B, D, Z = 2, 5, 3
    tgcn = TGCN(in_channels=D, out_channels=Z)
    node_feat = torch.rand((B, D))
    edge_index = torch.randint(0, B, size=(B, 2))
    h = tgcn(node_feat, edge_index)
    assert h.shape == (B, Z)
    assert not torch.isnan(h).any()

    h = tgcn(node_feat, edge_index, H=h)
    assert h.shape == (B, Z)
    assert not torch.isnan(h).any()
