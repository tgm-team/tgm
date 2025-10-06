import torch

from tgm.nn import GCLSTM


def test_gclstm():
    B, D, Z = 2, 5, 3
    gclstm = GCLSTM(in_channels=D, out_channels=Z, K=1)
    node_feat = torch.rand((B, D))
    edge_index = torch.randint(0, B, size=(B, 2))
    h, c = gclstm(node_feat, edge_index)
    assert h.shape == (B, Z)
    assert c.shape == (B, Z)
    assert not torch.isnan(h).any()
    assert not torch.isnan(c).any()

    h, c = gclstm(node_feat, edge_index, H=h, C=c)
    assert h.shape == (B, Z)
    assert c.shape == (B, Z)
    assert not torch.isnan(h).any()
    assert not torch.isnan(c).any()
