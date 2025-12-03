import torch

from tgm.nn import EvolveGCNH, EvolveGCNO


def test_evolvegcno():
    B, D = 2, 5
    model = EvolveGCNO(in_channels=D)
    node_feat = torch.rand((B, D))
    edge_index = torch.randint(0, B, size=(B, 2))
    h = model(node_feat, edge_index)
    assert h.shape == (B, D)
    assert not torch.isnan(h).any()

    h = model(node_feat, edge_index)
    assert h.shape == (B, D)
    assert not torch.isnan(h).any()


def test_evolvegcnh():
    B, D = (
        100,
        5,
    )  # for some reason, this only works when B > D, this is a weird thing with evolvegcnh
    model = EvolveGCNH(num_nodes=B, in_channels=D)
    node_feat = torch.rand((B, D))
    edge_index = torch.stack((torch.arange(0, B), torch.arange(0, B)))
    h = model(node_feat, edge_index)
    assert h.shape == (B, D)
    assert not torch.isnan(h).any()

    h = model(node_feat, edge_index)
    assert h.shape == (B, D)
    assert not torch.isnan(h).any()
