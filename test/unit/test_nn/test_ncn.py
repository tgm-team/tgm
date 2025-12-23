import pytest
import torch

from tgm.nn import NCNPredictor
from tgm.nn.decoder.ncnpred import HAS_TORCH_SPARSE


@pytest.mark.parametrize('k', [2, 4, 8])
def test_ncn(k):
    B, D, Z = 2, 5, 1
    ncn = NCNPredictor(in_channels=D, hidden_dim=Z, out_channels=Z, k=k)
    node_feat = torch.rand((B, D))
    edge_index = torch.randint(0, B, size=(B, 2))
    h = ncn(node_feat, edge_index, edge_index)
    assert h.shape[0] == (B)
    assert not torch.isnan(h).any()


@pytest.mark.skipif(not HAS_TORCH_SPARSE, reason='torch_sparse required')
def test_ncn_sparse(k):
    B, D, Z = 2, 5, 1
    ncn = NCNPredictor(in_channels=D, hidden_dim=Z, out_channels=Z, k=k)
    node_feat = torch.rand((B, D))
    edge_index = torch.randint(0, B, size=(B, 2))
    h = ncn(node_feat, edge_index, edge_index)
    assert h.shape[0] == (B)
    assert not torch.isnan(h).any()
