from unittest.mock import patch

import pytest
import torch

from tgm.exceptions import DependencyError
from tgm.nn import NCNPredictor


@pytest.mark.parametrize('k', [2, 4, 8])
def test_ncn(k):
    B, D, Z = 2, 5, 1
    ncn = NCNPredictor(in_channels=D, hidden_dim=Z, out_channels=Z, k=k)
    node_feat = torch.rand((B, D))
    edge_index = torch.randint(0, B, size=(B, 2))
    h = ncn(node_feat, edge_index, edge_index)
    assert h.shape[0] == (B)
    assert not torch.isnan(h).any()


def test_bad_init():
    with pytest.raises(ValueError):
        B, D, Z = 2, 5, 1
        ncn = NCNPredictor(in_channels=D, hidden_dim=Z, out_channels=Z, k=16)


def test_force_torch_sparse_without_installing():
    B, D, Z = 2, 5, 1
    with patch(
        'tgm.nn.decoder.ncnpred._has_torch_sparse',
        return_value=False,
    ):
        with pytest.raises(DependencyError):
            from tgm.nn import NCNPredictor

            ncn = NCNPredictor(
                in_channels=D, hidden_dim=Z, out_channels=Z, k=2, use_torch_sparse=True
            )
