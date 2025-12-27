import pytest
import torch

from tgm.nn import NCNPredictor


@pytest.mark.parametrize('k', [2, 4, 8])
@pytest.mark.parametrize('cn_time_decay', [True, False])
def test_ncn(k, cn_time_decay):
    B, D, Z = 2, 5, 1
    ncn = NCNPredictor(
        in_channels=D,
        hidden_dim=Z,
        out_channels=Z,
        k=k,
        cn_time_decay=cn_time_decay,
    )
    node_feat = torch.rand((B, D))
    edge_index = torch.randint(0, B, size=(B, 2))
    edge_time = torch.randint(0, B, size=(B,))
    last_update = torch.zeros(size=(B,))

    h = ncn(node_feat, edge_index, edge_index, (last_update, edge_time))
    assert h.shape[0] == (B)
    assert not torch.isnan(h).any()


@pytest.mark.parametrize('k', [2, 4, 8])
def test_missing_time_info(k):
    with pytest.raises(RuntimeError):
        B, D, Z = 2, 5, 1
        ncn = NCNPredictor(
            in_channels=D,
            hidden_dim=Z,
            out_channels=Z,
            k=k,
            cn_time_decay=True,
        )
        node_feat = torch.rand((B, D))
        edge_index = torch.randint(0, B, size=(B, 2))
        ncn(node_feat, edge_index, edge_index)


def test_bad_init():
    with pytest.raises(ValueError):
        B, D, Z = 2, 5, 1
        ncn = NCNPredictor(in_channels=D, hidden_dim=Z, out_channels=Z, k=16)
