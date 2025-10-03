import pytest
import torch

from tgm.nn import LinkPredictor
from tgm.nn.decoder.linkproppred import cat_merge


@pytest.fixture
def edge_factory():
    src = torch.rand(200, 128)
    dst = torch.rand(200, 128)
    return src, dst


def test_cat_merge():
    src = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    dst = torch.tensor([[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]])
    merge_result = cat_merge(src, dst)

    expected = torch.tensor(
        [[1, 2, 3, 4, 5, 11, 12, 13, 14, 15], [6, 7, 8, 9, 10, 16, 17, 18, 19, 20]]
    )

    assert torch.equal(expected, merge_result)


def test_output(edge_factory):
    decoder = LinkPredictor(node_dim=128, nlayers=5, hidden_dim=64)
    src, dst = edge_factory

    out = decoder(src, dst)

    assert not torch.isnan(out).any()
    assert len(decoder.model) == 9  # 5 layers + 4 ReLU
    assert out.shape[0] == 200
    assert out.shape[1] == 1

    # check the first layer
    assert decoder.model[0].in_features == 128 * 2  # concat 2 nodes embeddings
    assert decoder.model[0].out_features == 64  # concat 2 nodes embeddings

    # check the last layer
    assert decoder.model[-1].in_features == 64
    assert decoder.model[-1].out_features == 1

    for i in range(
        2, len(decoder.model) - 1, 2
    ):  # exclude the first and the last layer
        assert decoder.model[i].in_features == 64
        assert decoder.model[i].out_features == 64


def test_bad_init():
    with pytest.raises(ValueError):
        LinkPredictor(node_dim=128, nlayers=5, hidden_dim=64, merge_op='foo')
