import pytest
import torch

from tgm.nn import NodePredictor


@pytest.fixture
def node_embedding_factory():
    node_embeddings = torch.rand(200, 128)
    return node_embeddings


def test_output(node_embedding_factory):
    decoder = NodePredictor(in_dim=128, nlayers=5, hidden_dim=64)
    node_embeddings = node_embedding_factory

    out = decoder(node_embeddings)

    assert not torch.isnan(out).any()
    assert len(decoder.model) == 9  # 5 layers + 4 ReLU
    assert out.shape[0] == 200
    assert out.shape[1] == 1

    # check the first layer
    assert decoder.model[0].in_features == 128  # concat 2 nodes embeddings
    assert decoder.model[0].out_features == 64  # concat 2 nodes embeddings

    # check the last layer
    assert decoder.model[-1].in_features == 64
    assert decoder.model[-1].out_features == 1

    for i in range(
        2, len(decoder.model) - 1, 2
    ):  # exclude the first and the last layer
        assert decoder.model[i].in_features == 64
        assert decoder.model[i].out_features == 64
