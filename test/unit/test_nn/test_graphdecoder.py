import pytest
import torch

from tgm.exceptions import BadAggregatorProtocolError
from tgm.nn import GraphPredictor
from tgm.nn.modules import MeanEmbdPooling, SumEmbdPooling


class FooPooling:
    # Bad pooling operation implementation: missing implementation of def dim() -> int:
    def __call__(self, z: torch.Tensor):
        return z


@pytest.fixture
def node_embedding_factory():
    node_embeddings = torch.rand(200, 128)
    return node_embeddings


def test_pooling():
    node_embeddings = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]
    )
    pooling_op = SumEmbdPooling(5)
    assert pooling_op.out_channels == 5
    result = pooling_op(node_embeddings)
    expected = torch.tensor([7.0, 9.0, 11.0, 13.0, 15.0])
    assert torch.equal(result, expected)

    pooling_op = MeanEmbdPooling(5)
    assert pooling_op.dim == 5
    result = pooling_op(node_embeddings)
    expected = torch.tensor([3.5, 4.5, 5.5, 6.5, 7.5], dtype=torch.float32)
    assert torch.equal(result, expected)


def test_output(node_embedding_factory):
    decoder = GraphPredictor(in_dim=128, nlayers=5, hidden_dim=64)
    node_embeddings = node_embedding_factory

    out = decoder(node_embeddings)

    assert not torch.isnan(out).any()
    assert len(decoder.model) == 9  # 5 layers + 4 ReLU
    assert out.shape[0] == 1

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


def test_bad_pooling_op():
    with pytest.raises(BadAggregatorProtocolError):
        GraphPredictor(in_dim=128, nlayers=5, hidden_dim=64, graph_pooling=FooPooling())
