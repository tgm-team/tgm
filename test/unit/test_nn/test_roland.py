import pytest
import torch

from tgm.nn import ROLAND


@pytest.mark.parametrize('update', ['moving', 'learnable', 'gru', 'mlp', None])
def test_roland(update):
    B, D, Z = 2, 5, 3
    roland = ROLAND(input_channel=D, out_channel=Z, num_nodes=B, update=update)
    node_feat = torch.rand((B, D))
    edge_index = torch.randint(0, B, size=(B, 2))
    last_embeddings = [
        torch.Tensor([[0 for _ in range(Z)] for _ in range(B)]),
        torch.Tensor([[0 for _ in range(Z)] for _ in range(B)]),
    ]
    embeddings = roland(node_feat, edge_index, last_embeddings, B, None)
    last_embeddings = embeddings
    assert len(embeddings) == 2

    assert embeddings[0].shape == (B, Z)
    assert embeddings[1].shape == (B, Z)
    assert not torch.isnan(embeddings[0]).any()
    assert not torch.isnan(embeddings[1]).any()

    embeddings = roland(node_feat, edge_index, last_embeddings, B, B)
    last_embeddings = embeddings
    assert len(embeddings) == 2

    assert embeddings[0].shape == (B, Z)
    assert embeddings[1].shape == (B, Z)
    assert not torch.isnan(embeddings[0]).any()
    assert not torch.isnan(embeddings[1]).any()

    roland.reset_parameters()
