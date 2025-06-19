import pytest
import torch

from tgm.nn.attention import TemporalAttention


def test_temporal_attention_bad_init_shape():
    node_dim, edge_dim, time_dim = 2, 4, 5
    n_heads, out_dim = 3, 5  # BAD: out_dim % n_heads == 0 must be true!
    with pytest.raises(ValueError):
        TemporalAttention(n_heads, node_dim, edge_dim, time_dim, out_dim)

    node_dim = 0
    n_heads, out_dim = 3, 6
    with pytest.raises(ValueError):
        TemporalAttention(n_heads, node_dim, edge_dim, time_dim, out_dim)


def test_temporal_attention_forward():
    node_dim, edge_dim, time_dim = 2, 4, 5
    n_heads, out_dim = 3, 6
    attn = TemporalAttention(n_heads, node_dim, edge_dim, time_dim, out_dim)

    batch_size, num_nbr = 32, 7
    num_nbr = 7

    node_feat = torch.rand(batch_size, node_dim)
    time_feat = torch.rand(batch_size, time_dim)
    edge_feat = torch.rand(batch_size, num_nbr, edge_dim)
    nbr_node_feat = torch.rand(batch_size, num_nbr, node_dim)
    nbr_time_feat = torch.rand(batch_size, num_nbr, time_dim)
    nbr_mask = torch.zeros(batch_size, num_nbr)

    out = attn(node_feat, time_feat, edge_feat, nbr_node_feat, nbr_time_feat, nbr_mask)
    assert torch.is_tensor(out)
    assert out.shape == (batch_size, out_dim)
