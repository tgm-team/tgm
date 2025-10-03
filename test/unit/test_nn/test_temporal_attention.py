import pytest
import torch

from tgm.nn import TemporalAttention


def test_temporal_attention_bad_init_shape():
    node_dim, edge_dim, time_dim, n_heads = 0, 4, 5, 3
    with pytest.raises(ValueError):
        TemporalAttention(n_heads, node_dim, edge_dim, time_dim)

    node_dim, edge_dim, time_dim, n_heads = 4, 0, 5, 3
    with pytest.raises(ValueError):
        TemporalAttention(n_heads, node_dim, edge_dim, time_dim)

    node_dim, edge_dim, time_dim, n_heads = 4, 5, 0, 3
    with pytest.raises(ValueError):
        TemporalAttention(n_heads, node_dim, edge_dim, time_dim)

    node_dim, edge_dim, time_dim, n_heads = 4, 5, 3, 0
    with pytest.raises(ValueError):
        TemporalAttention(n_heads, node_dim, edge_dim, time_dim)


def test_temporal_attention_forward():
    node_dim, edge_dim, time_dim, n_heads = 2, 4, 8, 5
    attn = TemporalAttention(n_heads, node_dim, edge_dim, time_dim)
    assert attn.pad_dim == 0
    assert attn.out_dim == node_dim + time_dim + attn.pad_dim

    batch_size, num_nbr = 32, 7
    node_feat = torch.rand(batch_size, node_dim)
    time_feat = torch.rand(batch_size, time_dim)
    edge_feat = torch.rand(batch_size, num_nbr, edge_dim)
    nbr_node_feat = torch.rand(batch_size, num_nbr, node_dim)
    nbr_time_feat = torch.rand(batch_size, num_nbr, time_dim)
    nbr_mask = torch.zeros(batch_size, num_nbr, dtype=bool)

    out = attn(node_feat, time_feat, edge_feat, nbr_node_feat, nbr_time_feat, nbr_mask)
    assert torch.is_tensor(out)
    assert out.shape == (batch_size, node_dim + time_dim)


def test_temporal_attention_forward_with_padding():
    node_dim, edge_dim, time_dim = 2, 4, 8
    n_heads = 3  # out_dim = node_dim + time_dim = 10
    attn = TemporalAttention(n_heads, node_dim, edge_dim, time_dim)

    assert attn.pad_dim == 2  # n_heads - out_dim % n_heads
    assert attn.out_dim == node_dim + time_dim + attn.pad_dim

    batch_size, num_nbr = 32, 7
    node_feat = torch.rand(batch_size, node_dim)
    time_feat = torch.rand(batch_size, time_dim)
    edge_feat = torch.rand(batch_size, num_nbr, edge_dim)
    nbr_node_feat = torch.rand(batch_size, num_nbr, node_dim)
    nbr_time_feat = torch.rand(batch_size, num_nbr, time_dim)
    nbr_mask = torch.zeros(batch_size, num_nbr, dtype=bool)

    out = attn(node_feat, time_feat, edge_feat, nbr_node_feat, nbr_time_feat, nbr_mask)
    assert torch.is_tensor(out)
    assert out.shape == (batch_size, node_dim + time_dim + attn.pad_dim)
