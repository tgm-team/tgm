import pytest
import torch

from tgm.nn import DyGFormer


def test_dygformer():
    B, D, H = 7, 5, 4
    node_dim, edge_dim, time_dim = 16, 17, 18
    dygformer = DyGFormer(
        node_feat_dim=node_dim,
        edge_feat_dim=edge_dim,
        time_feat_dim=time_dim,
        output_dim=edge_dim,
        channel_embedding_dim=H,
        max_input_sequence_length=D + 1,
    )
    x = torch.rand(2 * B, node_dim)
    edge_index = torch.randint(0, B, size=(2, B))
    edge_time = torch.rand(B)
    nbrs = torch.randint(0, B, size=(2 * B, D))
    nbrs_time = torch.rand(2 * B, D)
    nbrs_edge_feat = torch.rand(2 * B, D, edge_dim)

    z_src, z_dst = dygformer(x, edge_index, edge_time, nbrs, nbrs_time, nbrs_edge_feat)
    assert z_src.shape == (B, edge_dim)
    assert z_dst.shape == (B, edge_dim)
    assert not torch.isnan(z_src).any()
    assert not torch.isnan(z_dst).any()


def test_dygformer_bad_init_patch_size():
    with pytest.raises(ValueError):
        DyGFormer(
            node_feat_dim=1,
            edge_feat_dim=1,
            time_feat_dim=1,
            channel_embedding_dim=1,
            max_input_sequence_length=8,
            patch_size=7,  # must divide max_input_sequence_length
        )
