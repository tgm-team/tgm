import torch

from tgm.nn import RandomProjectionModule, TPNet


def test_tpnet():
    B, D = 7, 5
    node_dim, edge_dim, time_dim = 16, 17, 18
    tpnet = TPNet(
        node_feat_dim=node_dim,
        edge_feat_dim=edge_dim,
        time_feat_dim=time_dim,
        output_dim=edge_dim,
        num_neighbors=D,
    )
    x = torch.rand(2 * B, node_dim)
    edge_index = torch.randint(0, B, size=(2, B))
    edge_time = torch.rand(B)
    nbrs = torch.randint(0, B, size=(2 * B, D))
    nbrs_time = torch.rand(2 * B, D)
    nbrs_edge_feat = torch.rand(2 * B, D, edge_dim)

    z_src, z_dst = tpnet(x, edge_index, edge_time, nbrs, nbrs_time, nbrs_edge_feat)
    assert z_src.shape == (B, edge_dim)
    assert z_dst.shape == (B, edge_dim)
    assert not torch.isnan(z_src).any()
    assert not torch.isnan(z_dst).any()


def test_random_prj_default():
    N, L = 20, 2

    rp = RandomProjectionModule(
        num_nodes=N, num_layer=L, time_decay_weight=0.000001, beginning_time=0.0
    )
    src = torch.arange(0, 10)
    dst = torch.arange(10, 20)
    out = rp(src, dst)
    assert out.shape == (10, (2 * L + 2) ** 2)
    assert not torch.isnan(out).any()


def test_random_prj_lightweight():
    N, L = 20, 2

    rp = RandomProjectionModule(
        num_nodes=N,
        num_layer=L,
        time_decay_weight=0.000001,
        beginning_time=0.0,
        concat_src_dst=False,
    )
    src = torch.arange(0, 10)
    dst = torch.arange(10, 20)
    out = rp(src, dst)
    assert out.shape == (10, (L + 1) ** 2)
    assert not torch.isnan(out).any()
