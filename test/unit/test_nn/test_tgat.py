import torch

from tgm.nn.encoder import TGAT


def test_tgat():
    num_nbr = 7
    node_dim = 16
    time_dim = 32
    edge_dim = 8
    num_hop = 3
    num_nodes = 20
    emd_dim = 64

    node_feat = torch.rand(num_nodes, node_dim)
    seed_nodes, seed_times = [], []
    nbr_nids, nbr_edge_x, nbr_edge_time = [], [], []

    for i in range(num_hop):
        num_seed_nodes = num_nodes * (num_nbr**i)
        seed_nodes.append(torch.randint(0, num_nodes, size=(num_seed_nodes,)))
        seed_times.append(torch.randint(0, num_seed_nodes, size=(num_seed_nodes,)))
        nbr_nids.append(torch.randint(0, num_nodes, size=(num_seed_nodes, num_nbr)))
        nbr_edge_x.append(torch.rand(num_seed_nodes, num_nbr, edge_dim))
        nbr_edge_time.append(
            torch.randint(0, num_seed_nodes, size=(num_seed_nodes, num_nbr))
        )

    model = TGAT(
        node_dim=node_dim,
        edge_dim=edge_dim,
        time_dim=time_dim,
        embed_dim=emd_dim,
        num_layers=num_hop,
    )
    z = model(
        X=node_feat,
        seed_nids=seed_nodes,
        seed_times=seed_times,
        nbr_nids=nbr_nids,
        nbr_edge_x=nbr_edge_x,
        nbr_edge_time=nbr_edge_time,
    )

    assert z.shape == (num_nodes, emd_dim)
    assert not torch.isnan(z).any()
