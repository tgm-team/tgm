import torch

from tgm.nn.encoder import CTAN, CTANMemory, LastAggregator


def test_ctan_last_aggre():
    B, S = 10, 1
    E, M, T = 7, 5, 2

    edge_index = torch.randint(0, B, size=(2, B))
    edge_time = torch.randint(0, B, size=(B,))
    edge_feat = torch.randint(0, B, size=(B, E))
    memory = CTANMemory(
        B,
        M,
        aggr_module=LastAggregator(),
    )
    encoder = CTAN(
        edge_dim=E,
        memory_dim=M,
        time_dim=T,
        node_dim=S,
    )
    memory.train()
    encoder.train()
    z, last_update = memory(torch.unique(edge_index))
    z = torch.cat([z, torch.rand((len(z), 1))], dim=-1)  # Dummy random node feats
    z = encoder(z, last_update, edge_index, edge_time, edge_feat)
    memory.detach()
    memory.reset_parameters()

    assert z.shape == (B, M)
    assert not torch.isnan(z).any()

    memory.eval()
    encoder.eval()
    z, last_update = memory(torch.unique(edge_index))
    z = torch.cat([z, torch.rand((len(z), 1))], dim=-1)  # Dummy random node feats
    z = encoder(z, last_update, edge_index, edge_time, edge_feat)

    memory.update_state(
        src=edge_index[0],
        pos_dst=edge_index[1],
        t=edge_time,
        src_emb=z[edge_index[0]],
        pos_dst_emb=z[edge_index[1]],
    )
    memory.detach()

    assert z.shape == (B, M)
    assert not torch.isnan(z).any()
