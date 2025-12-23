import torch

from tgm.nn import TGNMemory
from tgm.nn.encoder.tgn import (
    GraphAttentionEmbedding,
    IdentityMessage,
    LastAggregator,
    MeanAggregator,
)


def test_tgn_last_aggre():
    B, Z = 10, 100
    E, M, T = 7, 5, 2

    edge_index = torch.randint(0, B, size=(2, B))
    edge_time = torch.randint(0, B, size=(B,))
    edge_feat = torch.randint(0, B, size=(B, E))
    memory = TGNMemory(
        B,
        E,
        M,
        T,
        message_module=IdentityMessage(E, M, T),
        aggregator_module=LastAggregator(),
    )
    encoder = GraphAttentionEmbedding(
        in_channels=M,
        out_channels=Z,
        msg_dim=E,
        time_enc=memory.time_enc,
    )
    memory.train()
    encoder.train()
    z, last_update = memory(torch.unique(edge_index))
    z = encoder(z, last_update, edge_index, edge_time, edge_feat)
    memory.detach()
    memory.reset_parameters()

    assert z.shape == (B, Z)
    assert not torch.isnan(z).any()

    memory.eval()
    encoder.eval()
    z, last_update = memory(torch.unique(edge_index))
    z = encoder(z, last_update, edge_index, edge_time, edge_feat)
    memory.detach()

    assert z.shape == (B, Z)
    assert not torch.isnan(z).any()


def test_tgn_mean_aggre():
    B, Z = 10, 100
    E, M, T = 7, 5, 2

    edge_index = torch.randint(0, B, size=(2, B))
    edge_time = torch.randint(0, B, size=(B,))
    edge_feat = torch.randint(0, B, size=(B, E))
    memory = TGNMemory(
        B,
        E,
        M,
        T,
        message_module=IdentityMessage(E, M, T),
        aggregator_module=MeanAggregator(),
    )
    encoder = GraphAttentionEmbedding(
        in_channels=M,
        out_channels=Z,
        msg_dim=E,
        time_enc=memory.time_enc,
    )
    memory.train()
    encoder.train()
    z, last_update = memory(torch.unique(edge_index))
    z = encoder(z, last_update, edge_index, edge_time, edge_feat)
    memory.update_state(edge_index[0], edge_index[1], edge_time, edge_feat.float())

    memory.detach()

    assert z.shape == (B, Z)
    assert not torch.isnan(z).any()

    memory.eval()
    encoder.eval()
    z, last_update = memory(torch.unique(edge_index))
    z = encoder(z, last_update, edge_index, edge_time, edge_feat)
    memory.update_state(edge_index[0], edge_index[1], edge_time, edge_feat.float())
    memory.detach()

    assert z.shape == (B, Z)
    assert not torch.isnan(z).any()
