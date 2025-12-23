import torch

from tgm.nn import TGNMemory
from tgm.nn.encoder.tgn import (
    GraphAttentionEmbedding,
    IdentityMessage,
    LastAggregator,
)


def test_tgn():
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
    z, last_update = memory(torch.unique(edge_index))
    z = encoder(z, last_update, edge_index, edge_time, edge_feat)

    assert z.shape == (B, Z)
    assert not torch.isnan(z).any()
