import torch

from tgm.nn import TGNMemory
from tgm.nn.encoder.tgn import (
    EncodeIndexMessage,
    GraphAttentionEmbedding,
    IdentityMessage,
    LastAggregator,
    MeanAggregator,
    TGNv2Memory,
)


def test_tgn_last_aggre():
    B, Z = 10, 100
    E, M, T = 7, 5, 2

    n_id = torch.arange(B)
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
    z, last_update = memory(n_id)
    z = encoder(z, last_update, edge_index, edge_time, edge_feat)
    memory.detach()
    memory.reset_parameters()

    assert z.shape == (B, Z)
    assert not torch.isnan(z).any()

    memory.eval()
    encoder.eval()
    z, last_update = memory(n_id)
    z = encoder(z, last_update, edge_index, edge_time, edge_feat)
    memory.detach()

    assert z.shape == (B, Z)
    assert not torch.isnan(z).any()


def test_tgn_mean_aggre():
    B, Z = 10, 100
    E, M, T = 7, 5, 2

    n_id = torch.arange(B)
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
    z, last_update = memory(n_id)
    z = encoder(z, last_update, edge_index, edge_time, edge_feat)
    memory.update_state(edge_index[0], edge_index[1], edge_time, edge_feat.float())

    memory.detach()

    assert z.shape == (B, Z)
    assert not torch.isnan(z).any()

    memory.eval()
    encoder.eval()
    z, last_update = memory(n_id)
    z = encoder(z, last_update, edge_index, edge_time, edge_feat)
    memory.update_state(edge_index[0], edge_index[1], edge_time, edge_feat.float())
    memory.detach()

    assert z.shape == (B, Z)
    assert not torch.isnan(z).any()


def test_tgnv2_encode_index_message():
    B, E, M, T, I = 3, 2, 4, 5, 6
    msg_module = EncodeIndexMessage(E, M, T, I)

    z_src = torch.randn(B, M)
    z_dst = torch.randn(B, M)
    raw_msg = torch.randn(B, E)
    t_enc = torch.randn(B, T)
    src_enc = torch.randn(B, I)
    dst_enc = torch.randn(B, I)

    out = msg_module(z_src, z_dst, raw_msg, t_enc, src_enc, dst_enc)
    expected = torch.cat([z_src, z_dst, raw_msg, src_enc, dst_enc, t_enc], dim=-1)

    assert msg_module.out_channels == 2 * M + E + 2 * I + T
    assert out.shape == (B, msg_module.out_channels)
    torch.testing.assert_close(out, expected)


def test_tgnv2_memory():
    B, Z = 10, 100
    E, M, T, I = 7, 5, 2, 3

    n_id = torch.arange(B)
    edge_index = torch.randint(0, B, size=(2, B))
    edge_time = torch.randint(0, B, size=(B,))
    edge_feat = torch.randn(B, E)
    memory = TGNv2Memory(
        B,
        E,
        M,
        T,
        I,
        message_module=EncodeIndexMessage(E, M, T, I),
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
    z, last_update = memory(n_id)
    z = encoder(z, last_update, edge_index, edge_time, edge_feat)
    memory.update_state(edge_index[0], edge_index[1], edge_time, edge_feat)
    memory.detach()

    assert z.shape == (B, Z)
    assert not torch.isnan(z).any()

    memory.eval()
    encoder.eval()
    z, last_update = memory(n_id)
    z = encoder(z, last_update, edge_index, edge_time, edge_feat)
    memory.update_state(edge_index[0], edge_index[1], edge_time, edge_feat)
    memory.detach()

    assert z.shape == (B, Z)
    assert not torch.isnan(z).any()
