"""Regression tests for the direction of the neighbor edge_index.

PyG's TransformerConv (flow='source_to_target') aggregates messages at
edge_index[1], so a seed node only attends over its sampled neighbors when the
neighbor edge_index is built as [neighbor, seed] (matching TGB/PyG's
LastNeighborLoader). The reversed [seed, neighbor] ordering silently cuts the
seed off from its neighbors. See tgm-team/tgm#427.
"""

import pytest
import torch

from tgm.nn.encoder import CTAN
from tgm.nn.encoder.tgn import GraphAttentionEmbedding
from tgm.nn.modules.time_encoding import Time2Vec

SEED = 0
NBRS = [1, 2, 3]
NUM_NODES = 1 + len(NBRS)
X_DIM, MSG_DIM, TIME_DIM = 8, 4, 2


def _make_graph_attention_embedding() -> torch.nn.Module:
    return GraphAttentionEmbedding(
        in_channels=X_DIM,
        out_channels=X_DIM,
        msg_dim=MSG_DIM,
        time_enc=Time2Vec(TIME_DIM),
    )


def _make_ctan() -> torch.nn.Module:
    # CTAN's enc_x maps (memory_dim + node_dim) -> memory_dim
    return CTAN(edge_dim=MSG_DIM, memory_dim=X_DIM - 1, time_dim=TIME_DIM, node_dim=1)


ENCODERS = {
    'graph_attention_embedding': _make_graph_attention_embedding,
    'ctan': _make_ctan,
}


def _inputs():
    x = torch.randn(NUM_NODES, X_DIM)
    last_update = torch.tensor([5, 3, 2, 1])
    t = torch.tensor([4, 3, 2])
    msg = torch.randn(len(NBRS), MSG_DIM)
    seeds = torch.full((len(NBRS),), SEED, dtype=torch.long)
    nbrs = torch.tensor(NBRS, dtype=torch.long)
    return x, last_update, t, msg, seeds, nbrs


def _seed_output_delta(encoder, edge_index, x, last_update, t, msg) -> float:
    with torch.no_grad():
        z = encoder(x, last_update, edge_index, t, msg)
        x_perturbed = x.clone()
        x_perturbed[NBRS[0]] += 10.0  # change one neighbor's input features
        z_perturbed = encoder(x_perturbed, last_update, edge_index, t, msg)
    return (z[SEED] - z_perturbed[SEED]).abs().max().item()


@pytest.mark.parametrize('encoder_name', ENCODERS.keys())
def test_seed_depends_on_neighbors_with_nbr_seed_ordering(encoder_name):
    torch.manual_seed(0)
    encoder = ENCODERS[encoder_name]().eval()
    x, last_update, t, msg, seeds, nbrs = _inputs()

    edge_index = torch.stack([nbrs, seeds])  # [neighbor, seed]: correct
    delta = _seed_output_delta(encoder, edge_index, x, last_update, t, msg)
    assert delta > 1e-4


@pytest.mark.parametrize('encoder_name', ENCODERS.keys())
def test_seed_ignores_neighbors_with_reversed_ordering(encoder_name):
    # Documents the bug fixed in #427/#428: with [seed, neighbor] the seed
    # receives no messages, so its embedding is independent of its neighbors.
    torch.manual_seed(0)
    encoder = ENCODERS[encoder_name]().eval()
    x, last_update, t, msg, seeds, nbrs = _inputs()

    edge_index = torch.stack([seeds, nbrs])  # [seed, neighbor]: buggy
    delta = _seed_output_delta(encoder, edge_index, x, last_update, t, msg)
    assert delta == 0.0
