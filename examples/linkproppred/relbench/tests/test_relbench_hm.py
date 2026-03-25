"""Unit tests for the rel-hm → TGM adaptation pipeline.

All tests use synthetic in-memory data so no RelBench download is required.
Tests are grouped by phase and can be run with:
    pytest examples/linkproppred/relbench/tests/test_relbench_hm.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers — synthetic data factories
# ---------------------------------------------------------------------------

N_ART = 10
N_CUST = 5
N_EDGES = 20
TARGET_DIM = 8  # small dim for fast tests


def _make_article_df(n=N_ART) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            'article_id': np.arange(n, dtype=np.int32),
            'product_code': rng.integers(100, 200, n),
            'product_type_no': rng.integers(0, 5, n),
            'department_no': rng.integers(0, 3, n),
            'section_no': rng.integers(0, 4, n),
            'garment_group_no': rng.integers(0, 2, n),
            'perceived_colour_master_id': rng.integers(0, 10, n),
            'product_type_name': rng.choice(['Dress', 'Shirt', 'Pants'], n),
            'product_group_name': rng.choice(
                ['Garment Upper body', 'Garment Lower body'], n
            ),
            'graphical_appearance_no': rng.choice(['Solid', 'Striped'], n),
            'graphical_appearance_name': rng.choice(['Solid', 'Striped'], n),
            'colour_group_code': rng.choice(['Black', 'White'], n),
            'colour_group_name': rng.choice(['Black', 'White'], n),
            'perceived_colour_value_id': rng.choice(['Light', 'Dark'], n),
            'perceived_colour_value_name': rng.choice(['Light', 'Dark'], n),
            'perceived_colour_master_name': rng.choice(['Grey', 'Blue'], n),
            'department_name': rng.choice(['Ladies', 'Men'], n),
            'index_code': rng.choice(['A', 'B'], n),
            'index_name': rng.choice(['Ladieswear', 'Menswear'], n),
            'index_group_no': rng.choice(['1', '2'], n),
            'index_group_name': rng.choice(['Ladieswear', 'Divided'], n),
            'garment_group_name': rng.choice(['Jersey', 'Denim'], n),
        }
    )


def _make_customer_df(n=N_CUST) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            'customer_id': np.arange(n, dtype=np.int32),
            'age': rng.integers(18, 65, n).astype(float),
            'FN': rng.choice(['1', float('nan')], n),
            'Active': rng.choice(['1', float('nan')], n),
            'club_member_status': rng.choice(['ACTIVE', 'PRE-CREATE'], n),
            'fashion_news_frequency': rng.choice(['Regularly', 'None'], n),
            'postal_code': rng.choice(['12345', '67890', '11111'], n),
        }
    )


def _make_transactions_df(n_art=N_ART, n_cust=N_CUST, n=N_EDGES) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    # Use integer unix timestamps (seconds) to avoid timezone complications
    base_ts = 1_546_300_800  # 2019-01-01 00:00:00 UTC
    timestamps = sorted((base_ts + rng.integers(0, 86400 * 365, n)).tolist())
    return pd.DataFrame(
        {
            't_dat': pd.to_datetime(timestamps, unit='s'),
            'price': rng.random(n).astype(np.float32),
            'sales_channel_id': rng.integers(1, 3, n).astype(np.float32),
            'customer_id': rng.integers(0, n_cust, n).astype(np.int32),
            'article_id': rng.integers(0, n_art, n).astype(np.int32),
        }
    )


# ---------------------------------------------------------------------------
# Phase 1 Tests — data.py
# ---------------------------------------------------------------------------


class TestTaskOneOne:
    """Task 1.1: raw table shapes and dtypes."""

    def test_article_has_article_id(self):
        df = _make_article_df()
        assert 'article_id' in df.columns

    def test_customer_has_customer_id(self):
        df = _make_customer_df()
        assert 'customer_id' in df.columns

    def test_transactions_required_cols(self):
        df = _make_transactions_df()
        for col in ['t_dat', 'price', 'sales_channel_id', 'customer_id', 'article_id']:
            assert col in df.columns

    def test_table_shapes(self):
        assert len(_make_article_df()) == N_ART
        assert len(_make_customer_df()) == N_CUST
        assert len(_make_transactions_df()) == N_EDGES


class TestTaskOneThree:
    """Task 1.3: node ID remapping."""

    def test_customer_ids_shifted(self):
        art = _make_article_df()
        cust = _make_customer_df()
        trans = _make_transactions_df()
        # Dummy split tables
        train_t = trans[['customer_id', 'article_id']].copy()
        val_t = train_t.copy()
        test_t = train_t.copy()

        from examples.linkproppred.relbench.data import remap_node_ids

        offset = remap_node_ids(art, cust, trans, train_t, val_t, test_t)

        expected_offset = int(art['article_id'].max()) + 1
        assert offset == expected_offset
        assert cust['customer_id'].min() >= expected_offset
        assert trans['customer_id'].min() >= expected_offset

    def test_no_id_collision(self):
        art = _make_article_df()
        cust = _make_customer_df()
        trans = _make_transactions_df()
        train_t = trans[['customer_id', 'article_id']].copy()
        val_t = train_t.copy()
        test_t = train_t.copy()

        from examples.linkproppred.relbench.data import remap_node_ids

        remap_node_ids(art, cust, trans, train_t, val_t, test_t)

        art_ids = set(art['article_id'].tolist())
        cust_ids = set(cust['customer_id'].tolist())
        assert art_ids.isdisjoint(cust_ids), 'Article and customer IDs must not overlap'

    def test_n_nodes_within_int32(self):
        art = _make_article_df()
        cust = _make_customer_df()
        n_nodes = len(art) + len(cust)
        assert n_nodes < 2**31 - 1


class TestTaskOneFour:
    """Task 1.4: edge tensor construction."""

    def test_edge_time_non_negative(self):
        trans = _make_transactions_df()
        art = _make_article_df()
        cust = _make_customer_df()
        train_t = trans[['customer_id', 'article_id']].copy()
        val_t = train_t.copy()
        test_t = train_t.copy()

        from examples.linkproppred.relbench.data import (
            build_edge_tensors,
            remap_node_ids,
        )

        remap_node_ids(art, cust, trans, train_t, val_t, test_t)
        edge_time, edge_index, edge_x = build_edge_tensors(trans)

        assert (edge_time >= 0).all()

    def test_edge_tensors_shapes(self):
        trans = _make_transactions_df()
        art = _make_article_df()
        cust = _make_customer_df()
        train_t = trans[['customer_id', 'article_id']].copy()
        val_t = train_t.copy()
        test_t = train_t.copy()

        from examples.linkproppred.relbench.data import (
            build_edge_tensors,
            remap_node_ids,
        )

        remap_node_ids(art, cust, trans, train_t, val_t, test_t)
        edge_time, edge_index, edge_x = build_edge_tensors(trans)

        assert edge_time.shape == (N_EDGES,)
        assert edge_index.shape == (N_EDGES, 2)
        assert edge_x.shape == (N_EDGES, 2)

    def test_edge_time_sorted(self):
        trans = _make_transactions_df()
        art = _make_article_df()
        cust = _make_customer_df()
        train_t = trans[['customer_id', 'article_id']].copy()
        val_t = train_t.copy()
        test_t = train_t.copy()

        from examples.linkproppred.relbench.data import (
            build_edge_tensors,
            remap_node_ids,
        )

        remap_node_ids(art, cust, trans, train_t, val_t, test_t)
        edge_time, _, _ = build_edge_tensors(trans)

        diffs = edge_time[1:] - edge_time[:-1]
        assert (diffs >= 0).all(), 'edge_time must be non-decreasing'

    def test_edge_x_is_float32(self):
        trans = _make_transactions_df()
        art = _make_article_df()
        cust = _make_customer_df()
        train_t = trans[['customer_id', 'article_id']].copy()
        val_t = train_t.copy()
        test_t = train_t.copy()

        from examples.linkproppred.relbench.data import (
            build_edge_tensors,
            remap_node_ids,
        )

        remap_node_ids(art, cust, trans, train_t, val_t, test_t)
        _, _, edge_x = build_edge_tensors(trans)
        assert edge_x.dtype == torch.float32


# ---------------------------------------------------------------------------
# Phase 2 Tests — embed.py
# ---------------------------------------------------------------------------


class TestTaskTwoOne:
    """Task 2.1: column-to-stype mappings cover expected columns."""

    def test_article_numerical_cols_present(self):
        from examples.linkproppred.relbench.embed import ARTICLE_COL_TO_STYPE

        numerical = [c for c, s in ARTICLE_COL_TO_STYPE.items() if s == 'numerical']
        assert 'product_code' in numerical
        assert 'department_no' in numerical

    def test_customer_age_is_numerical(self):
        from examples.linkproppred.relbench.embed import CUSTOMER_COL_TO_STYPE

        assert CUSTOMER_COL_TO_STYPE['age'] == 'numerical'

    def test_postal_code_is_categorical(self):
        from examples.linkproppred.relbench.embed import CUSTOMER_COL_TO_STYPE

        assert CUSTOMER_COL_TO_STYPE['postal_code'] == 'categorical'


class TestTaskTwoTwo:
    """Task 2.2: encode_table output shape and dtype."""

    def test_article_encode_shape(self):
        pytest.importorskip('torch_frame')
        from examples.linkproppred.relbench.embed import (
            ARTICLE_COL_TO_STYPE,
            encode_table,
        )

        art = _make_article_df().drop(columns=['article_id'])
        emb = encode_table(art, ARTICLE_COL_TO_STYPE, target_dim=TARGET_DIM)
        assert emb.shape == (N_ART, TARGET_DIM)

    def test_customer_encode_shape(self):
        pytest.importorskip('torch_frame')
        from examples.linkproppred.relbench.embed import (
            CUSTOMER_COL_TO_STYPE,
            encode_table,
        )

        cust = _make_customer_df().drop(columns=['customer_id'])
        emb = encode_table(cust, CUSTOMER_COL_TO_STYPE, target_dim=TARGET_DIM)
        assert emb.shape == (N_CUST, TARGET_DIM)

    def test_encode_float32(self):
        pytest.importorskip('torch_frame')
        from examples.linkproppred.relbench.embed import (
            ARTICLE_COL_TO_STYPE,
            encode_table,
        )

        art = _make_article_df().drop(columns=['article_id'])
        emb = encode_table(art, ARTICLE_COL_TO_STYPE, target_dim=TARGET_DIM)
        assert emb.dtype == torch.float32

    def test_nan_imputation_does_not_raise(self):
        pytest.importorskip('torch_frame')
        from examples.linkproppred.relbench.embed import (
            CUSTOMER_COL_TO_STYPE,
            encode_table,
        )

        cust = _make_customer_df().drop(columns=['customer_id'])
        cust.loc[0, 'age'] = float('nan')
        cust.loc[1, 'club_member_status'] = float('nan')
        emb = encode_table(cust, CUSTOMER_COL_TO_STYPE, target_dim=TARGET_DIM)
        assert not torch.isnan(emb).any()


class TestTaskTwoThree:
    """Task 2.3: static_node_x concatenation."""

    def test_static_node_x_shape(self):
        pytest.importorskip('torch_frame')
        from examples.linkproppred.relbench.embed import (
            build_static_node_features,
        )

        art = _make_article_df()
        cust = _make_customer_df()
        # remap IDs first
        cust = cust.copy()
        cust['customer_id'] += int(art['article_id'].max()) + 1
        snx = build_static_node_features(art, cust, target_dim=TARGET_DIM)
        assert snx.shape == (N_ART + N_CUST, TARGET_DIM)

    def test_static_node_x_no_nan(self):
        pytest.importorskip('torch_frame')
        from examples.linkproppred.relbench.embed import (
            build_static_node_features,
        )

        art = _make_article_df()
        cust = _make_customer_df()
        cust = cust.copy()
        cust['customer_id'] += int(art['article_id'].max()) + 1
        snx = build_static_node_features(art, cust, target_dim=TARGET_DIM)
        assert not torch.isnan(snx).any()


# ---------------------------------------------------------------------------
# Phase 3 Tests — DGData construction & splits
# ---------------------------------------------------------------------------


def _make_full_data(with_static=False):
    """Build a minimal DGData from synthetic tensors (no RelBench download)."""
    from examples.linkproppred.relbench.data import (
        build_dgdata,
        build_edge_tensors,
        remap_node_ids,
    )

    art = _make_article_df()
    cust = _make_customer_df()
    trans = _make_transactions_df()
    train_t = trans[['customer_id', 'article_id']].copy()
    val_t = train_t.copy()
    test_t = train_t.copy()

    remap_node_ids(art, cust, trans, train_t, val_t, test_t)
    edge_time, edge_index, edge_x = build_edge_tensors(trans)

    n_art = len(art)
    n_cust = len(cust)
    n_nodes = n_art + n_cust
    node_type = torch.cat(
        [
            torch.zeros(n_art, dtype=torch.int32),
            torch.ones(n_cust, dtype=torch.int32),
        ]
    )

    static_node_x = torch.randn(n_nodes, TARGET_DIM) if with_static else None

    # Use 40th / 80th percentile timestamps as split boundaries
    ts = edge_time.numpy()
    t_val = int(np.percentile(ts, 40))
    t_test = int(np.percentile(ts, 80))

    data = build_dgdata(
        edge_time, edge_index, edge_x, static_node_x, node_type, t_val, t_test
    )
    return data, n_art, n_cust, n_nodes


class TestTaskThreeOne:
    """Task 3.1: DGData fields."""

    def test_dgdata_num_nodes(self):
        data, n_art, n_cust, n_nodes = _make_full_data()
        assert data.num_nodes == n_nodes

    def test_dgdata_num_edges(self):
        data, *_ = _make_full_data()
        assert data.num_edges == N_EDGES

    def test_dgdata_edge_x_dim(self):
        data, *_ = _make_full_data()
        assert data.edge_x.shape[1] == 2

    def test_dgdata_with_static_node_x(self):
        data, _, _, n_nodes = _make_full_data(with_static=True)
        assert data.static_node_x is not None
        assert data.static_node_x.shape == (n_nodes, TARGET_DIM)

    def test_dgdata_node_type_shape(self):
        data, n_art, n_cust, n_nodes = _make_full_data()
        assert data.node_type.shape[0] == n_nodes
        assert int(data.node_type[:n_art].sum()) == 0  # articles are type 0
        assert int(data.node_type[n_art:].min()) == 1  # customers are type 1


class TestTaskThreeTwo:
    """Task 3.2: split() returns three non-empty DGData objects."""

    def test_split_returns_three(self):
        data, *_ = _make_full_data()
        splits = data.split()
        assert len(splits) == 3

    def test_split_edges_partition(self):
        data, *_ = _make_full_data()
        train, val, test = data.split()
        total = train.num_edges + val.num_edges + test.num_edges
        assert total == N_EDGES

    def test_split_static_node_x_shared(self):
        data, _, _, n_nodes = _make_full_data(with_static=True)
        train, val, test = data.split()
        # static_node_x is shared (not sliced) across splits
        for split in [train, val, test]:
            assert split.static_node_x is not None
            assert split.static_node_x.shape[0] == n_nodes


# ---------------------------------------------------------------------------
# Phase 4 Tests — evaluation helpers
# ---------------------------------------------------------------------------


class TestEvalHelpers:
    """Task 4.2: AP and NDCG@10 helpers."""

    def test_ap_perfect(self):
        from examples.linkproppred.relbench.train import compute_ap_ndcg

        scores = np.array([1.0, 0.0])
        labels = np.array([1, 0])
        ap, _ = compute_ap_ndcg(scores, labels)
        assert ap == pytest.approx(1.0)

    def test_ap_worst(self):
        from examples.linkproppred.relbench.train import compute_ap_ndcg

        scores = np.array([0.0, 1.0])
        labels = np.array([1, 0])
        ap, _ = compute_ap_ndcg(scores, labels)
        assert ap < 1.0

    def test_ndcg_perfect(self):
        from examples.linkproppred.relbench.train import _ndcg_at_k

        relevance = np.array([1, 0, 0, 0, 0])
        ndcg = _ndcg_at_k(relevance, k=5)
        assert ndcg == pytest.approx(1.0)

    def test_ndcg_zero(self):
        from examples.linkproppred.relbench.train import _ndcg_at_k

        relevance = np.array([0, 0, 0, 0, 0])
        ndcg = _ndcg_at_k(relevance, k=5)
        assert ndcg == pytest.approx(0.0)

    def test_ndcg_k_truncation(self):
        from examples.linkproppred.relbench.train import _ndcg_at_k

        relevance = np.array([0, 1])  # positive is rank-2
        ndcg5 = _ndcg_at_k(relevance, k=5)
        ndcg1 = _ndcg_at_k(relevance, k=1)
        assert ndcg5 > 0
        assert ndcg1 == pytest.approx(0.0)  # positive not in top-1


# ---------------------------------------------------------------------------
# Phase 5 Tests — model components (lightweight, no full training run)
# ---------------------------------------------------------------------------


class TestModelComponents:
    """Task 5.1/5.2: model instantiation and forward shapes."""

    @pytest.fixture
    def model_setup(self):
        from tgm.nn import LinkPredictor, TGNMemory
        from tgm.nn.encoder.tgn import (
            GraphAttentionEmbedding,
            IdentityMessage,
            LastAggregator,
        )

        n_nodes = N_ART + N_CUST
        edge_x_dim = 2
        mem_dim = 16
        embed_dim = 16
        time_dim = 16

        memory = TGNMemory(
            n_nodes,
            edge_x_dim,
            mem_dim,
            time_dim,
            message_module=IdentityMessage(edge_x_dim, mem_dim, time_dim),
            aggregator_module=LastAggregator(),
        )
        encoder = GraphAttentionEmbedding(
            in_channels=mem_dim,
            out_channels=embed_dim,
            msg_dim=edge_x_dim,
            time_enc=memory.time_enc,
        )
        decoder = LinkPredictor(node_dim=embed_dim, hidden_dim=embed_dim)
        return memory, encoder, decoder, n_nodes, mem_dim, embed_dim

    def test_memory_output_shape(self, model_setup):
        memory, _, _, n_nodes, mem_dim, _ = model_setup
        nids = torch.arange(5, dtype=torch.int32)
        z, last_update = memory(nids)
        assert z.shape == (5, mem_dim)

    def test_decoder_output_shape(self, model_setup):
        _, _, decoder, _, _, embed_dim = model_setup
        src = torch.randn(4, embed_dim)
        dst = torch.randn(4, embed_dim)
        out = decoder(src, dst)
        assert out.shape == (4,)

    def test_static_augmented_encoder_shape(self, model_setup):
        from examples.linkproppred.relbench.train import StaticAugmentedEncoder

        memory, base_encoder, _, n_nodes, mem_dim, embed_dim = model_setup
        static = torch.randn(n_nodes, TARGET_DIM)
        aug_enc = StaticAugmentedEncoder(
            base_encoder=base_encoder,
            static_node_x=static,
            static_dim=TARGET_DIM,
            memory_dim=mem_dim,
        )
        # Just check the projection layer shape
        assert aug_enc.proj.in_features == mem_dim + TARGET_DIM
        assert aug_enc.proj.out_features == mem_dim

    def test_neg_sampler_hook_range(self):
        from tgm.hooks.negatives import NegativeEdgeSamplerHook

        # Negatives must be within article IDs only
        hook = NegativeEdgeSamplerHook(low=0, high=N_ART)
        assert hook.low == 0
        assert hook.high == N_ART
