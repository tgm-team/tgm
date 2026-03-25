"""Phase 2: pytorch_frame-based static node feature embedding for rel-hm.

Produces a single [N_nodes, TARGET_DIM] float32 tensor from the raw article
and customer DataFrames by:
  1. Defining per-table column → stype mappings (numerical / categorical).
  2. Building a torch_frame Dataset, materialising it, and encoding with
     StypeWiseFeatureEncoder (LinearEncoder per stype).
  3. Concatenating article and customer embeddings (same TARGET_DIM so no
     padding is needed).

Usage:
    from examples.linkproppred.relbench.embed import build_static_node_features
    static_node_x = build_static_node_features(table_article, table_customer)
    # static_node_x: [N_art + N_cust, TARGET_DIM]
"""

from __future__ import annotations

import logging
from typing import Dict

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Unified output dimension for both node types.
# Same dim avoids zero-padding and satisfies DGData's single D_static requirement.
TARGET_DIM: int = 64

# ---------------------------------------------------------------------------
# Task 2.1 — Column → stype mappings
# ---------------------------------------------------------------------------

# Text columns are skipped for the simplest case (no sentence encoder needed).
# garment_group_no classified as numerical per schema_hm.

ARTICLE_COL_TO_STYPE: Dict[str, str] = {
    'product_code': 'numerical',
    'product_type_no': 'numerical',
    'department_no': 'numerical',
    'section_no': 'numerical',
    'garment_group_no': 'numerical',
    'perceived_colour_master_id': 'numerical',
    'product_type_name': 'categorical',
    'product_group_name': 'categorical',
    'graphical_appearance_no': 'categorical',
    'graphical_appearance_name': 'categorical',
    'colour_group_code': 'categorical',
    'colour_group_name': 'categorical',
    'perceived_colour_value_id': 'categorical',
    'perceived_colour_value_name': 'categorical',
    'perceived_colour_master_name': 'categorical',
    'department_name': 'categorical',
    'index_code': 'categorical',
    'index_name': 'categorical',
    'index_group_no': 'categorical',
    'index_group_name': 'categorical',
    'garment_group_name': 'categorical',
    # 'prod_name', 'section_name', 'detail_desc'  -- text: skipped
}

CUSTOMER_COL_TO_STYPE: Dict[str, str] = {
    'age': 'numerical',
    'FN': 'categorical',
    'Active': 'categorical',
    'club_member_status': 'categorical',
    'fashion_news_frequency': 'categorical',
    'postal_code': 'categorical',
}


# ---------------------------------------------------------------------------
# Task 2.2 — Build TensorFrame datasets and encode
# ---------------------------------------------------------------------------


def _impute_dataframe(df, col_to_stype):
    """Fill NaNs: median for numerical cols, '__missing__' for categorical."""
    df = df.copy()
    for col, stype in col_to_stype.items():
        if col not in df.columns:
            continue
        if stype == 'numerical':
            med = df[col].median()
            df[col] = df[col].fillna(med)
        else:
            df[col] = df[col].fillna('__missing__').astype(str)
    return df


def _build_stype_map(col_to_stype: Dict[str, str], available_cols):
    """Return a torch_frame stype dict restricted to columns present in df."""
    from torch_frame import stype

    mapping = {
        'numerical': stype.numerical,
        'categorical': stype.categorical,
    }
    return {
        col: mapping[st]
        for col, st in col_to_stype.items()
        if col in available_cols and st in mapping
    }


def encode_table(
    df, col_to_stype: Dict[str, str], target_dim: int = TARGET_DIM
) -> Tensor:
    """Encode a single table into a [N, target_dim] float32 tensor.

    Uses a LinearEncoder per stype, concatenated and projected to target_dim.
    """
    from torch_frame import stype
    from torch_frame.data import Dataset
    from torch_frame.nn import (
        EmbeddingEncoder,
        LinearEncoder,
        StypeWiseFeatureEncoder,
    )

    df = _impute_dataframe(df, col_to_stype)
    col_to_stype_tf = _build_stype_map(col_to_stype, df.columns)

    if not col_to_stype_tf:
        logger.warning(
            'No usable columns found in table — returning zeros [%d, %d]',
            len(df),
            target_dim,
        )
        return torch.zeros(len(df), target_dim)

    dataset = Dataset(df=df, col_to_stype=col_to_stype_tf)
    dataset.materialize()
    tf = dataset.tensor_frame

    stype_encoder_dict = {}
    if stype.numerical in tf.col_names_dict:
        stype_encoder_dict[stype.numerical] = LinearEncoder(
            out_channels=target_dim,
            stats_list=[
                dataset.col_stats[col] for col in tf.col_names_dict[stype.numerical]
            ],
            stype=stype.numerical,
        )
    if stype.categorical in tf.col_names_dict:
        stype_encoder_dict[stype.categorical] = EmbeddingEncoder(
            out_channels=target_dim,
            stats_list=[
                dataset.col_stats[col] for col in tf.col_names_dict[stype.categorical]
            ],
            stype=stype.categorical,
        )

    encoder = StypeWiseFeatureEncoder(
        out_channels=target_dim,
        col_stats=dataset.col_stats,
        col_names_dict=tf.col_names_dict,
        stype_encoder_dict=stype_encoder_dict,
    )
    encoder.eval()
    with torch.no_grad():
        out, _ = encoder(tf)  # [N, num_cols, target_dim]

    # Pool across columns dimension → [N, target_dim]
    emb = out.mean(dim=1).float()
    logger.info('Encoded table: shape=%s', tuple(emb.shape))
    return emb


# ---------------------------------------------------------------------------
# Task 2.3 — Concatenate into static_node_x
# ---------------------------------------------------------------------------


def build_static_node_features(
    table_article,
    table_customer,
    target_dim: int = TARGET_DIM,
) -> Tensor:
    """Return static_node_x [N_art + N_cust, target_dim] float32.

    Articles occupy rows 0..N_art-1; customers N_art..N_art+N_cust-1,
    consistent with the ID remapping in data.py.
    """
    article_df = table_article.drop(columns=['article_id'], errors='ignore')
    customer_df = table_customer.drop(columns=['customer_id'], errors='ignore')

    logger.info('Encoding article table (%d rows)…', len(article_df))
    article_emb = encode_table(article_df, ARTICLE_COL_TO_STYPE, target_dim)

    logger.info('Encoding customer table (%d rows)…', len(customer_df))
    customer_emb = encode_table(customer_df, CUSTOMER_COL_TO_STYPE, target_dim)

    # Both are target_dim — no padding required
    static_node_x = torch.cat([article_emb, customer_emb], dim=0)
    logger.info(
        'static_node_x assembled: shape=%s (articles=%d, customers=%d)',
        tuple(static_node_x.shape),
        len(article_emb),
        len(customer_emb),
    )
    return static_node_x
