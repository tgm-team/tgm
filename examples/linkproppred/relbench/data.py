"""Phase 1-3: Load rel-hm from RelBench, remap node IDs, build DGData.

Usage:
    from examples.linkproppred.relbench.data import build_relbench_hm_data
    full_data, meta = build_relbench_hm_data()
    train_data, val_data, test_data = full_data.split()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from tgm.data import DGData
from tgm.data.split import TemporalSplit

logger = logging.getLogger(__name__)

DATASET_NAME = 'rel-hm'
TASK_NAME = 'user-item-purchase'
TIME_DELTA = 'D'  # daily granularity matches H&M purchase data


@dataclass
class RelHMMetadata:
    """Auxiliary metadata produced alongside the DGData object."""

    n_articles: int
    n_customers: int
    n_nodes: int
    n_edges: int
    article_id_offset: int  # 0 (articles keep their original IDs)
    customer_id_offset: int  # == n_articles
    edge_x_dim: int  # 2: [price, sales_channel_id]
    t_val_unix: int  # Unix-second boundary separating train / val
    t_test_unix: int  # Unix-second boundary separating val / test
    train_table: object  # raw pandas DataFrame (with shifted customer_id)
    val_table: object
    test_table: object


def _unix_seconds(ts_series) -> np.ndarray:
    """Convert a pandas datetime64 or int64 Series to Unix seconds (int64)."""
    arr = ts_series.to_numpy()
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype('datetime64[s]').astype(np.int64)
    # Already numeric (nanoseconds) – divide down to seconds
    return arr.astype(np.int64) // 10**9


# ---------------------------------------------------------------------------
# Task 1.1 — Load raw tables
# ---------------------------------------------------------------------------


def load_raw_tables(dataset_name: str = DATASET_NAME):
    """Download (if needed) and return the three H&M tables as DataFrames."""
    from relbench.datasets import get_dataset

    dataset = get_dataset(name=dataset_name, download=True)
    db = dataset.get_db()

    table_article = db.table_dict['article'].df.copy()
    table_customer = db.table_dict['customer'].df.copy()
    table_transactions = db.table_dict['transactions'].df.copy()

    logger.info(
        'Loaded tables — article: %s, customer: %s, transactions: %s',
        table_article.shape,
        table_customer.shape,
        table_transactions.shape,
    )
    logger.info('Article dtypes:\n%s', table_article.dtypes)
    logger.info('Customer dtypes:\n%s', table_customer.dtypes)
    logger.info('Transactions dtypes:\n%s', table_transactions.dtypes)
    logger.info('Article null counts:\n%s', table_article.isnull().sum())
    logger.info('Customer null counts:\n%s', table_customer.isnull().sum())

    return dataset, table_article, table_customer, table_transactions


# ---------------------------------------------------------------------------
# Task 1.2 — Load task and extract split boundaries
# ---------------------------------------------------------------------------


def load_task_splits(dataset, task_name: str = TASK_NAME):
    """Return (task, train_table, val_table, test_table) from RelBench task.

    relbench 2.x API:
      - get_task(dataset_name, task_name) with the short task name
      - split tables accessed via task.get_table('train'/'val'/'test')
      - timestamp column identified by task.time_col
    """
    from relbench.tasks import get_task

    task = get_task(DATASET_NAME, task_name, download=True)

    train_table = task.get_table('train').df.copy()
    val_table = task.get_table('val').df.copy()
    test_table = task.get_table('test').df.copy()

    ts_col = task.time_col  # 'timestamp' in relbench 2.x

    t_val_end = _unix_seconds(val_table[ts_col]).max()
    t_test_end = _unix_seconds(test_table[ts_col]).max()

    logger.info('Split boundaries — val_end=%d, test_end=%d', t_val_end, t_test_end)
    return task, train_table, val_table, test_table, int(t_val_end), int(t_test_end)


# ---------------------------------------------------------------------------
# Task 1.3 — Remap node IDs to a single contiguous space
# ---------------------------------------------------------------------------


def remap_node_ids(
    table_article,
    table_customer,
    table_transactions,
    train_table,
    val_table,
    test_table,
):
    """Shift customer IDs so articles occupy [0, N_art) and customers [N_art, N_art+N_cust).

    Modifies all DataFrames in-place and returns the offset value.
    """
    offset = int(table_article['article_id'].max()) + 1

    table_customer['customer_id'] = table_customer['customer_id'] + offset
    table_transactions['customer_id'] = table_transactions['customer_id'] + offset

    for tbl in [train_table, val_table, test_table]:
        tbl['customer_id'] = tbl['customer_id'] + offset

    n_nodes = offset + len(table_customer)
    assert n_nodes < 2**31 - 1, 'Node count overflows int32'
    logger.info(
        'Node ID remapping done — N_art=%d, N_cust=%d, N_nodes=%d',
        offset,
        len(table_customer),
        n_nodes,
    )
    return offset


# ---------------------------------------------------------------------------
# Task 1.4 — Build edge tensors from the transactions table
# ---------------------------------------------------------------------------


def build_edge_tensors(table_transactions) -> Tuple[Tensor, Tensor, Tensor]:
    """Return (edge_time [E], edge_index [E,2], edge_x [E,2])."""
    table_transactions = table_transactions.sort_values('t_dat').reset_index(drop=True)

    ts_sec = _unix_seconds(table_transactions['t_dat'])
    edge_time = torch.from_numpy(ts_sec)  # int64

    src_dst = table_transactions[['customer_id', 'article_id']].to_numpy(dtype=np.int32)
    edge_index = torch.from_numpy(src_dst)  # [E, 2] int32

    edge_feat = table_transactions[['price', 'sales_channel_id']].to_numpy(
        dtype=np.float32
    )
    edge_x = torch.from_numpy(edge_feat)  # [E, 2] float32

    assert (edge_time >= 0).all(), 'Negative timestamps found'
    logger.info(
        'Edge tensors — E=%d, edge_x_dim=%d, time range [%d, %d]',
        edge_time.shape[0],
        edge_x.shape[1],
        edge_time.min(),
        edge_time.max(),
    )
    return edge_time, edge_index, edge_x


# ---------------------------------------------------------------------------
# Task 3.1 + 3.2 — Assemble DGData with TemporalSplit
# ---------------------------------------------------------------------------


def build_dgdata(
    edge_time: Tensor,
    edge_index: Tensor,
    edge_x: Tensor,
    static_node_x: Tensor | None,
    node_type: Tensor,
    t_val_unix: int,
    t_test_unix: int,
) -> DGData:
    """Wrap tensors into a DGData ready to call .split()."""
    split_strategy = TemporalSplit(val_time=t_val_unix, test_time=t_test_unix)

    data = DGData.from_raw(
        time_delta=TIME_DELTA,
        edge_time=edge_time,
        edge_index=edge_index,
        edge_x=edge_x,
        static_node_x=static_node_x,
        node_type=node_type,
    )
    # Attach split strategy after construction (DGData is a dataclass)
    data._split_strategy = split_strategy
    return data


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_relbench_hm_data(
    static_node_x: Tensor | None = None,
    dataset_name: str = DATASET_NAME,
    task_name: str = TASK_NAME,
) -> Tuple[DGData, RelHMMetadata]:
    """Build a fully prepared DGData for the rel-hm user-item-purchase task.

    Args:
        static_node_x: Pre-computed static node features [N_nodes, D].
                        If None, DGData.static_node_x will be None (TGN memory-only mode).
        dataset_name: RelBench dataset identifier.
        task_name: RelBench task identifier.

    Returns:
        (full_data, meta) where meta carries split tables and dimension constants.
    """
    dataset, table_article, table_customer, table_transactions = load_raw_tables(
        dataset_name
    )
    task, train_table, val_table, test_table, t_val_unix, t_test_unix = (
        load_task_splits(dataset, task_name)
    )

    customer_id_offset = remap_node_ids(
        table_article,
        table_customer,
        table_transactions,
        train_table,
        val_table,
        test_table,
    )

    edge_time, edge_index, edge_x = build_edge_tensors(table_transactions)

    n_articles = len(table_article)
    n_customers = len(table_customer)
    n_nodes = n_articles + n_customers

    node_type = torch.cat(
        [
            torch.zeros(n_articles, dtype=torch.int32),
            torch.ones(n_customers, dtype=torch.int32),
        ]
    )

    full_data = build_dgdata(
        edge_time,
        edge_index,
        edge_x,
        static_node_x,
        node_type,
        t_val_unix,
        t_test_unix,
    )

    meta = RelHMMetadata(
        n_articles=n_articles,
        n_customers=n_customers,
        n_nodes=n_nodes,
        n_edges=edge_time.shape[0],
        article_id_offset=0,
        customer_id_offset=customer_id_offset,
        edge_x_dim=edge_x.shape[1],
        t_val_unix=t_val_unix,
        t_test_unix=t_test_unix,
        train_table=train_table,
        val_table=val_table,
        test_table=test_table,
    )

    logger.info(
        'DGData assembled — nodes=%d, edges=%d, static_node_x=%s',
        n_nodes,
        edge_time.shape[0],
        'None' if static_node_x is None else str(tuple(static_node_x.shape)),
    )
    return full_data, meta
