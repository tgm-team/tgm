# rel-hm TGM Adaptation — Progress

## Phase 1 — Dataset Loading & Task Extraction ✅

### Task 1.1 — Load RelBench dataset and extract raw tables ✅

**File:** `data.py` → `load_raw_tables()`

- Downloads `rel-hm` via `relbench.datasets.get_dataset(download=True)`.
- Returns three DataFrames: `table_article` (~105k rows, 25 cols),
  `table_customer` (~1.37M rows, 7 cols), `table_transactions` (~31.8M rows, 5 cols).
- Logs shapes, dtypes, and null counts for all three tables.

### Task 1.2 — Load `user-item-purchase` task and extract split boundaries ✅

**File:** `data.py` → `load_task_splits()`

- Loads `rel-hm-user-item-purchase` task via `relbench.tasks.get_task()`.
- Extracts `train_table`, `val_table`, `test_table` DataFrames.
- Converts `timestamp_col` to Unix seconds via `_unix_seconds()`, takes `max()`
  of val and test tables to obtain `t_val_unix` and `t_test_unix`.
- These feed directly into `TemporalSplit(val_time, test_time)`.

### Task 1.3 — Remap node IDs to a unified contiguous range ✅

**File:** `data.py` → `remap_node_ids()`

- Articles keep `[0, N_art)`. Customers shift to `[N_art, N_art + N_cust)`.
- Offset = `max(article_id) + 1` (== N_art when 0-indexed).
- Applied in-place to `table_customer`, `table_transactions`, and all three split tables.
- Confirms `N_nodes < 2^31 - 1` (int32-safe).

### Task 1.4 — Prepare transactions as temporal edges ✅

**File:** `data.py` → `build_edge_tensors()`

- Sorts transactions by `t_dat`.
- Converts `t_dat` → Unix seconds (int64) via `_unix_seconds()`.
- Returns `edge_time [E] int64`, `edge_index [E,2] int32`, `edge_x [E,2] float32`
  where `edge_x = [price, sales_channel_id]`.

______________________________________________________________________

## Phase 2 — Node Feature Embedding via pytorch_frame ✅

### Task 2.1 — Define column schema for pytorch_frame ✅

**File:** `embed.py` → `ARTICLE_COL_TO_STYPE`, `CUSTOMER_COL_TO_STYPE`

- Article: 6 numerical + 15 categorical cols. Text cols (`prod_name`, `section_name`,
  `detail_desc`) skipped for simplest case.
- Customer: `age` numerical + 5 categorical cols (including `postal_code`).
- Null handling: median fill for numerical, `"__missing__"` token for categorical.

### Task 2.2 — Build TensorFrame datasets and run embedding ✅

**File:** `embed.py` → `encode_table()`

- Builds `torch_frame.data.Dataset`, calls `.materialize()`.
- Uses `StypeWiseFeatureEncoder` with `LinearEncoder` (numerical) and
  `EmbeddingEncoder` (categorical).
- Pools across the columns dimension → `[N, TARGET_DIM]` float32.
- `TARGET_DIM = 64` (configurable; same for both node types).

### Task 2.3 — Concatenate into `static_node_x` ✅

**File:** `embed.py` → `build_static_node_features()`

- Concatenates `article_emb [N_art, 64]` and `customer_emb [N_cust, 64]`.
- Returns `static_node_x [N_nodes, 64]` float32.
- No zero-padding needed (both encoders output `TARGET_DIM`).

______________________________________________________________________

## Phase 3 — DGData Construction & Splits ✅

### Task 3.1 — Build edge_mask and assemble DGData ✅

**File:** `data.py` → `build_dgdata()`

- Calls `DGData.from_raw()` with `edge_time`, `edge_index`, `edge_x`,
  `static_node_x` (optional), `node_type`, and `time_delta='D'`.
- Attaches `TemporalSplit(val_time=t_val_unix, test_time=t_test_unix)`.

### Task 3.2 — Validate and split ✅

**File:** `data.py` → `build_relbench_hm_data()` (calls `full_data.split()` in caller)

- Returns `(full_data, meta)` where `meta` is `RelHMMetadata` carrying all
  dimension constants and raw split tables.
- Callers run `train_data, val_data, test_data = full_data.split()`.

______________________________________________________________________

## Phase 4 — Evaluation Label Setup ✅

### Task 4.1 — Build positive / negative pairs ✅

**File:** `train.py` → `evaluate()`

- Scores positive destination (`batch.edge_dst`) and one randomly sampled
  negative destination per event.
- Collects `(score, label)` pairs across all batches.

### Task 4.2 — Define evaluation metric ✅

**File:** `train.py` → `compute_ap_ndcg()`, `_ndcg_at_k()`

- Primary metric: **Average Precision (AP)** via `sklearn.metrics.average_precision_score`.
- Secondary metric: **NDCG@10** computed from ranked binary relevance array.
- Reported per epoch for val; once for test.

______________________________________________________________________

## Phase 5 — TGN Training Pipeline ✅

### Task 5.1 — Create DGraphs and hook manager ✅

**File:** `train.py` → `main()`

- Creates `train_dg`, `val_dg`, `test_dg` from split `DGData` objects.
- `NegativeEdgeSamplerHook(low=0, high=N_articles)` — negatives from article pool only.
- `RecencyNeighborHook` shared across all splits.
- `DeduplicationHook` shared.
- `HookManager(keys=['train', 'val', 'test'])` with per-key negative samplers.

### Task 5.2 — Initialize TGN model components ✅

**File:** `train.py`

- `TGNMemory(N_nodes, edge_x_dim=2, memory_dim=100, time_dim=100, IdentityMessage, LastAggregator)`.
- `GraphAttentionEmbedding(in=100, out=100, msg_dim=2)`.
- `LinkPredictor(node_dim=100, hidden_dim=100)`.
- **Option A (default):** memory-only mode; `static_node_x` stored but not consumed.
- **Option B (`--use-static-features`):** wraps encoder in `StaticAugmentedEncoder`
  which projects `[memory || static]` → `memory_dim` before attention.

### Task 5.3 — Training loop ✅

**File:** `train.py` → `train()`

- Standard TGN loop: forward → BCE loss (pos + neg) → `memory.update_state()` →
  `loss.backward()` → `memory.detach()`.
- Matches the pattern in `examples/linkproppred/tgn.py`.

### Task 5.4 — Validation and test evaluation ✅

**File:** `train.py` → `evaluate()`

- Memory state carried over from training (no `reset_state()` between train→val).
- Logs AP and NDCG@10 per epoch for val; final test metrics after last epoch.

______________________________________________________________________

## Phase 6 — Static Feature Integration ✅ (implemented, not yet ablated)

### Task 6.1 — `StaticAugmentedEncoder` ✅

**File:** `train.py` → `StaticAugmentedEncoder`

- Registers `static_node_x` as a buffer.
- On `forward`: fetches `static[unique_nids]`, concatenates with memory state `z`,
  projects back to `memory_dim`, then calls base `GraphAttentionEmbedding`.
- Enabled via `--use-static-features` flag.

### Task 6.2 — Ablation table ✅

| Config            | Flag                                   | Status |
| ----------------- | -------------------------------------- | ------ |
| Baseline          | (default)                              | Ready  |
| + Static (frozen) | `--use-static-features`                | Ready  |
| + Static (joint)  | `--use-static-features --joint-static` | Ready  |

**Implementation:** `--joint-static` passes `trainable_static=True` to
`StaticAugmentedEncoder`, which registers `static_node_x` as `nn.Parameter`
instead of a frozen buffer. Gradients flow through the static embeddings during
back-prop; the existing optimizer already covers `encoder.parameters()`.

### Task 6.3 — Per-customer AP aggregation ✅

**File:** `train.py` → `evaluate()`

- `evaluate()` now returns `(ap, ndcg, mean_per_cust_ap)`.
- Per-customer AP: events are grouped by `edge_src` (customer ID); AP is
  computed independently per customer, then averaged across customers.
- Matches the RelBench eval protocol (one AP value per user, then macro-average).
- Logged as `Val/Test Per-Customer AP` alongside global AP and NDCG@10.

______________________________________________________________________

## Files Created

| File                        | Purpose                         |
| --------------------------- | ------------------------------- |
| `data.py`                   | Phase 1–3: DGData construction  |
| `embed.py`                  | Phase 2: pytorch_frame encoders |
| `train.py`                  | Phase 4–5: training & eval loop |
| `tests/test_relbench_hm.py` | Unit tests for all phases       |
| `progress.md`               | This file                       |

## Tests

Run with:

```bash
pytest examples/linkproppred/relbench/tests/test_relbench_hm.py -v
```

Tests that require `torch_frame` are skipped automatically if it is not installed
(`pytest.importorskip`).

## Next Steps

- [ ] Run a smoke test end-to-end with the actual downloaded dataset.
- [ ] Profile memory usage for the 31.8M-edge transactions table.
