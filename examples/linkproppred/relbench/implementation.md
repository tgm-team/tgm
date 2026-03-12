# rel-hm → TGM Adaptation: Implementation Plan

Adapting the [H&M RelBench dataset](https://relbench.stanford.edu/datasets/rel-hm/) (`rel-hm`) for
temporal graph models (TGM). The target task is `user-item-purchase` (link prediction between
customers and articles), run through TGM's `DGData` / `DGraph` / TGN pipeline.

______________________________________________________________________

## Dataset Overview

| Table          | Rows   | Role                            |
| -------------- | ------ | ------------------------------- |
| `article`      | ~105k  | Node (product), static features |
| `customer`     | ~1.37M | Node (user), static features    |
| `transactions` | ~31.8M | Temporal edges (purchases)      |

**Schema** (maintained in `rel-hm.py` as `schema_hm`):

- `article`: numerical IDs + categorical product attributes + text (`prod_name`, `detail_desc`)
- `customer`: numerical `age` + categorical membership/activity flags
- `transactions`: `t_dat` (timestamp), `price`, `sales_channel_id`, `customer_id`, `article_id`

**Node ID scheme**: articles keep `0 .. N_art - 1`; customers shift to `N_art .. N_art + N_cust - 1`.

______________________________________________________________________

## Phase 1 — Dataset Loading & Task Extraction

### Task 1.1 — Load RelBench dataset and extract raw tables

```python
from relbench.datasets import get_dataset

dataset = get_dataset(name='rel-hm', download=True)
db = dataset.get_db()

table_article       = db.table_dict['article'].df        # [N_art, 25]
table_customer      = db.table_dict['customer'].df       # [N_cust, 7]
table_transactions  = db.table_dict['transactions'].df   # [E, 5]
```

- Inspect shapes, dtypes, and null counts for all three tables; log summaries.
- Note: `article_id` in the article table is already 0-indexed. `customer_id` in the customer
  table is 0-indexed separately.

### Task 1.2 — Load the `user-item-purchase` task and extract split boundaries

```python
from relbench.tasks import get_task

task = get_task(dataset, 'rel-hm-user-item-purchase')

train_table = task.train_table   # DataFrame: customer_id, article_id, timestamp
val_table   = task.val_table
test_table  = task.test_table

t_train_end = train_table[task.timestamp_col].max()
t_val_end   = val_table[task.timestamp_col].max()
t_test_end  = test_table[task.timestamp_col].max()
```

- These three timestamp boundaries define the `TemporalSplit` for `DGData`.
- Train graph context: all transactions with `t ≤ t_train_end`.
- Val graph: train + `t_train_end < t ≤ t_val_end` (sliding window, TGB convention).
- Test graph: val + `t_val_end < t ≤ t_test_end`.
- Positive/negative label pairs come from the task tables, not from the graph itself.

### Task 1.3 — Remap node IDs to a unified contiguous range

```python
import numpy as np

N_art  = len(table_article)
N_cust = len(table_customer)
offset = table_article['article_id'].max() + 1   # == N_art if 0-indexed

# Remap in-place
table_customer['customer_id']       += offset
table_transactions['customer_id']   += offset
# Also remap task tables
for tbl in [train_table, val_table, test_table]:
    tbl['customer_id'] += offset
```

- Final node count: `N_nodes = N_art + N_cust`
- Confirm `N_nodes - 1 < 2**31 - 1` (int32 safe).

### Task 1.4 — Prepare transactions as temporal edges

```python
import torch

# Sort by time (relbench may already guarantee this)
table_transactions = table_transactions.sort_values('t_dat').reset_index(drop=True)

# Convert datetime → Unix seconds (int64)
ts_seconds = table_transactions['t_dat'].astype(np.int64) // 10**9

edge_time  = torch.from_numpy(ts_seconds.to_numpy())          # [E] int64
edge_index = torch.from_numpy(
    table_transactions[['customer_id', 'article_id']].to_numpy()
).int()                                                        # [E, 2] int32

# Edge features: price + sales_channel_id
edge_x = torch.from_numpy(
    table_transactions[['price', 'sales_channel_id']].to_numpy()
).float()                                                      # [E, 2] float32
```

- Validate: all `edge_time >= 0`, all values `< 2**31 - 1`.
- H&M's transaction dates (~2018–2020) in Unix seconds are well within int32 range.

______________________________________________________________________

## Phase 2 — Node Feature Embedding via pytorch_frame

### Task 2.1 — Define column schema for pytorch_frame

**Article table** (drop `article_id`, treat the rest):

| Column                                                                                                                                                                                                                                                                   | ColType                               |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------- |
| `product_code`, `product_type_no`, `department_no`, `section_no`, `garment_group_no`, `perceived_colour_master_id`                                                                                                                                                       | `stype.numerical`                     |
| `product_type_name`, `product_group_name`, `graphical_appearance_no/name`, `colour_group_code/name`, `perceived_colour_value_id/name`, `perceived_colour_master_name`, `department_name`, `index_code/name`, `index_group_no/name`, `section_name`, `garment_group_name` | `stype.categorical`                   |
| `prod_name`, `detail_desc`                                                                                                                                                                                                                                               | `stype.text` (skip for simplest case) |

**Customer table** (drop `customer_id`):

| Column                                                                        | ColType             |
| ----------------------------------------------------------------------------- | ------------------- |
| `age`                                                                         | `stype.numerical`   |
| `FN`, `Active`, `club_member_status`, `fashion_news_frequency`, `postal_code` | `stype.categorical` |

**Null handling**:

- `age`: fill NaN with median before embedding.
- Categorical NaNs: treat as a dedicated `"__missing__"` token.
- `postal_code` has high cardinality (~352k unique); cap embeddings at dim 16 or hash bucket.

### Task 2.2 — Build TensorFrame datasets and run embedding

```python
import torch_frame
from torch_frame import stype
from torch_frame.data import Dataset
from torch_frame.nn import StypeWiseFeatureEncoder, ResNet

TARGET_DIM = 64   # unified output dim for both node types

# --- Article ---
col_stats_article = {...}   # computed from table_article
dataset_article = Dataset(
    df=table_article.drop(columns=['article_id']),
    col_to_stype=article_col_to_stype,
    col_stats=col_stats_article,
)
dataset_article.materialize()
tf_article = dataset_article.tensor_frame   # TensorFrame

encoder_article = StypeWiseFeatureEncoder(
    out_channels=TARGET_DIM,
    col_stats=col_stats_article,
    col_names_dict=tf_article.col_names_dict,
    stype_encoder_dict={
        stype.numerical: ...,
        stype.categorical: ...,
    }
)
article_emb = encoder_article(tf_article)   # [N_art, TARGET_DIM]

# --- Customer (same pattern) ---
customer_emb = encoder_customer(tf_customer)  # [N_cust, TARGET_DIM]
```

- Use `TARGET_DIM = 64` (configurable); same dim for both tables avoids padding.
- Simplest encoder: `LinearEncoder` per stype, then sum/concat → linear projection to `TARGET_DIM`.
- These encoders are pretrained (or jointly trained); for simplest case, use random or
  identity initialization and let TGN memory carry the learning.

### Task 2.3 — Concatenate into `static_node_x`

```python
# Ensure float32
article_emb  = article_emb.detach().float()    # [N_art,  TARGET_DIM]
customer_emb = customer_emb.detach().float()   # [N_cust, TARGET_DIM]

static_node_x = torch.cat([article_emb, customer_emb], dim=0)  # [N_nodes, TARGET_DIM]
node_type     = torch.cat([
    torch.zeros(N_art,  dtype=torch.int32),
    torch.ones(N_cust,  dtype=torch.int32),
])                                                               # [N_nodes]
```

- TGM's `DGData.static_node_x` expects `[num_nodes, D_static]` with a single unified `D_static`.
  The fixed `TARGET_DIM` output from both encoders satisfies this constraint.
- If article and customer encoders produce different dims, zero-pad the smaller one on the right.

______________________________________________________________________

## Phase 3 — DGData Construction & Splits

### Task 3.1 — Build edge_mask and assemble DGData

```python
from tgm import TimeDeltaDG
from tgm.data import DGData
from tgm.data.split import TemporalSplit

E = edge_index.shape[0]
edge_mask = torch.ones(E, dtype=torch.int32)   # all events are edge events

# Map split timestamps to event indices (binary search on sorted edge_time)
i_train = torch.searchsorted(edge_time, torch.tensor(t_train_end_unix)).item()
i_val   = torch.searchsorted(edge_time, torch.tensor(t_val_end_unix)).item()

split_strategy = TemporalSplit(
    train_end=i_train,
    val_end=i_val,
)

full_data = DGData(
    time_delta=TimeDeltaDG('D'),   # daily granularity
    time=edge_time,                # [E] int64
    edge_mask=edge_mask,           # [E] int32 all-ones
    edge_index=edge_index,         # [E, 2] int32
    edge_x=edge_x,                 # [E, 2] float32
    static_node_x=static_node_x,  # [N_nodes, 64] float32
    node_type=node_type,           # [N_nodes] int32
    _split_strategy=split_strategy,
)
```

### Task 3.2 — Validate and split

```python
train_data, val_data, test_data = full_data.split()

print(f'Nodes:      {full_data.num_nodes}')
print(f'Edges:      {full_data.num_edges}')
print(f'edge_x dim: {full_data.edge_x.shape[1]}')   # should be 2
print(f'Train edges: {train_data.num_edges}')
print(f'Val edges:   {val_data.num_edges}')
print(f'Test edges:  {test_data.num_edges}')
```

- Fallback: if `TemporalSplit` API doesn't accept raw indices, use `TemporalRatioSplit(0.70, 0.15, 0.15)`.

______________________________________________________________________

## Phase 4 — Evaluation Label Setup

### Task 4.1 — Build positive / negative pairs for val and test

```python
# val_table columns: customer_id (already shifted), article_id, timestamp, label
val_pos_mask   = val_table['label'] == 1
val_pos_pairs  = val_table[val_pos_mask][['customer_id', 'article_id']].values  # [P, 2]
val_neg_cands  = task.val_neg_table   # per-query negative article candidates (if available)
```

- RelBench's `user-item-purchase` task pre-provides negative candidates per query entity.
  Check `task` attributes: `task.val_neg_table` or `task.test_neg_table`.
- If not pre-provided, sample uniformly from `0 .. N_art - 1` (only articles are valid
  negative destinations since all edges are customer→article).

### Task 4.2 — Define evaluation metric

- Primary metric: **Average Precision (AP)** — RelBench's standard for this task.
- Secondary metric: **NDCG@10**.
- Use `relbench.metrics.average_precision` and `relbench.metrics.ndcg` if available, or
  compute manually from ranked scores.
- Adapt the TGM eval loop from `examples/linkproppred/tgn.py`: replace the MRR block with
  AP computation per `(src, pos_dst, neg_dst_list)` triplet.

______________________________________________________________________

## Phase 5 — TGN Training Pipeline

### Task 5.1 — Create DGraphs and hook manager

```python
from tgm import DGraph
from tgm.data import DGDataLoader
from tgm.hooks import (
    DeduplicationHook,
    NegativeEdgeSamplerHook,
    RecencyNeighborHook,
)

device = 'cuda'  # or 'cpu'

train_dg = DGraph(train_data, device=device)
val_dg   = DGraph(val_data,   device=device)
test_dg  = DGraph(test_data,  device=device)

nbr_hook = RecencyNeighborHook(
    num_nbrs=[10],
    num_nodes=full_data.num_nodes,
    seed_nodes_keys=['edge_src', 'edge_dst', 'neg'],
    seed_times_keys=['edge_time', 'edge_time', 'neg_time'],
)

# Negative sampler: draw negatives only from article node pool [0, N_art - 1]
neg_hook = NegativeEdgeSamplerHook(
    num_nodes=N_art,    # restrict to article IDs as negative destinations
    seed_nodes_key='edge_src',
)
```

- `RecencyNeighborHook` handles recency-weighted neighbor sampling — no changes needed.
- The `NegativeEdgeSamplerHook` should sample from `[0, N_art - 1]` since all ground-truth
  edges are `customer → article`; sampling a customer as a negative destination is invalid.

### Task 5.2 — Initialize TGN model components

```python
from tgm.nn import LinkPredictor, TGNMemory
from tgm.nn.encoder.tgn import GraphAttentionEmbedding, IdentityMessage, LastAggregator

EDGE_X_DIM  = 2     # price + sales_channel_id
MEMORY_DIM  = 100
EMBED_DIM   = 100
TIME_DIM    = 100

memory = TGNMemory(
    full_data.num_nodes,
    EDGE_X_DIM,
    MEMORY_DIM,
    TIME_DIM,
    message_module=IdentityMessage(EDGE_X_DIM, MEMORY_DIM, TIME_DIM),
    aggregator_module=LastAggregator(),
).to(device)

encoder = GraphAttentionEmbedding(
    in_channels=MEMORY_DIM,
    out_channels=EMBED_DIM,
    msg_dim=EDGE_X_DIM,
    time_enc=memory.time_enc,
).to(device)

decoder = LinkPredictor(node_dim=EMBED_DIM, hidden_dim=EMBED_DIM).to(device)
```

- **Phase 5 (Option A)**: use TGN memory only; `static_node_x` stored in `DGData` but not
  yet consumed by the model. This validates the pipeline before adding static features.
- Phase 6 (Option B) integrates `static_node_x` into the encoder (see Phase 6).

### Task 5.3 — Training loop

```python
opt = torch.optim.Adam(
    set(memory.parameters()) | set(encoder.parameters()) | set(decoder.parameters()),
    lr=1e-4,
)

for epoch in range(1, N_EPOCHS + 1):
    memory.train(); encoder.train(); decoder.train()
    memory.reset_state()
    total_loss = 0.0

    for batch in tqdm(train_loader):
        opt.zero_grad()

        # Neighborhood aggregation (same as tgn.py)
        nbr_nodes    = batch.nbr_nids[0].flatten()
        nbr_mask     = nbr_nodes != PADDED_NODE_ID
        num_nbrs     = len(nbr_nodes) // (len(batch.edge_src) + len(batch.edge_dst) + len(batch.neg))
        src_nodes    = torch.cat([
            batch.edge_src.repeat_interleave(num_nbrs),
            batch.edge_dst.repeat_interleave(num_nbrs),
            batch.neg.repeat_interleave(num_nbrs),
        ])
        nbr_edge_idx = torch.stack([
            batch.global_to_local(src_nodes[nbr_mask]),
            batch.global_to_local(nbr_nodes[nbr_mask]),
        ]).long()
        nbr_edge_time = batch.nbr_edge_time[0].flatten()[nbr_mask]
        nbr_edge_x    = batch.nbr_edge_x[0].flatten(0, -2).float()[nbr_mask]

        z, last_update = memory(batch.unique_nids)
        z = encoder(z, last_update, nbr_edge_idx, nbr_edge_time, nbr_edge_x)

        inv_src = batch.global_to_local(batch.edge_src)
        inv_dst = batch.global_to_local(batch.edge_dst)
        inv_neg = batch.global_to_local(batch.neg)
        pos_out = decoder(z[inv_src], z[inv_dst])
        neg_out = decoder(z[inv_src], z[inv_neg])

        loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        loss += F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))

        memory.update_state(batch.edge_src, batch.edge_dst, batch.edge_time, batch.edge_x.float())
        loss.backward()
        opt.step()
        memory.detach()
        total_loss += float(loss)
```

### Task 5.4 — Validation and test evaluation

- Iterate over `val_table` grouped by query entity (customer).
- For each `(customer_id, [pos_article_id], [neg_article_ids])`:
  - Score all candidate articles using `decoder(z[customer], z[candidates]).sigmoid()`.
  - Compute AP and NDCG@10 over the ranked list.
- Carry over memory state from training into val (standard TGN protocol; no `reset_state()`
  between train and val).
- Log `AP` and `NDCG@10` per epoch.

______________________________________________________________________

## Phase 6 — Static Feature Integration

### Task 6.1 — Integrate `static_node_x` into the encoder

```python
class StaticAugmentedEncoder(nn.Module):
    """Wraps GraphAttentionEmbedding, prepending static node features to memory."""

    def __init__(self, base_encoder, static_node_x, static_dim, memory_dim):
        super().__init__()
        self.base_encoder = base_encoder
        self.register_buffer('static_node_x', static_node_x)  # [N_nodes, static_dim]
        self.proj = nn.Linear(memory_dim + static_dim, memory_dim)

    def forward(self, z, last_update, nbr_edge_index, nbr_edge_time, nbr_edge_x, unique_nids):
        static = self.static_node_x[unique_nids]         # [B, static_dim]
        z_aug  = self.proj(torch.cat([z, static], dim=-1))  # [B, memory_dim]
        return self.base_encoder(z_aug, last_update, nbr_edge_index, nbr_edge_time, nbr_edge_x)
```

- Adjust `in_channels` of `GraphAttentionEmbedding` if fusing before attention (alternative
  approach to the projection wrapper above).
- The pytorch_frame encoders from Phase 2 can optionally be made jointly trainable here
  (attach them to the optimizer) rather than precomputed and frozen.

### Task 6.2 — Ablation

| Config           | Static features         | Expected AP |
| ---------------- | ----------------------- | ----------- |
| Baseline         | No                      | —           |
| + Static         | Yes (frozen embeddings) | —           |
| + Static (joint) | Yes (jointly trained)   | —           |

______________________________________________________________________

## Key Constants & Decisions

| Parameter      | Value                | Notes                                                 |
| -------------- | -------------------- | ----------------------------------------------------- |
| `TARGET_DIM`   | 64                   | Unified static feature dim for articles and customers |
| `EDGE_X_DIM`   | 2                    | `[price, sales_channel_id]`                           |
| `MEMORY_DIM`   | 100                  | TGN memory state dimension                            |
| `EMBED_DIM`    | 100                  | Output of graph attention encoder                     |
| `TIME_DIM`     | 100                  | Time encoding dimension                               |
| `num_nbrs`     | `[10]`               | 1-hop, 10 neighbors per node                          |
| `time_delta`   | `'D'` (daily)        | Matches H&M's purchase granularity                    |
| Edge direction | `customer → article` | Negatives sampled from article pool only              |
| Timestamp unit | Unix seconds         | H&M (~2018–2020) safely within int32 range            |
| Text columns   | Skipped initially    | `prod_name`, `section_name`, `detail_desc`            |
| Eval metric    | AP, NDCG@10          | RelBench primary metrics for this task                |

## File Layout

```
examples/linkproppred/relbench/
├── implementation.md       # this file
├── rel-hm.py               # data loading scaffold (existing)
├── data.py                 # Phase 1–3: DGData construction (to be created)
├── embed.py                # Phase 2: pytorch_frame encoders (to be created)
└── train.py                # Phase 4–5: training & evaluation loop (to be created)
```
