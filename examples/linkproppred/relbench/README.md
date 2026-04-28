# rel-hm → TGM: Link Property Prediction

Adapts the [H&M RelBench dataset](https://relbench.stanford.edu/datasets/rel-hm/)
(`rel-hm`) for TGM's temporal graph pipeline. The target task is
`user-item-purchase` (link prediction between customers and articles),
run through `DGData` / `DGraph` / TGN.

## File Layout

```
examples/linkproppred/relbench/
├── README.md               # this file
├── implementation.md       # detailed phase-by-phase design notes
├── progress.md             # per-task implementation status
├── data.py                 # Phase 1–3: dataset loading, ID remapping, DGData
├── embed.py                # Phase 2: pytorch_frame static node encoders
├── train.py                # Phase 4–6: TGN training & evaluation
└── tests/
    └── test_relbench_hm.py # unit tests (no download required)
```

## Prerequisites

### 1 — Install TGM and its dependencies

From the repo root:

```bash
uv sync                   # core deps (torch, torch-geometric, …)
uv sync --group dev       # adds pytest, ruff, mypy, etc.
```

### 2 — Install example-specific extras

```bash
uv pip install relbench torch-frame scikit-learn tqdm
```

`torch-frame` is only needed for the `--use-static-features` / `--joint-static`
modes. The baseline TGN run works without it.

______________________________________________________________________

## Running the Pipeline

All commands are run from the **repo root**.

### Baseline — TGN memory only

```bash
python -m examples.linkproppred.relbench.train
```

The first run downloads `rel-hm` (~2 GB) via RelBench and caches it locally.

### + Static node features (frozen)

Encodes the article and customer attribute tables via `pytorch_frame` and
prepends the resulting embeddings to the TGN memory state:

```bash
python -m examples.linkproppred.relbench.train --use-static-features
```

### + Static node features (jointly trained)

Makes the static embeddings trainable — gradients flow back through them during
every backward pass:

```bash
python -m examples.linkproppred.relbench.train \
    --use-static-features \
    --joint-static
```

### GPU run

```bash
python -m examples.linkproppred.relbench.train --device cuda
```

### Full example with common overrides

```bash
python -m examples.linkproppred.relbench.train \
    --device cuda        \
    --epochs 10          \
    --bsize 512          \
    --memory-dim 128     \
    --embed-dim 128      \
    --time-dim 128       \
    --n-nbrs 10          \
    --lr 3e-4            \
    --use-static-features \
    --log-file-path run.log
```

______________________________________________________________________

## CLI Reference

| Flag                    | Default | Description                                                         |
| ----------------------- | ------- | ------------------------------------------------------------------- |
| `--device`              | `cpu`   | `cpu` or `cuda`                                                     |
| `--epochs`              | `5`     | Training epochs                                                     |
| `--bsize`               | `200`   | Batch size (events per batch)                                       |
| `--memory-dim`          | `100`   | TGN memory dimension                                                |
| `--embed-dim`           | `100`   | Graph attention output dimension                                    |
| `--time-dim`            | `100`   | Time encoding dimension                                             |
| `--n-nbrs`              | `10`    | Recency neighbours per node per hop                                 |
| `--lr`                  | `1e-4`  | Adam learning rate                                                  |
| `--seed`                | `1337`  | Global random seed                                                  |
| `--use-static-features` | off     | Encode article/customer tables via `pytorch_frame`                  |
| `--joint-static`        | off     | Make static embeddings trainable (requires `--use-static-features`) |
| `--log-file-path`       | None    | Write structured logs to this path in addition to stdout            |

______________________________________________________________________

## Running Tests

Unit tests use **synthetic in-memory data** — no RelBench download required.

```bash
# all tests
source .venv/bin/activate && pytest examples/linkproppred/relbench/tests/test_relbench_hm.py -v

# single test class
source .venv/bin/activate && pytest examples/linkproppred/relbench/tests/test_relbench_hm.py::TestPerCustomerAP -v
```

Tests that require `torch_frame` are skipped automatically if it is not
installed.

______________________________________________________________________

## Metrics

Three metrics are reported at the end of each epoch (val) and after training (test):

| Metric              | Description                                                                                                     |
| ------------------- | --------------------------------------------------------------------------------------------------------------- |
| **AP**              | Global average precision over all scored (positive, negative) pairs                                             |
| **NDCG@10**         | Normalised discounted cumulative gain at rank 10                                                                |
| **Per-Customer AP** | AP computed independently per customer, then macro-averaged — matches the official RelBench evaluation protocol |

______________________________________________________________________

## Architecture

### Graph construction

- **Nodes:** articles `[0, N_art)` + customers `[N_art, N_art + N_cust)` —
  unified contiguous ID space (~1.48 M nodes total).
- **Edges:** transactions table → temporal edges `customer → article` with
  features `[price, sales_channel_id]` (~15.2 M edges).
- **Splits:** `TemporalSplit` at the val / test timestamp boundaries from the
  RelBench task tables.

### Node features (optional)

`embed.py` encodes article and customer attribute tables via `pytorch_frame`:

- Numerical columns → `LinearEncoder`
- Categorical columns → `EmbeddingEncoder`
- Output: `static_node_x [N_nodes, 64]` float32 (same dim for both node types)

### Model

Standard TGN stack:

```
TGNMemory → GraphAttentionEmbedding → LinkPredictor
```

With `--use-static-features`, a `StaticAugmentedEncoder` wrapper fuses
`static_node_x[unique_nids]` with the memory state via a linear projection
before the attention layer. With `--joint-static`, that static tensor is an
`nn.Parameter` updated by the optimizer.

### Ablation configs

| Config            | Flags                                  |
| ----------------- | -------------------------------------- |
| Baseline          | *(none)*                               |
| + Static (frozen) | `--use-static-features`                |
| + Static (joint)  | `--use-static-features --joint-static` |
