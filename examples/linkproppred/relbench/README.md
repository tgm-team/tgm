# rel-hm → TGM: Link Property Prediction

Adapts the [H&M RelBench dataset](https://relbench.stanford.edu/datasets/rel-hm/) (`rel-hm`) for
TGM's temporal graph pipeline. The target task is `rel-hm-user-item-purchase` (link prediction
between customers and articles), run through `DGData` / `DGraph` / TGN.

## File Layout

```
examples/linkproppred/relbench/
├── README.md               # this file
├── implementation.md       # detailed phase-by-phase design notes
├── progress.md             # per-task implementation status
├── rel-hm.py               # original data exploration scaffold
├── data.py                 # Phase 1–3: dataset loading, ID remapping, DGData
├── embed.py                # Phase 2: pytorch_frame static node encoders
├── train.py                # Phase 4–5: TGN training & AP / NDCG@10 evaluation
└── tests/
    └── test_relbench_hm.py # unit tests (no download required)
```

## Installation

```bash
pip install relbench torch-frame scikit-learn tqdm
```

## Running

### Baseline (TGN memory only)

```bash
python -m examples.linkproppred.relbench.train
```

### With static node features (pytorch_frame)

```bash
python -m examples.linkproppred.relbench.train --use-static-features
```

### Common options

| Flag                    | Default | Description                                      |
| ----------------------- | ------- | ------------------------------------------------ |
| `--device`              | `cpu`   | `cpu` or `cuda`                                  |
| `--epochs`              | `5`     | Number of training epochs                        |
| `--bsize`               | `200`   | Batch size                                       |
| `--memory-dim`          | `100`   | TGN memory dimension                             |
| `--embed-dim`           | `100`   | Graph attention output dimension                 |
| `--time-dim`            | `100`   | Time encoding dimension                          |
| `--n-nbrs`              | `10`    | Neighbours per node per hop                      |
| `--lr`                  | `1e-4`  | Learning rate                                    |
| `--use-static-features` | off     | Encode article/customer tables via pytorch_frame |

## Running Tests

No dataset download required — tests use synthetic in-memory data.

```bash
pytest examples/linkproppred/relbench/tests/test_relbench_hm.py -v
```

Tests that require `torch_frame` are skipped automatically if it is not installed.

## Architecture

### Graph construction

- **Nodes:** articles `[0, N_art)` + customers `[N_art, N_art + N_cust)` — unified contiguous ID space.
- **Edges:** transactions table → temporal edges `customer → article` with features `[price, sales_channel_id]`.
- **Splits:** `TemporalSplit` at the val / test timestamp boundaries from the RelBench task tables.

### Node features (Phase 2, optional)

`embed.py` encodes article and customer tables via `pytorch_frame`:

- Numerical columns: `LinearEncoder`.
- Categorical columns: `EmbeddingEncoder`.
- Output: `static_node_x [N_nodes, 64]` float32 (same dim for both node types — no padding).

### Model (Phase 5)

Standard TGN stack from `tgm`:

```
TGNMemory → GraphAttentionEmbedding → LinkPredictor
```

With `--use-static-features`, a `StaticAugmentedEncoder` wrapper fuses
`static_node_x[unique_nids]` with the memory state before the attention layer.

### Evaluation

- **Average Precision (AP)** — primary metric (RelBench standard).
- **NDCG@10** — secondary metric.
- Negative destinations are sampled from the article pool `[0, N_art)` only.
