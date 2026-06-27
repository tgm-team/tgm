# AGENTS.md

Orientation for coding agents working in TGM (`tgm-lib`), a research library for
ML on temporal graphs. This file covers the things that are easy to get wrong or
that you'd otherwise re-derive every session. For API details, read `docs/` and
the docstrings (every public class/method is documented inline) — don't duplicate
them here.

## Environment & commands

Everything runs through [`uv`](https://docs.astral.sh/uv/) against the `uv.lock`
lockfile. Prepend `uv run` to Python/pytest invocations, or activate `.venv`.

| Task                                | Command                                                                                    |
| ----------------------------------- | ------------------------------------------------------------------------------------------ |
| Install core deps                   | `uv sync`                                                                                  |
| Install dev deps (needed for tests) | `uv sync --group dev`                                                                      |
| Install example deps                | `pip install -e .[examples]` (adds `py-tgb`, `tgb-seq`, `torchmetrics`, …)                 |
| Pre-commit hooks                    | `uv run pre-commit install`                                                                |
| Unit tests                          | `./scripts/run_unit_tests.sh` (CPU-only; pass `--gpu` to include `gpu`-marked tests)       |
| Performance/benchmarks              | `./scripts/run_perf_tests.sh [--small\|--medium\|--large] [--gpu]` (defaults to `--small`) |
| Run an example                      | `python examples/linkproppred/tgat.py --dataset tgbl-wiki --device cuda`                   |
| Build docs                          | `./scripts/build_docs.sh`                                                                  |
| Add/remove a dependency             | `uv add <pkg>` / `uv add --group <group> <pkg>` (commits `pyproject.toml` + `uv.lock`)     |

- Python **3.10** required. `ruff` (pinned `0.11.1`), `isort`, and `mypy` (strict:
  `disallow_untyped_defs`) all run in pre-commit and CI — type every function.
- Ruff format uses **single quotes**; line length 88. Docstrings follow Google
  convention (pydocstyle `D`).
- **Integration tests** (`test/integration/`) are *not* runnable locally — they
  dispatch SLURM jobs on the TGM CI cluster (note the `@pytest.mark.slurm`
  resource requests and `slurm_job_runner` fixture) and require an auth key.
  Don't try to run them; run the corresponding `examples/` script locally instead
  to validate a change end-to-end.
- Examples default to `--device cpu`. Link-pred examples default to `tgbl-wiki`,
  node-pred to `tgbn-trade`. Datasets download on first use via `py-tgb`.

## The core mental model: CTDG vs DTDG

TGM treats **continuous-time (CTDG)** and **discrete-time (DTDG)** graphs through
a *single* abstraction. There is no separate CTDG/DTDG class — the distinction is
entirely a matter of (a) the graph's `TimeDeltaDG` granularity and (b) how you
configure the `DGDataLoader`. Understand this before touching data/loader code.

`TimeDeltaDG` (`tgm/core/timedelta.py`) is the linchpin:

- **Event-ordered** (`unit='r'`, the special unit; only `value=1` allowed): a
  CTDG indexed by event position, no wall-clock meaning. Cannot be discretized or
  compared to time units (raises `EventOrderedConversionError`).
- **Time-ordered** (`'s'`, `'m'`, `'h'`, `'D'`, `'W'`, `'M'`, `'Y'`, `ms/us/ns`):
  events carry real timestamps at this resolution.

How the two regimes are iterated, both via `DGDataLoader(dg, batch_size, batch_unit=...)`:

- **CTDG / event batching**: `batch_unit='r'` (default). Slices by event index.
- **DTDG / time batching**: `batch_unit` is a time unit (e.g. `'Y'`). Slices by
  timestamp into snapshots. **Hard rule:** `batch_unit` must not be *coarser* than
  the graph's own `time_delta`, or the loader raises `InvalidDiscretizationError`.
  You cannot iterate an event-ordered graph with a time-ordered `batch_unit`
  (raises `EventOrderedConversionError`).

To turn a fine CTDG into coarser DTDG snapshots, call
`DGData.discretize(time_delta, reduce_op='first')` *before* building the `DGraph`,
then iterate the discretized graph with a matching `batch_unit`. Discretization
only ever goes coarser. See `examples/linkproppred/gclstm.py` for the canonical
DTDG pattern: it keeps a fine-grained event loader *and* a separate snapshot
loader, advancing the snapshot when event time crosses a snapshot boundary.

Per-dataset default granularities live in `TGB_TIME_DELTAS` / `TGB_SEQ_TIME_DELTAS`
(e.g. `tgbn-trade` is `'Y'`, most `tgbl-*` are `'s'`). `DGData.from_tgb` sets these
automatically.

## Module layout (the part agents get wrong)

There is **no `tgm/models/` directory**. Models live under `tgm/nn/`, split three ways:

- **`tgm/nn/encoder/`** — temporal encoders that produce node embeddings (TGN,
  TGAT graph attention, DyGFormer, TPNet, TGCN, GCLSTM, CTAN, ROLAND).
- **`tgm/nn/decoder/`** — task heads: `LinkPredictor`, `NodePredictor`,
  `GraphPredictor`, `NCNPredictor`.
- **`tgm/nn/modules/`** — reusable building blocks (`Time2Vec`, `TemporalAttention`,
  `EdgeBankPredictor`, `MLPMixer`, `PopTrackPredictor`, `tCoMemPredictor`).

Everything in `tgm/nn/` is a plain `torch.nn.Module` — there is no TGM-specific
model base class. Public symbols are re-exported from `tgm/nn/__init__.py`.

The other layers:

- **`tgm/core/`** — `DGraph` (immutable, slice-able view over `DGStorage`),
  `DGBatch` (the materialized batch dataclass), `TimeDeltaDG`. `DGraph` and
  `DGBatch` are also re-exported from the top-level `tgm` package.
- **`tgm/data/`** — `DGData` (loads/holds raw graph data; `from_tgb`, `from_csv`,
  `from_pandas`, `from_raw`, `from_tgb_seq`, plus `split()` and `discretize()`)
  and `DGDataLoader`. **`DGData` is the input you construct a `DGraph` from** —
  the typical flow is `DGData.from_tgb(name).split()` → `DGraph(split_data)` →
  `DGDataLoader(dg, ...)`.
- **`tgm/hooks/`** — the execution layer (see below).

Relationship in one line: `tgm/data` produces graph data → `tgm/core` views and
materializes it into `DGBatch` → `tgm/hooks` enrich the batch → `tgm/nn` consumes
the batch. Models never touch `DGData` or storage directly; they read fields off
`DGBatch`.

## How batches actually work (and the DyGLib trap)

If you've used DyGLib / the TGB reference implementations, **do not carry those
assumptions over.** TGM does not pass around the DyGLib-style padded
`src_node_ids / dst_node_ids / node_interact_times` numpy arrays, and neighbor
sampling is *not* baked into the model. Instead:

- A `DGBatch` (`tgm/core/batch.py`) always has `edge_src`, `edge_dst`, `edge_time`
  (shape `(E,)`), and optionally `edge_x`, `edge_type`, and node feature/label
  triples (`node_x`/`node_x_nids`/`node_x_time`, same for `node_y`).
- **Everything else a model needs (sampled neighbors, negatives, local indexing)
  is added onto the batch by hooks at load time**, not by the model. Hooks
  declare `requires`/`produces` string keys and the `HookManager` topologically
  orders them. A hook writes attributes via `add_batch_attribute`; the model reads
  them by name. So `batch.nbr_nids`, `batch.neg`, `batch.unique_nids`,
  `batch.global_to_local` etc. exist only if the corresponding hook is registered.
- **Padded neighbors use `PADDED_NODE_ID = -1`** (`tgm/constants.py`), not 0. Mask
  with `nbr != PADDED_NODE_ID` before indexing (see `examples/linkproppred/tgn.py`).
  Using `-1` as a real node ID in `DGData` raises `InvalidNodeIDError`.
- **Indexing is global-by-default.** `edge_src`/`edge_dst`/neighbor IDs are global
  node IDs. The `DeduplicationHook` produces `batch.unique_nids` (sorted unique
  nodes in the batch) and `batch.global_to_local` (a callable doing
  `searchsorted` into `unique_nids`). Models gather embeddings in *local* space:
  embed `unique_nids`, then index with `global_to_local(global_ids)`. Forgetting
  this mixes up the embedding rows.

Negative sampling and neighbor sampling are configured through hooks +
**recipes**. `RecipeRegistry.build(RECIPE_TGB_LINK_PRED, dataset_name=..., train_dg=...)`
returns a `HookManager` with `train`/`val`/`test` keys wired with the *correct*
negative samplers per split (random for train, TGB's fixed negatives for val/test)
— use it rather than hand-rolling negatives, which is the #1 way to produce
non-comparable link-pred numbers. Activate a split with the context manager:
`with hm.activate('train'): ...`, and call `hm.reset_state()` between epochs
(stateful hooks like memory/recency-neighbors carry state across batches).

## Adding a new model

Follow the project's stated process (`.github/CONTRIBUTING.md` → "Proposing a new
model"), which is deliberately example-first:

1. **Write an example script first**, under `examples/<task>/<model>.py`
   (`task` ∈ `linkproppred`, `nodeproppred`, `graphproppred`, `analytics`).
   Mirror an existing example closely: argparse with the standard flags
   (`--dataset`, `--device`, `--bsize`, `--epochs`, `--seed`, …), `seed_everything`,
   `DGData.from_tgb(...).split()`, hook/recipe setup, the train/eval loops using
   `hm.activate(...)`. Keep the model itself as `torch.nn.Module`s in the script.
1. **Add an integration test** at `test/integration/test_<model>.py` mirroring an
   existing one (`@pytest.mark.integration` + `@pytest.mark.slurm(resources=[...])`,
   request the minimum resources, assert the job reaches `COMPLETED`).
1. Only **after** the implementation is validated on the cluster do reusable
   pieces (a layer, a hook) get promoted into `tgm/nn/` or `tgm/hooks/`. Anything
   moved into core **must** ship with unit tests in `test/unit/` and a justification.

To register a promoted model so `from tgm.nn import MyModel` works, add it to the
relevant subpackage `__init__.py` (`encoder/`, `decoder/`, or `modules/`) **and**
to `tgm/nn/__init__.py`'s imports and `__all__`. New hooks register in
`tgm/hooks/__init__.py`. A custom recipe is registered with the
`@RecipeRegistry.register(name)` decorator.

## Hard constraints — do not violate

- **Reproducibility / benchmark integrity.** Examples are the reference
  implementations whose reported numbers must stay comparable. Never silently
  change default hyperparameters, the seed handling (`seed_everything`, default
  `--seed 1337`), the metric (`mrr` for link, `ndcg` for node — see
  `METRIC_TGB_*`), or the negative-sampling setup in an existing example. The
  val/test negative samplers come from TGB's frozen negatives via the recipe;
  swapping in random negatives for eval invalidates results.
- **Don't reach past the public API.** `DGStorage` and the `_storage` backend are
  private (leading underscore). Go through `DGData` / `DGraph` / `DGBatch`.
- **`DGraph` is an immutable view.** `slice_events` / `slice_time` return new views
  sharing storage; cached properties (`num_nodes`, `edges`, …) are *not*
  invalidated if you mutate underlying data. Don't mutate; re-slice.
- **`slice_events` uses event indices, `slice_time` uses timestamp values**, and
  both treat the end as *exclusive*. Mixing them up silently shifts batches.
- **Determinism on GPU** additionally needs
  `torch.backends.cudnn.deterministic = True` / `benchmark = False` (see the note
  in `tgm/util/seed.py`); `seed_everything` alone does not guarantee it.
- **Core dependencies stay minimal.** Anything beyond numpy/torch/torch-geometric
  goes in a dependency group (`dev`, `examples`, `analytics`, `docs`), never the
  core `dependencies`.

## Pointers

- Architecture overview: `docs/architecture.md` (three layers: Data / Execution / ML).
- API reference: `docs/api/`. Tutorials (hooks, time-delta, DGraph, THGL,
  TGB-Seq): `docs/tutorials/`.
- Paper: <https://arxiv.org/abs/2510.07586>.
