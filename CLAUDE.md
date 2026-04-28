# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Setup:**

```sh
uv sync                        # Install core dependencies
uv sync --group dev            # Install with dev dependencies
uv run pre-commit install      # Install pre-commit hooks
```

## Shell Environment

Always prefix Python/pytest/ruff commands with `source /Users/andang/tgm/.venv/bin/activate &&` when running in Bash, e.g.:

```bash
source /Users/andang/tgm/.venv/bin/activate && python ...
source /Users/andang/tgm/.venv/bin/activate && pytest ...
```

**Run unit tests (CPU-only):**

```sh
./scripts/run_unit_tests.sh
```

**Run a single test:**

```sh
uv run pytest test/unit/path/to/test_file.py::test_name -vvv
```

**Run tests with GPU:**

```sh
./scripts/run_unit_tests.sh --gpu
```

**Lint / format:**

```sh
uv run ruff format tgm         # Format code (single quotes, 88 char line length)
uv run ruff check tgm          # Lint
uv run mypy                    # Type checking (strict)
```

**Build docs:**

```sh
./scripts/build_docs.sh
```

## Architecture

TGM is a research library for temporal graph ML, organized as three layers:

### 1. Data Layer (`tgm/core/`, `tgm/data/`)

- **`DGStorage`** (`tgm/core/_storage/`): Immutable, time-sorted coordinate-format backend. Uses binary search over timestamps for efficient neighbor retrieval.
- **`DGraph`** (`tgm/core/graph.py`): Lightweight view over `DGStorage`. Supports `slice_events()` (by position) and `slice_time()` (by timestamp value). Operations like `.to(device)` and slicing return new views without copying data. `materialize()` produces dense tensors.
- **`DGBatch`** (`tgm/core/batch.py`): Batch container yielded by the dataloader.
- **`TimeDeltaDG`** (`tgm/core/timedelta.py`): Specifies time granularity (continuous vs. discrete).
- **`DGData`** (`tgm/data/dg_data.py`): Immutable dataclass holding edge/node events, timestamps, and features. Input to `DGraph`.
- **`DGDataLoader`** (`tgm/data/loader.py`): Iterates through the temporal graph stream.

### 2. Execution Layer (`tgm/hooks/`)

- **`HookManager`** (`tgm/hooks/hook_manager.py`): Orchestrates data transformations during loading (neighbor sampling, negatives, analytics, device placement). Hooks are registered under conditions (train, inference, etc.) and dynamically augment each `DGBatch`.
- **`Recipe`** (`tgm/hooks/recipe.py`): Pre-defined hook combinations for common tasks (e.g., TGB link prediction). Use recipes to avoid common pitfalls like mismanaging negative samples.
- Hook modules: `neighbors.py`, `negatives.py`, `dedup.py`, `device.py`, `batch_analytics.py`, `node_analytics.py`, `node_tracks.py`.

### 3. ML Layer (`tgm/nn/`)

- **Encoders** (`tgm/nn/encoder/`): TGCN, GC-LSTM, TGAT, TGN, DygFormer, TPNet, CTAN, ROLAND.
- **Decoders** (`tgm/nn/decoder/`): `LinkPropPred`, `NodePropPred`, `GraphPropPred`, `NCNPred`.
- **Memory** (`tgm/nn/memory/`): Memory modules for temporal state.
- **Modules** (`tgm/nn/modules/`): Attention, EdgeBank, MLPMixer, PopTrack, TimeEncoding.

### Examples (`examples/`)

Organized by task: `linkproppred/`, `nodepropred/`, `graphpropred/`. Each example is a standalone script with CLI args for dataset and device selection. New models should first be implemented as examples before being considered for inclusion in core.

## Key Conventions

- **Dependencies**: Core deps are minimal (`numpy`, `torch`, `torch-geometric`). Optional groups: `dev`, `analytics`, `docs`, `examples`. Manage with `uv add [--group <group>] <package>`.
- **Testing**: New additions to core require unit tests. GPU tests are marked `@pytest.mark.gpu`. Coverage target is 90%.
- **Type checking**: MyPy strict mode — all public functions must have type annotations.
- **Code style**: Ruff with single quotes, 88-char lines, Google-style docstrings.
- **New models**: Write an example first → add to integration test harness → migrate reusable components to core if broadly useful.
- **PRs**: Open an issue first, then fork and submit a PR referencing the issue.
