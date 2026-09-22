# TGTalker — LLMs as temporal graph learners

A tgm-native reference port of [**TGTalker**](https://github.com/shenyangHuang/TGTalker)
(paper: *"Are Large Language Models Good Temporal Graph Learners?"*,
[arXiv:2506.05393](https://arxiv.org/abs/2506.05393)).

TGTalker performs **zero-shot temporal link prediction**: it serializes a node's
recent temporal-graph history into a natural-language prompt and asks a *frozen*
Hugging Face causal LLM to predict the next `Destination Node`. There is no
training — the LLM weights are never updated. Predictions are scored against the
official TGB negative samples with **MRR**.

This example reuses tgm's data and hook primitives so the method drops directly
into the standard TGB link-prediction setup.

## Contents

| File                    | Purpose                                                                                   |
| ----------------------- | ----------------------------------------------------------------------------------------- |
| `TGTalker.py`           | Main link-prediction script: base mode + background context + ICL + CoT.                  |
| `multihop.py`           | Multi-hop variant: prompt includes the source's k-hop temporal neighborhood.              |
| `tgtalker_utils.py`     | Prompt construction, scoring, and sliding-window helpers (dependency-light, unit tested). |
| `schemas.py`            | Pydantic schemas for structured LLM output via `outlines`.                                |
| `test_tgtalker.py`      | Unit tests for the prompt/util logic (no GPU/model needed).                               |
| `posthoc_explanations/` | Optional 3-stage pipeline that uses an OpenAI model to explain predictions.               |

## Install

The LLM dependencies live in the `llm` extra (`outlines`, `transformers`); TGB
datasets come from the `analytics`/`dev` groups (`py-tgb`):

```bash
uv sync --group llm --group analytics      # or: pip install "outlines>=1.2.9" "transformers>=4.36.2" py-tgb
```

## Run

```bash
# Base zero-shot mode
python examples/llm/TGTalker.py --dataset tgbl-wiki --model Qwen/Qwen3-1.7B --device cuda

# + global background-edge context window (the shared "TEMPORAL GRAPH")
python examples/llm/TGTalker.py --dataset tgbl-wiki --bg-size 300

# + in-context-learning demonstrations
python examples/llm/TGTalker.py --dataset tgbl-wiki --icl --in-size 5

# + chain-of-thought reasoning
python examples/llm/TGTalker.py --dataset tgbl-wiki --cot

# multi-hop neighborhood context
python examples/llm/multihop.py --dataset tgbl-wiki --hops 2 --n-nbrs 2

# quick smoke test: tiny model, CPU, only the first 8 test edges
python examples/llm/TGTalker.py --dataset tgbl-wiki --model Qwen/Qwen3-0.6B \
  --device cpu --bsize 8 --max-test-edges 8
```

### Arguments (`TGTalker.py`)

| Arg                                                         | Default           | Meaning                                      | Original    |
| ----------------------------------------------------------- | ----------------- | -------------------------------------------- | ----------- |
| `--dataset`                                                 | `tgbl-wiki`       | TGB link-prediction dataset                  | `--data`    |
| `--model`                                                   | `Qwen/Qwen3-1.7B` | HF causal LM path                            | `--model`   |
| `--bsize`                                                   | `200`             | test batch size                              | `--batch`   |
| `--n-nbrs`                                                  | `2`               | recent neighbors of the source in the prompt | `--nbr`     |
| `--bg-size`                                                 | `300`             | size of the global background-edge window    | `--bg_size` |
| `--in-size`                                                 | `5`               | number of ICL demonstrations                 | `--in_size` |
| `--icl`                                                     | off               | enable in-context learning                   | `--icl`     |
| `--cot`                                                     | off               | enable chain-of-thought                      | `--cot`     |
| `--device`, `--seed`, `--log-file-path`, `--max-test-edges` | —                 | runtime options                              | —           |

## Implementation details

### Data → hook → prompt → LLM → MRR

1. **Load & split.** `DGData.from_tgb(dataset).split()` → train/val/test
   `DGraph`s (`tgm/data/dg_data.py`, `tgm/core/graph.py`).
1. **Neighbor history.** A `RecencyNeighborHook`
   (`tgm/hooks/neighbors/recency.py`) maintains per-node circular buffers of the
   most recent neighbors. It is seeded on `edge_src`/`edge_time` and configured
   `directed=True` so each source keeps only its **outgoing** history, matching
   the original `NeighborTracker`. The hook produces `nbr_nids` and
   `nbr_edge_time` on each batch; `batch.nbr_nids[0][i]` are the recent neighbors
   of source `i`. Queries are processed in chronological (event) order so the
   neighbors of an edge never include the edge itself.
1. **Negatives.** `RecipeRegistry.build(RECIPE_TGB_LINK_PRED, ...)`
   (`tgm/hooks/recipe.py`) registers `TGBNegativeEdgeSamplerHook` for val/test,
   producing `batch.neg_batch_list` — the official per-edge candidate negatives.
1. **Warm-up.** Iterating the train and val loaders populates the recency
   buffers (and seeds the background/ICL windows from the most recent edges)
   before test inference begins.
1. **Prompt.** A system prompt frames the task and (optionally) carries the
   shared background graph + ICL demonstrations + a chain-of-thought directive; a
   per-edge user prompt lists the source's recent interactions and asks for the
   next destination. Built with `tokenizer.apply_chat_template(...)`.
1. **Structured generation.** `outlines.from_transformers(model, tokenizer)`
   constrains the output to a JSON schema (`TGAnswer`, or `TGReasoning` with
   `--cot`). The `destination_node` field is parsed out.
1. **Score.** `predict_link` builds a 0/1 vector over `[true_dst, *negatives]`
   (1.0 where a candidate equals the LLM's prediction); the TGB `Evaluator`
   converts it to MRR. Picking the true destination → MRR 1.0; picking a negative
   or an out-of-set node → 0.0.

### Sliding-window context (no leakage)

`BackgroundBuffer` and `ICLWindow` (`collections.deque`-backed) hold the most
recent global edges / demonstrations. They are seeded from the validation tail
and **updated with each test batch only after that batch's prompts are built**,
so shared context never reveals the edge currently being predicted.

### Prompt format

System prompt (with `--bg-size`, `--icl`, `--cot` all on):

```
You are an expert temporal graph learning agent. Your task is to predict the
next interaction (i.e. `Destination Node`) given the `Source Node` and
`Timestamp`.

Description of the temporal graph is provided below, where each line is a tuple
of (`Source Node`, `Destination Node`, `Timestamp`).

TEMPORAL GRAPH:
(2, 4, 5)
(1, 5, 6)
...

Let's think step by step about the problem.

Here are some examples:
Predict the next interaction for `Source Node` 1 at `Timestamp` 6. Answer: {"destination_node": 5}
...
```

User prompt (with neighbor history):

```
`Source Node` 2 has the following past interactions:
(2, 0, 2)
(2, 4, 5)
Please predict the most likely `Destination Node` for `Source Node` 2 at `Timestamp` 8.
```

### Multi-hop (`multihop.py`)

`RecencyNeighborHook(num_nbrs=[k] * hops)` returns one neighbor tensor per hop.
For hop `h`, source `i` owns rows `[i * k**h, (i + 1) * k**h)` of the flattened
hop-`h` tensors; `gather_hop_edges` reconstructs the per-hop `(seed, neighbor, time)` edges for each source (skipping padded seeds/neighbors). Because hop-`h`
queries use the hop-`(h-1)` edge time, the multi-hop context is temporally
causal — a 2-hop neighbor is only included if it occurred before the 1-hop edge.

### Post-hoc explanations (`posthoc_explanations/`)

A 3-stage pipeline mirroring the original's `answer_cache → prompt_cache → explanation` flow:

```bash
cd examples/llm/posthoc_explanations
python answer_cache.py --dataset tgbl-wiki --max-test-edges 5000   # stage 1: cache base-model answers (needs llm extra)
python prompt_cache.py --only-incorrect                            # stage 2: build explanation prompts (no heavy deps)
export OPENAI_API_KEY=sk-...
python generate_explanations.py --model gpt-4o-mini --limit 100    # stage 3: GPT explanations + categories (needs `openai`)
```

Stage 3 asks a GPT-4-class model to explain each prediction and classify the
reasoning into `repetition / recency / popularity / structural / uncertain`, and
prints the category distribution. It makes paid API calls and is **not** run in
CI; it exits cleanly if `OPENAI_API_KEY` is unset.

## Tests

```bash
pytest examples/llm/test_tgtalker.py
```

These cover the prompt builders, `predict_link`, the sliding windows, and the
multi-hop edge gathering. They require neither a GPU nor a model download. (The
schema test self-skips if `pydantic` is not installed.)

## Divergences from the original repo

This port is faithful to the method but intentionally differs from the published
code in the following ways:

1. **Bug fixes in the pre-existing tgm port.** The earlier draft of this example
   could not run. Fixed: a stray `quit()` + debug `print`s that aborted before
   any inference; wrong batch attribute names (`batch.src` → `batch.edge_src`,
   `batch.times[0]` → `batch.edge_time`, `batch.nbr_times` → `batch.nbr_edge_time`,
   and `seed_nodes_keys=['src']`/`['time']` → `['edge_src']`/`['edge_time']`); and
   a `make_user_prompt` `isinstance(src, int)` check that rejected the
   `numpy.int64` scalars the pipeline actually yields (now coerced via a helper
   that accepts python/numpy/torch integer scalars).

1. **Prompt typo fixes.** The original user prompt contained malformed strings;
   they are cleaned up here:

   - Original: `f",Source Node\` {src} has the following past interactions:\\n"```  (leading comma, unbalanced backtick). Cleaned: `` f" ```Source Node\` {src} has the following past interactions:\\n" \`\`
   - Original edge line: `f"{src}, {dst}, {timestamp}) \n"` (unbalanced
     parenthesis, trailing space).
     Cleaned: `f"({src}, {dst}, {timestamp})\n"`.

1. **Inference backend.** The original uses **vLLM** (with prefix caching, and a
   `transformers` fallback only for CoT). This port uses **`transformers` +
   `outlines`** throughout for a lighter, single-path dependency footprint. The
   trade-off is no prefix caching, so the large shared background/ICL prefix is
   re-encoded per query (slower, same outputs).

1. **tgm-hook reimplementation.** The original's bespoke `NeighborTracker`,
   `background_rows` numpy slicing, and ICL example windows are reimplemented on
   tgm primitives: `RecencyNeighborHook` for neighbor history and
   `BackgroundBuffer` / `ICLWindow` deques for the sliding context. Behavior is
   equivalent (recent history seeded from val, updated chronologically without
   leakage); the recency hook is set `directed=True` to match the original's
   outgoing-only source history.

1. **Argument renames** (to match tgm's hyphenated style): `--data` → `--dataset`,
   `--batch` → `--bsize`, `--nbr` → `--n-nbrs`, `--in_size` → `--in-size`,
   `--bg_size` → `--bg-size`. Added `--max-test-edges` to cap inference for quick
   smoke tests. `--n-nbrs` defaults to **2** (the original `--nbr` default).

1. **`row2text` formatting.** The original concatenated tuples with no separator;
   this port writes one `(src, dst, ts)` tuple per line for readability.

1. **ICL demonstration form.** Demonstrations reuse the neighbor-free user-prompt
   form (`Predict the next interaction for Source Node ... at Timestamp ...`)
   paired with a JSON answer, rather than reconstructing each historical
   demonstration's own neighbor context (which would require re-querying the hook
   at past timestamps).
