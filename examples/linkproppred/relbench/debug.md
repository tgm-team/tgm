# RelHM OOM Debug Notes

Job killed at ~45% of a training epoch with 32 GB memory.
Run parameters (`run.sh`): `--device cuda --bsize 512 --n-nbrs 20 --memory-dim 128`.

______________________________________________________________________

## Issue 1 — CRITICAL: `LastAggregator` creates an O(N²) tensor on `memory.eval()`

**Where:** `tgm/nn/encoder/tgn.py`, `LastAggregator.forward()`

```python
scores = torch.full((dim_size, t.size(0)), float('-inf'), device=t.device)
```

At the end of every training epoch, `evaluate()` calls `memory.eval()`, which triggers
`TGNMemory.train(mode=False)` → `_update_memory(torch.arange(num_nodes))` — a full flush
of all 1,476,488 nodes. Inside `_get_updated_memory`, `LastAggregator` receives:

- `dim_size = 1,476,488`
- `t.size(0) ≈ 1,476,488` (one stored message per active node)

`scores = torch.full((1,476,488, 1,476,488), -inf)` = **~8.7 TB** → immediate OOM.

**Fix (train.py):** `_MemoryEfficientLastAggregator` subclass uses a double stable sort
(O(M log M) time, O(M) space) instead of the matrix:

```python
perm = torch.argsort(t, stable=True)
perm = perm[torch.argsort(index[perm], stable=True)]
sorted_idx = index[perm]
node_ids, counts = torch.unique_consecutive(sorted_idx, return_counts=True)
last_pos = counts.cumsum(0) - 1
out[node_ids] = msg[perm[last_pos]]
```

Passed as `aggregator_module=_MemoryEfficientLastAggregator()` when constructing `TGNMemory`.

______________________________________________________________________

## Issue 2 — HIGH: `_reset_message_store` pre-allocates 1.48 M dict entries

**Where:** `tgm/nn/encoder/tgn.py`, `_reset_message_store()`

```python
self.msg_s_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}
self.msg_d_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}
```

Called at the start of each epoch via `hm.reset_state()` → `memory.reset_state()`.
Two dicts with 1.48 M Python int keys + tuple values ≈ **~300 MB** of Python heap,
created and thrown away each epoch.

**Fix (train.py):** `_patch_memory_for_large_graphs()` replaces `_reset_message_store`
with a version that starts with empty dicts `{}` and stores a single sentinel tuple for
unseen nodes. `_compute_msg` is patched to use `dict.get(key, sentinel)`.

______________________________________________________________________

## Issue 3 — MEDIUM: `_update_msg_store` stores views into batch GPU tensors

**Where:** `tgm/nn/encoder/tgn.py`, `_update_msg_store()`

```python
msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx])
```

`src[idx]` is a **view** that shares GPU storage with the batch tensor `src`.
The GPU memory of each batch's `edge_src / edge_dst / edge_time / edge_x` tensors stays
live until every node from that batch is updated again in a later batch.

With 1.37 M customers and bsize=512, a customer re-appears every ~2,600 batches on
average. At batch ~9,400 (45% of epoch), up to 9,400 × 12 KB = **~112 MB** of GPU
memory is held by stale batch references.

**Fix (train.py):** `_patch_memory_for_large_graphs()` replaces `_update_msg_store`
with a version that calls `.clone()` on every stored slice, giving each entry its own
independent storage.

______________________________________________________________________

## Issue 4 — MEDIUM: `RelHMMetadata` holds large DataFrames never used in `evaluate()`

**Where:** `data.py` (`RelHMMetadata` dataclass), `train.py` (`evaluate()` signature)

`RelHMMetadata` stored `train_table`, `val_table`, `test_table` — full copies of the
RelBench task DataFrames. The `evaluate()` function accepted a `split_table` parameter
but never referenced it in its body. These DataFrames were loaded and held in CPU RAM
throughout the entire training run for no reason.

**Fix:**

- `data.py`: removed `train_table`, `val_table`, `test_table` from `RelHMMetadata`.
- `train.py`: removed `meta` and `split_table` from `evaluate()` signature and updated
  both call sites in `main()`.

______________________________________________________________________

## Memory budget (run.sh parameters)

| Component                                                     | Size        | Location              |
| ------------------------------------------------------------- | ----------- | --------------------- |
| TGNMemory.memory `[1.48M, 128]`                               | 755 MB      | GPU                   |
| RecencyNeighborHook `_nbr_ids/times/feats` `[1.48M, 20, ...]` | ~595 MB     | GPU                   |
| TGNMemory.last_update + \_assoc                               | ~24 MB      | GPU                   |
| Edge tensors (DGData)                                         | ~366 MB     | CPU                   |
| Task DataFrames (before fix)                                  | 100–500 MB  | CPU                   |
| msg_store dicts (before fix)                                  | ~300 MB     | CPU (per epoch reset) |
| msg_store GPU views (before fix, peak at 45%)                 | ~112 MB     | GPU                   |
| `scores` tensor during eval flush (before fix)                | **~8.7 TB** | GPU → OOM             |

______________________________________________________________________

## Files changed

All changes are confined to `examples/linkproppred/relbench/`.

- `train.py` — added `_MemoryEfficientLastAggregator`, `_patch_memory_for_large_graphs`;
  wired both into `main()`; removed dead `meta` / `split_table` params from `evaluate()`.
- `data.py` — removed `train_table`, `val_table`, `test_table` from `RelHMMetadata`.

The underlying bugs in `tgm/nn/encoder/tgn.py` (`LastAggregator`, `_reset_message_store`,
`_update_msg_store`) are worked around from the example layer. They should be fixed
upstream in a separate PR.
