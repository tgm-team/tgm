# Issue #417: Discount node_y from batch size in event-ordered DGDataLoader

## Problem

When using event-ordered batching (`batch_unit='r'`), `node_y` (node label) events occupy
positions in the global event array and consume the batch size budget alongside edge and
node feature events. For a nodeproppred task with `batch_size=200`, some of those 200
"events" are actually label annotations, not structural graph events. This means the model
sees fewer edge events per batch than intended.

### Example

Given this global event sequence (sorted by time):

```
position: 0      1      2      3        4      5
event:     edge0  edge1  node_y  edge2   edge3  node_y
```

With `batch_size=3` (current behavior — node_y counts):
- Batch 0: positions [0, 3) → edge0, edge1, node_y  (only 2 edge events)
- Batch 1: positions [3, 6) → edge2, edge3, node_y   (only 2 edge events)

With `batch_size=3` (desired — node_y does NOT count):
- Batch 0: positions [0, 4) → edge0, edge1, node_y, edge2  (3 non-label events + node_y at correct chronological position)
- Batch 1: positions [4, 6) → edge3, node_y                (1 non-label event + node_y)

node_y still appears in each batch in chronological order; it just doesn't consume a batch slot.

## Rationale for each change

### Why a new parameter instead of always discounting?

Adding `count_node_labels: bool = True` (opt-in via `False`) is backward-compatible.
Changing the default would silently alter batch boundaries for existing users.

### Why only applies to event-ordered batching?

For time-ordered batching (`batch_unit='s'`, `'Y'`, etc.), the batch size is measured in
time units, not event counts. node_y events already do not consume "time budget". The
parameter is accepted but is a no-op for time-ordered batching.

### Why store a `_slice_ends` dict?

The current DataLoader precomputes a uniform `range(start, stop, step)` and computes
`end = start + batch_size` in `__call__`. When node_y is discounted, batch boundaries are
non-uniform (they depend on where node_y events sit in the global sequence). A dict
mapping `start → end` handles both cases with a single fallback: if the key is missing
(uniform batching), fall back to `start + batch_size`.

## Key data structure facts

- `DGData.time`: 1-D tensor, shape `[num_edge_events + num_node_events + num_node_labels]`,
  all event timestamps in **chronological order**.
- `DGData.node_y_mask`: tensor of **absolute positions** in `time` where node_y events sit.
- `get_num_events(slice)` in the storage backend returns `ub_idx - lb_idx` — it counts
  everything including node_y.
- `slice_events(start, end)` slices by absolute position in the global event array.
- For a full (unsliced) `DGraph`, `lb_idx = 0`, so absolute positions == relative positions.
  The DataLoader always operates on the full `dg` before slicing, so this holds throughout.

## Step-by-step implementation

### Step 1 — `tgm/core/_storage/base.py`

Add an abstract method to `DGStorageBase` after `get_num_events`:

```python
@abstractmethod
def get_node_y_event_positions(self, slice: DGSliceTracker) -> Tensor:
    """Return absolute event-array positions of node_y events within the slice."""
```

**Rationale**: All storage backends must implement this so `DGraph` can call it without
knowing which backend is in use.

### Step 2 — `tgm/core/_storage/backends/array_backend.py`

Implement the method (mirrors `get_node_labels` but returns positions, not nids/times):

```python
def get_node_y_event_positions(self, slice: DGSliceTracker) -> Tensor:
    if self._data.node_y_mask is None:
        return torch.empty(0, dtype=torch.long)
    lb_idx, ub_idx = self._binary_search(slice)
    in_slice = (self._data.node_y_mask >= lb_idx) & (self._data.node_y_mask < ub_idx)
    return self._data.node_y_mask[in_slice]
```

**Rationale**: `node_y_mask` stores the exact absolute positions of node_y events in the
global sorted `time` array. Filtering by `[lb_idx, ub_idx)` restricts to the current
slice.

### Step 3 — `tgm/core/graph.py`

Add a cached property alongside `num_node_labels` (~line 209):

```python
@_logged_cached_property
def node_y_event_positions(self) -> Tensor:
    """Absolute event-array positions of node_y events in the current slice."""
    return self._storage.get_node_y_event_positions(self._slice)
```

**Rationale**: Exposes the new storage method through the `DGraph` public API so the
DataLoader can access it without touching private internals.

### Step 4 — `tgm/data/loader.py`  ← main change

#### 4a. New parameter

Add `count_node_labels: bool = True` to `DGDataLoader.__init__` signature and docstring.

#### 4b. New field

Add `self._slice_ends: dict = {}` before the `if batch_time_delta.is_event_ordered` block.

#### 4c. Non-uniform batch boundary computation

Inside the `if batch_time_delta.is_event_ordered:` branch, after setting
`start_idx, stop_idx = 0, dg.num_events`, add:

```python
if not count_node_labels and dg.num_node_labels > 0:
    total = dg.num_events
    node_y_pos = dg.node_y_event_positions          # absolute positions tensor
    is_node_y = torch.zeros(total, dtype=torch.bool)
    is_node_y[node_y_pos] = True
    non_label_pos = torch.where(~is_node_y)[0]      # positions of non-node_y events

    drop_last = kwargs.get('drop_last', False)
    if drop_last:
        n_complete = len(non_label_pos) // batch_size
        starts = [non_label_pos[i * batch_size].item() for i in range(n_complete)]
        ends   = [non_label_pos[(i + 1) * batch_size].item() for i in range(n_complete)]
    else:
        starts = non_label_pos[::batch_size].tolist()
        ends   = non_label_pos[batch_size::batch_size].tolist() + [total]

    self._slice_ends = dict(zip(starts, ends))
    slice_start = starts
else:
    # existing range-based logic unchanged
    if kwargs.get('drop_last', False):
        slice_start = range(start_idx, stop_idx - batch_size, batch_size)
    else:
        slice_start = range(start_idx, stop_idx, batch_size)
```

#### 4d. Update `__call__`

```python
def __call__(self, slice_start: List[int]) -> DGBatch:
    start = slice_start[0]
    end = self._slice_ends.get(start, start + self._batch_size)
    dg = self._slice_op(start, end)
    batch = dg.materialize()
    ...  # rest unchanged
```

**Rationale for the dict lookup**: When `_slice_ends` is populated (non-uniform batching),
it returns the precomputed end. When empty (uniform batching), it falls back to
`start + self._batch_size`, preserving the exact existing behaviour.

### Step 5 — `test/unit/test_data/test_dataloader.py`

Add the following test cases.

#### Test A: correct batch contents with interleaved node_y

```
Graph events (time-sorted):
  pos 0: edge (t=0)
  pos 1: edge (t=1)
  pos 2: node_y (t=1)   ← label event
  pos 3: edge (t=2)
  pos 4: node_y (t=3)   ← label event
  pos 5: edge (t=4)

non_label_pos = [0, 1, 3, 5]
batch_size = 2
```

Expected batches with `count_node_labels=False`:
- Batch 0: slice [0, 3) → events at pos 0, 1, 2 → 2 edges + 1 node_y (node_y at t=1 ✓)
- Batch 1: slice [3, 6) → events at pos 3, 4, 5 → 2 edges + 1 node_y (node_y at t=3 ✓)

Assert:
- `len(list(loader)) == 2`
- Batch 0 has `batch.node_y is not None` and correct nids/values
- Batch 1 has `batch.node_y is not None` and correct nids/values
- Each batch has exactly 2 edges

#### Test B: drop_last with count_node_labels=False

Same graph, `batch_size=3`, `drop_last=True`.

`non_label_pos = [0, 1, 3, 5]` → only 1 complete batch of 3 (positions 0, 1, 3).

- Batch 0: slice [0, 5) → 3 non-label events + node_y at pos 2 and 4
- No second batch (drop_last)

Assert: `len(list(loader)) == 1`

#### Test C: no node_y — falls back to uniform batching

Graph with only edge events, `count_node_labels=False`. Verify behaviour is identical
to `count_node_labels=True` (existing tests cover this path; here we just confirm no error
and same batches).

#### Test D: count_node_labels=True (default) — unchanged behaviour

Use the same interleaved graph as Test A with `count_node_labels=True` (default) and
verify that node_y events do count toward the batch size (original behaviour preserved).

## Files changed (summary)

| File | Change |
|------|--------|
| `tgm/core/_storage/base.py` | Add abstract `get_node_y_event_positions` |
| `tgm/core/_storage/backends/array_backend.py` | Implement `get_node_y_event_positions` |
| `tgm/core/graph.py` | Add `node_y_event_positions` cached property |
| `tgm/data/loader.py` | Add `count_node_labels` param, `_slice_ends` dict, non-uniform boundary logic, update `__call__` |
| `test/unit/test_data/test_dataloader.py` | Add Tests A–D above |

## Verification

```bash
# Run all dataloader tests
pytest test/unit/test_data/test_dataloader.py -v

# Quick smoke test with a nodeproppred dataset (if available)
python examples/nodeproppred/gcn.py --dataset tgbn-trade
```
