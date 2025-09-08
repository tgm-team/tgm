# Time Management in Temporal Graphs

This tutorial explains how **time deltas** work in TGM, how they influence graph construction, iteration, and discretization, and how to use them effectively.

______________________________________________________________________

## 1. TimeDeltaDG: High-Level Concept

A `TimeDeltaDG` defines the **temporal granularity** of a dynamic graph. It specifies the “unit of time” at which events (edges or nodes) are recorded. Think of it as the resolution of your graph’s timeline.

See `tgm.timedelta.TimeDeltaDG` for full reference.

### Construction

You can create a `TimeDeltaDG` using a string alias or by explicitly providing a unit and a multiplier:

```python
from tgm.timedelta import TimeDeltaDG

# Basic usage
td_seconds = TimeDeltaDG("s")       # 1-second granularity
td_days = TimeDeltaDG("D")          # 1-day granularity
td_biweekly = TimeDeltaDG("W", 2)   # 2-week (bi-weekly) granularity
```

#### Ordered vs. Non-Ordered

There are 2 broad classes of `TimeDeltaDG` which determine how timestamps on a graph are interpreted:

- Ordered (`r`): Events are only guaranteed to have a relative order. No actual temporal measurement is associated.
- Non-Ordered (e.g. second-wise (`s`), or daily (`D`)): Standard time units like seconds, minutes, days, etc. Can perform coarsening or time conversion.

```python
td_ordered = TimeDeltaDG("r") # Only relative order matters
```

The full list of non-ordered units is given below:

```python
'Y': Yearly
'M': Monthy
'W': Weekly
'D': Daily
'h': Hourly
'm': Minute-wise
's': Second-wise
'ms': Millisecond-wise
'us': Microsecond-wise
'ns': Nanosecond-wise
```

#### Coarser vs. Finer Granularities

*Coarser* time granularities use lower resolution time units (e.g. week is coarser than day):

```python
td_day = TimeDeltaDG("D")
td_week = TimeDeltaDG("W")

print(td_week.is_coarser_than(td_day)) # True
```

**Note**: Checking whether an ordered time delta is coarser or finer than an non-ordered is undefined and will raise a `OrderedGranularityConversionError`.

## 2. DGData Construction

Every `DGData` requires an associated `TimeDeltaDG`. Predefined datasets (e.g. `tgbl-wiki`) have native time deltas, usually in seconds.

If you are using a custom dataset, you must specify a time delta. If the exact temporal unit is unknown, you can resort to ordered granularity `TimeDelta('r')`:

```python
from tgm.data import DGData

# Custom dataset with day granularity
dg_data = DGData(
    time_delta="D",
    timestamps=timestamps,
    ...
)

# Ordered dataset (relative order only)
dg_ordered = DGData(
    time_delta="r",
    timestamps=timestamps,
    ...
)
```

See :class:`tgm.data.DGData` for full reference.

## 3. Temporal Data Iteration

The time delta also informs how you iterate over the graph. In this respect, the `DGDataLoader` uses two key parameters:

- `batch_unit`: Unit of time for batching (`r`, `D`, `h`)
- `batch_size`: Number of units or events per batch.

### Iteration Modes

1. By events (ordered) - default

   - Iterates over a fixed number of events at a time.
   - `batch_unit='r'` and `batch_size=N` yields N events per batch.

1. By time (non-ordered)

   - Iterates over a time window (e.g. 3 hours at a time).
   - Converts underyling timestamps to match the requested batch granularity
   - Can result in empty batches if no events occur in the window. Can specify `on_empty='raise'` to error on empty batches of `on_empty='skip` to ignore them.

```python
from tgm.loader import DGDataLoader

# Ordered iteration
loader = DGDataLoader(dg_data, batch_size=10, batch_unit='r')

# Time-based iteration (non-ordered)
loader_time = DGDataLoader(dg_data, batch_size=3, batch_unit='D', on_empty='skip')
```

See :class:`tgm.loader.DGDataLoader` for full reference.

### Discretization: Coarsening Graphs

Discretization allows you to *coarsen* a non-ordered graph to a new time granularity:

- multiple edge and node events are partitioned into time buckets
- we keep the first occurrence of duplicate events per bucket (future versions will support other reduction ops)

This is useful for tuning dataset granularity (e.g. converting from continuous to discrete temporal graphs).

```python
dg_daily = dg_data.discretize(time_delta="D", reduce_op="first")
```

**Note**: This is only applicable for non-ordered graphs. Attempting to discretize an ordered `DGData` is undefined and will raise `InvalidDiscretizationError`.

### Summary

The time delta is central to managing timestamps on your temporal graph. If using existing dataset (e.g. TGB), the time delta is already defined. For custom datasets, you need to provide either an ordered (`r`) or non-ordered (e.g. `s`) time unit.

Non-ordered time deltas are strictly more general in that they enable you to discretize (coarsen) you dataset, and iterate by temporal snapshots.
