# Time Management in Temporal Graphs

This tutorial explains how **time deltas** work in TGM, how they influence graph construction, iteration, and discretization, and how to use them effectively.

______________________________________________________________________

## 1. TimeDeltaDG: High-Level Concept

A `TimeDeltaDG` defines the **temporal granularity** of a dynamic graph. It specifies the “unit of time” at which events (edges or nodes) are recorded. Think of it as the resolution of your graph’s timeline.

See [`tgm.timedelta.TimeDeltaDG`](../api/timedelta.md) for full reference.

### Construction

You can create a `TimeDeltaDG` using a string alias or by explicitly providing a unit and a multiplier:

```python
from tgm import TimeDeltaDG

# Basic usage
td_seconds = TimeDeltaDG("s")       # 1-second granularity
td_days = TimeDeltaDG("D")          # 1-day granularity
td_biweekly = TimeDeltaDG("W", 2)   # 2-week (bi-weekly) granularity
```

#### Event-Ordered vs. Time-Ordered

There are 2 broad classes of `TimeDeltaDG` which determine how timestamps on a graph are interpreted:

- Event-Ordered (`r`): Events are only guaranteed to have a relative order. No real-world time unit is associated.
- Time-Ordered (e.g. second-wise (`s`), or daily (`D`)): Standard time units like seconds, minutes, days, etc. Can perform coarsening or time conversion.

```python
td_ordered = TimeDeltaDG("r") # Only relative order matters
```

The full list of time-ordered units is given below:

| Time Unit | Meaning          |
| --------- | ---------------- |
| "Y"       | Yearly           |
| "M"       | Monthly          |
| "W"       | Weekly           |
| "D"       | Daily            |
| "h"       | Hourly           |
| "m"       | Minute-wise      |
| "s"       | Second-wise      |
| "ms"      | Millisecond-wise |
| "us"      | Microsecond-wise |
| "ns"      | Nanosecond-wise  |

#### Coarser vs. Finer Granularities

*Coarser* time granularities use lower resolution time units (e.g. week is coarser than day):

```python
td_day = TimeDeltaDG("D")
td_week = TimeDeltaDG("W")
td_biweek = TimeDeltaDG("W", 2)
td_month = TimeDeltaDG("M")

print(td_week.is_coarser_than(td_day)) # True
print(td_biweek.is_coarser_than(td_week)) # True
print(td_month.is_coarser_than(td_biweek)) # True
```

> **Note**: Checking whether an event-ordered time delta is coarser or finer than an non-ordered is undefined and will raise a `EventOrderedConversionError`.

## 2. DGData Construction

Every `DGData` requires an associated `TimeDeltaDG`. Predefined datasets (e.g. `tgbl-wiki`) have native time deltas, usually in seconds.

If you are using a custom dataset, you must specify a time delta. If the exact temporal unit is unknown, you can resort to event-ordered granularity `TimeDelta('r')`, which is the default:

```python
from tgm.data import DGData

# Custom dataset with day granularity
dg_data = DGData(
    time_delta="D",
    time=timestamps,
    ...
)

# Ordered dataset (relative order only)
dg_event_ordered = DGData(
    time_delta="r",
    time=timestamps,
    ...
)
```

See [`tgm.data.DGData`](../api/data.md) for full reference.

## 3. Temporal Data Iteration

The time delta also informs how you iterate over the graph. In this respect, the `DGDataLoader` uses two key parameters:

- `batch_unit`: Unit of time for batching (`r`, `D`, `h`)
- `batch_size`: Number of units or events per batch.

### Iteration Modes

There are two different modes of iteration in TGM, depending on whether the `batch_unit` parameter is event-ordered or time-ordered:

| Iteration Mode            | Meaning                                          | Example                                                              | Requires Time-Ordered Graph TimeDelta | Can produce empty batches |
| ------------------------- | ------------------------------------------------ | -------------------------------------------------------------------- | ------------------------------------- | ------------------------- |
| By Events (Event-Ordered) | Iterates over a fixed number of events at a time | Batch unit = `r` and batch size `N` yields N events per batch        | No                                    | No                        |
| By Time (Time-Ordered)    | Iterates over a time window                      | Batch unit = `h` and batch size `3` yields 3 hours of data per batch | Yes                                   | Yes                       |

> **Note**: Time-based iteration can result in empty batches if no edge and no node events occur in the window. You can specify `on_empty='raise'` to error on empty batches, `on_empty='skip'` to ignore them, or `on_empty=None` to materialize the empty snapshots for your model. The default will materialize empty snapshots.

```python
from tgm.data import DGDataLoader

# Event-ordered iteration: yield 10 events per batch
loader = DGDataLoader(dg_data, batch_size=10)

# Time-ordered iteration: yield 3 days of data per batch, skip empty batches
loader_time = DGDataLoader(dg_data, batch_size=3, batch_unit='D', on_empty='skip')

# Time-ordered iteration: yield 3 days of data per batch, raise ValueError on empty batches
loader_time = DGDataLoader(dg_data, batch_size=3, batch_unit='D', on_empty='raise')
```

See `tgm.loader.DGDataLoader` for full reference.
See [`tgm.loader.DGDataLoader`](../api/loader.md) for full reference.

## 4. Discretization: Coarsening Graphs

Discretization allows you to *coarsen* a time-ordered graph to a new time granularity:

- multiple edge and node events are partitioned into time buckets based on the requested granularity
- if multiple events map to the same edge in the same bucket, only the first occurrence is kept (future versions will support other reduction ops)

This is useful for tuning dataset granularity (e.g. converting from continuous to discrete temporal graphs).

```python
dg_data_second_wise = DGData.from_raw(
    time_delta="s",
    edge_time=torch.tensor([15, 30, 45, 60]),
    edge_index=torch.tensor([[0, 1], [2, 3], [0, 1], [0, 1]),
    edge_x=torch.tensor([[100, 200, 300, 400]]),
)

# Discretize from second-wise to minutely data
dg_data_minute_wise = dg_data_second_wise.discretize(time_delta="m", reduce_op="first")

# After discretizing, note that the edge interaction between node 0 and 1 at time 15 and 45 are duplicates
# after grouping to minute-wise buckets (minute 0). In this case, we keep the first event (edge_x 100)
# and drop the second event (edge_x 400).
print(dg_data.time_delta) # TimeDeltaDG("m")
print(dg_data.edge_time) # torch.tensor([0, 0, 1])
print(dg_data.edge_index) # torch.tensor([[0, 1], [2, 3], [0, 1])
print(dg_data.edge_x) # torch.tensor([[[100, 200, 400]])
```

> **Note**: Discretization is only defined for time-ordered graphs. Attempting to discretize an even-ordered `DGData` is undefined and will raise `InvalidDiscretizationError`.

## 5. Workflows

### TGB Datasets, Continuous-Time Temporal Graph Model

This is the simplest setup. Simply use `DGData.from_tgb()` to load the TGB dataset with its native time granularity.
By default, `batch_unit='r'` in the data loader so we can iterate by batches of 200 events with:

```python
from tgm import DGraph
from tgm.data import DGDataLoader, DGData

data = DGData.from_tgb('tgbl-wiki')
dg = DGraph(data)
loader = DGDataLoader(dg, batch_size=200)
```

### TGB Datasets, Discrete-Time Temporal Graph Model

In this case, we can still load the native time granularity for the given TGB dataset. However, we need to specify a valid `batch_unit` in our dataloader. Recall, that internally, this applies a `TimeDeltaDG` conversion, and therefore, our iterating batch unit must be coarser (or the same granularity) as the underlying graph time unit.

Here, we use `tgbl-wiki` which has second-wise data, and we iterate over it in weekly snapshots:

```python
from tgm import DGraph
from tgm.data import DGDataLoader, DGData

data = DGData.from_tgb('tgbl-wiki')
dg = DGraph(data)
loader = DGDataLoader(dg, batch_unit='W')
```

We can just as easily iterate over biweekly graph snapshots:

```python
from tgm import DGraph
from tgm.data import DGDataLoader, DGData

data = DGData.from_tgb('tgbl-wiki')
dg = DGraph(data)
loader = DGDataLoader(dg, batch_unit='W', batch_size=2)
```

### Custom Datasets with Known TimeDelta

When working with custom datasets, it's likely that you have an underlying time granularity as determined by your data feed. For instance, you may be streaming log events with unix timestamps, or have pre-aggregated data arriving daily from a cron job.

In this case pretty much the same workflow as above can be used. Just make sure to pass the right unit when constructing your `DGData.from_raw()`.
You may also be interested in discretizing your dataset into various granularities, and running some data analysis on the underlying graphs (e.g. figuring out number of nodes, edges, connected components etc).

### Custom Datasets with Unknown TimeDelta

It could occur that the underlying source time unit is not known a priori. In this situation, you can use the even-ordered time unit `TimeDeltaDG('r')` which preserves the relative order of events without assuming a specific time unit.

______________________________________________________________________

## Summary

The time delta is central to managing timestamps on your temporal graph. If using existing dataset (e.g. TGB), the time delta is already defined. For custom datasets, you need to provide either an event-ordered (`r`) or time-ordered (e.g. `s`) unit.

Time-ordered units are strictly more general in that they enable you to discretize your dataset, and iterate by temporal snapshots.
