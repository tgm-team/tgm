# Temporal Graph Data in TGM

This tutorial shows the **core graph API** in TGM. By the end, you should understand how to:

- Construct and preprocess graph data (`DGData`)
- Split and discretize temporal datasets (`SplitStrategy`)
- Work with immutable graph views (`DGraph`)
- Train with batches (`DGBatch`)

We also highlight some important errors, caching behaviour, and best practices.

______________________________________________________________________

## 1. The Core Objects

TGM's graph API revolves around four main objects:

| Object      | Description                                                   | Mutable | Device Semantics | Typical Usage                                    |
| ----------- | ------------------------------------------------------------- | ------- | ---------------- | ------------------------------------------------ |
| `DGData`    | Mutable bulk dataset storage (IO, splits, transforms)         | Yes     | No               | Ingesting datasets from disk, TGB, preprocessing |
| `DGraph`    | Immutable graph view backed by storage engine                 | No      | Yes              | Main user-facing graph object                    |
| `DGBatch`   | Materialized batches of tensors from a temporal slice of data | Yes     | Yes              | What dataloaders yield, input to models          |
| `DGStorage` | Internal backend for graph data (non-user-facing)             | No      | Yes              | Powers graph querying, caching, slice ops        |

> **Note**: Users typically only interact with the first 3. `DGStorage` is internal and abstracted away. It is in our stream of work to build out more efficient storage backends for various workloads in the future.

## 2. Starting with `DGData`

`DGData` is your *main* entry point for working with temporal graph datasets. It's a dataclass that holds bulk storage of events, timestamps, features, and metadata.

Because it's mutable, you can freely transform and prepare it before moving to the immutable graph representation (`DGraph`).

#### Features of `DGData`

- Holds raw edge data (`edge_index`, `edge_timestamps`)
- Holds *static node features*, *dynamic node features*, and *edge_features* (on CPU)
- Provides IO constructors (CSV, Pandas, TGB, pyTorch)
- Supports *temporal splitting* and *discretization*
- Ensures data is sorted chronologically, valid node ids, valid tensor shapes, etc.

See below for a summary of the data class attributes of `DGData`:

```python
@dataclass
class DGData:
    """Container for dynamic graph data to be ingested by `DGStorage`.

    Stores edge and node events, their timestamps, features, and optional split strategy.
    Provides methods to split, discretize, and clone the data.

    Attributes:
        time_delta (TimeDeltaDG | str): Time granularity of the graph.
        timestamps (Tensor): 1D tensor of all event timestamps [num_edge_events + num_node_events].
        edge_event_idx (Tensor): Indices of edge events within `timestamps`.
        edge_index (Tensor): Edge connections [num_edge_events, 2].
        edge_feats (Tensor | None): Optional edge features [num_edge_events, D_edge].
        node_event_idx (Tensor | None): Indices of node events within `timestamps`.
        node_ids (Tensor | None): Node IDs corresponding to node events [num_node_events].
        dynamic_node_feats (Tensor | None): Node features over time [num_node_events, D_node_dynamic].
        static_node_feats (Tensor | None): Node features invariant over time [num_nodes, D_node_static].

    Raises:
        InvalidNodeIDError: If an edge or node ID match `PADDED_NODE_ID`.
        ValueError: If any data attributes have non-well defined tensor shapes.
        EmptyGraphError: If attempting to initialize an empty graph.

    Notes:
        - Timestamps must be non-negative and sorted; DGData will sort automatically if necessary.
        - Cloning creates a deep copy of tensors to prevent in-place modifications.
    """
```

See [`tgm.data.DGData`](../api/data.md) for full reference.

## 3. Constructing DGData

You can build datasets in multiple ways. Let's look at each.

### 3.1 From TGB

This is most likely all you need. The [Temporal Graph Benchmark (TGB)](https://tgb.complexdatalab.com/) provides a suite of temporal graph datasets with diverse scales and properties. We natively support direct construction from all the `tgbl-` and `tgbn-` in TGM.

> **Note**: Temporal knowledge graph (TKG) and temporal hypergraph (THG) are not yet supported in TGM.

> **Note**: To load a TGB dataset, you must have the `py-tgb` package in your python env.

```python
from tgm import DGData

# Load the Wikipedia dataset from TGB
data = DGData.from_tgb('tgbl-wiki')

print(data.time_delta) # TimeDelta('s', value=1)
print(data.edge_index.shape) # torch.Size([157474, 2])
print(data.dynamic_node_feats) # None, no dynamic node features in tgbl-wiki
print(data.static_node_feats) # None, no static node features in tgbl-wiki
```

> **TIP**: You can `print(data)` to see which features and events exist within the dataset.

### 3.2 Custom Datasets

If you have our own dataset in TGM, you can create a `DGData` object either `from_csv`, `from_pandas`, or directly from tensors. A brief overview of each is given below, consult the API reference for more details.

#### From CSV

Please consult our documentation for full description of our API. The table below summarizes the main pieces of data expected during construction. Note that analogous attributes are expected in the other IO constructors (e.g. `from_pandas`, `from_raw`)

| Attribute                     | Description                                                    | Type                  | Required                                              | Note                                                     |
| ----------------------------- | -------------------------------------------------------------- | --------------------- | ----------------------------------------------------- | -------------------------------------------------------- |
| `edge_file_path`              | Path to CSV file containing edge data                          | `str \| pathlib.Path` | Yes                                                   | `edge_df` if using `from_pandas`                         |
| `edge_src_col`                | Column name in edge file for src nodes                         | `str`                 | Yes                                                   | Cannot have ids matching `tgm.constants.PADDED_NODE_ID`  |
| `edge_dst_col`                | Column name in edge file for dst nodes                         | `str`                 | Yes                                                   | Cannot have ids matching  `tgm.constants.PADDED_NODE_ID` |
| `edge_time_col`               | Column name in edge file for edge times                        | `str`                 | Yes                                                   | Time must be non-negative                                |
| `node_file_path`              | Path to CSV file containing dynamic node data                  | `str \| pathlib.Path` | No                                                    | `node_df` is using `from_pandas`                         |
| `node_id_col`                 | Column name in node file for node event node ids               | `str`                 | No, unless `node_file_path` is specified              | Cannot have ids matching  `tgm.constants.PADDED_NODE_ID` |
| `node_time_col`               | Column name in node file for node event node times             | `str`                 | No, unless `node_file_path` is specified              | Time must be non-negative                                |
| `dynamic_node_feats_col`      | Column name in node file for dynamic node features             | `str`                 | No                                                    |                                                          |
| `static_node_feats_file_path` | Path to CSV file containing static node features               | `str \| pathlib.Path` | No                                                    | `static_node_feats_df` if using `from_pandas`            |
| `static_node_feats_col`       | Column name in static node feats file for static node features | `str`                 | No, unless `static_node_feats_file_path` is specified |                                                          |
| `time_delta`                  | Time granularity of the graph data                             | `TimeDeltaDG \| str`  | Yes                                                   | Default to *event_ordered* granularity `'r'`             |

A few key things to know:

- `time_delta`: defines how timestamps are interpreted on your custom dataset.
  - The default is 'r' which entails event-ordered semantics. This means there is no real-world time unit assigned to your timestamps. This prevents from doing things like discretizing your data, and iterating by temporal snapshots.
  - More often than not, your timestamps have some semantics meaning (e.g. *seconds*, *days*, etc). In this case, you should specify the appropriate `time_delta` value. See our [time management tutorial](../tutorials/time_delta_tutorial.md) for more details.
- edge data:
  - We expect an `edge_file_path` which is a csv file with `edge_src_col`, `edge_dst_col`, `edge_time_col` as a minimum.
  - Your edge csv file may also contain `edge_feats_col` which are the edge features on your data
- dynamic node data (optional)
  - If included, we expect a `node_file_path` which is a csv file with `node_id_col`, `node_time_col` as a minimum. These are your dynamic node events.
  - Your dynamic node data csv file may also include `dynamic_node_feats_col`, which are the dynamic node features in your data.
- static node data (optional)
  - If included, we expect a `static_node_feats_fil_path` which is a csv file with `static_node_feats_col`, the static node features for your dataset.

Internally, we perform various checks on the tensors shapes, node ranges, and timestamps values. If your data is well structured, everything should work. If you get an error message that is not intuitive, please let us know.

#### From Pandas

The API largely the same as above, except that we expected `edge_df`, `node_df`, and `static_node_feats_df` dataframes for the edge, dynamic node, and static node data respectively, instead of csv files.

```python
import pandas as pd

# Define Edge Data
edge_df = pd.DataFrame({
    'src': [2, 2, 1],
    'dst': [2, 4, 8],
    't': [1, 5, 10],
    'edge_feat': [torch.rand(5).tolist() for _ in range(3)], # Optional
})

# Define Dynamic Node Data (Optional)
dynamic_node_df = pd.DataFrame({
    'node': [2, 4, 6],
    't': [1, 2, 3],
    'dynamic_node_feat': [torch.rand(5).tolist() for _ in range(3)],
})

# Define Static Node Features (Optional)
static_node_df = pd.DataFrame({
    'static_node_feat': [torch.rand(11).tolist() for _ in range(9)]
})

dg = DGraph.from_pandas(
    edge_df=edge_df,
    edge_src_col='src',
    edge_dst_col='dst',
    edge_time_col='t',
    edge_feats_col='edge_feat',
    node_df=dynamic_node_df,
    node_id_col='node',
    node_time_col='t',
    dynamic_node_feats_col='dynamic_node_feat',
    static_node_feats_df=static_node_df,
    static_node_feats_col='static_node_feat',
    time_delta='s',  # second-wise granularity
)
```

#### From Tensors

If all your data is already in memory as `torch.Tensor` you can directly instantiate `DGdata` using the class method `DGData.from_raw`:

```python
import torch

# Define Edge Data
edge_index = torch.LongTensor([[2, 2], [2, 4], [1, 8]])
edge_timestamps = torch.LongTensor([1, 5, 20])
edge_feats = torch.rand(3, 5)  # optional edge features

# Define Dynamic Node Data (Optional)
node_timestamps = torch.LongTensor([1, 2, 3])
node_ids = torch.LongTensor([2, 4, 6])
dynamic_node_feats = torch.rand([3, 5])

# Define Static Node Features (Optional)
static_node_feats = torch.rand(9, 11)

data = DGData.from_raw(
    edge_timestamps=edge_timestamps,
    edge_index=edge_index,
    edge_feats=edge_feats,
    node_timestamps=node_timestamps,
    node_ids=node_ids,
    dynamic_node_feats=dynamic_node_feats,
    static_node_feats=static_node_feats,
    time_delta='s',  # second-wise granularity
)
```

### 3.3 Errors to know

- `tgm.exceptions.EmptyGraphError`: Raised when you try to construct a `DGData` object from empty data. This is probably not what you intended to do since downstream `DGraph` is immutable.
- `tgm.exceptions.InvalidNodeIDError`: Raised when you dataset contains `-1` as a node ID (reserved for padding).

## 4. Splitting `DGData`

After loading your data, you'll probably want to split your dataset into *train*, *validation*, and *test* splits. TGM provides a **strategy pattern** interface for different split strategies:

- `TemporalSplit`: Split by fixed timestamp boundaries
- `TemporalRatioSplit`: Split by ratio of both edge and node events
- `TGBSplit`: Pre-defined TGB data splits

> **Important**: The TGB data splits uses pre-defined event masks, to match the splits as per the TGB leaderboard. If you try to change this, you'll get a `ValueError`.

The split method is defined on `DGData`:

```python
def split(self, strategy: SplitStrategy | None = None) -> Tuple[DGData, ...]:
    """Split the dataset according to a strategy.

    Args:
        strategy (SplitStrategy | None): Optional strategy to override the
            default. If None, uses `_split_strategy` or defaults to `TemporalRatioSplit`.

    Returns:
        Tuple[DGData, ...]: Split datasets (train/val/test).

    Raises:
        ValueError: If attempting to override the split strategy for TGB datasets.

    Notes:
        - Splits preserve the underlying storage; only indices are filtered.
    """
```

### Splitting TGB Datasets

```python
from tgm import DGData

# Load the Wikipedia dataset from TGB
data = DGData.from_tgb('tgbl-wiki')

# Split using native TGB masks
train_data, val_data, test_data = data.split()

# If you tried to override the split strategy, you'll get an error

from tgm.split import TemporalRatioSplit
split_strategy = TemporalRatioSplit(train=0.8, val=0.1, test=0.1)
_ data.split(strategy=split_strategy) # Raises ValueError
```

## 5. Discretizing `DGData`

In TGM, we do not enforce strict definition of continuous time (resp. discrete time) dynamic graph CTDG (resp. DTDG). Instead, as you have seen, we define graphs based on their time granularity. Therefore, the user is able to convert between event-based and snapshot based views of the underlying data. You can learn more about this in [the UTG paper](https://arxiv.org/abs/2407.12269).

In TGM, we provide a method on `DGData` called `discretize` which allows you to coarsen your graph into different time granularities. The API looks like:

```python
def discretize(
    self, time_delta: TimeDeltaDG | str | None, reduce_op: str = 'first'
) -> DGData:
    """Return a copy of the dataset discretized to a coarser time granularity.

    Args:
        time_delta (TimeDeltaDG | str | None): Target time granularity.
        reduce_op (str): Aggregation method for multiple events per bucket. Default 'first'.

    Returns:
        DGData: New dataset with discretized timestamps and features.

    Raises:
        EventOrderedConversionError: If discretization is incompatible with event-ordered granularity
        InvalidDiscretizationError: If the target granularity is finer than the current granularity.
    """
```

> **Note**: This is only well defined if the DGData time delta is *time-ordered*. If you try discretizing an event-ordered dataset, you will get a `tgm.exceptions.EventOrderedConversionError`.

> **Note**: Discretization goes from finer time units (e.g. seconds) to coarse time units (e.g. hours). If your attempt to discretize in the other direction, you'll get a `tgm.exceptions.InvalidDiscretizationError`.

See our [time management tutorial](../tutorials/time_delta_tutorial.md) for more details on discretization and how it relates to `TimeDeltaDG`.

## 6. From `DGData` to `DGraph`

Once your dataset is ready to go, you can cast it to `DGraph`:

```python
from tgm import DGraph, DGData

data = DGData.from_tgb(...)
dg = DGraph(data, device=...)
```

Some things to note:

- `DGraph` is an immutable view over a temporal window of graph data.
- It is backed by `DGStorage` (internal engine). When you first create a `DGraph` as we did above, a new storage is created, and the view encapsulates the entire dataset.
- `DGraph` supports device semantics, you can choose what device your graph is on.

### DGraph Properties

Let's use our toy `DGData` we had above, cast to `DGraph` and inspect some of the properties of the entire dataset.

```python
data = DGData.from_raw(...) # As we had above
dg = DGraph(data) # Default to CPU

print(f'Start time                : {dg.start_time}') # 1
print(f'End time                  : {dg.end_time}') # 10
print(f'Number of nodes           : {dg.num_nodes}') # 9
print(f'Number of edge events     : {dg.num_edges}') # 3
print(f'Number of timestamps      : {dg.num_timestamps}') # or len(dg); 5
print(f'Total events (edge+node)  : {dg.num_events}') # 6
print(f'Edge feature dimension    : {dg.edge_feats_dim}') # 5
print(f'Static node feature dim   : {dg.static_node_feats_dim}') # 11
print(f'Dynamic node feature dim  : {dg.dynamic_node_feats_dim}') # 5

print(f'TimeDelta                 : {dg.time_delta}') # TimeDelta('s', value=1)
print(f'Device                    : {dg.device}') # torch.device(cpu)

# We can move the graph to GPU
dg = dg.to('cuda')
print(f'Device                    : {dg.device}') # torch.device(cuda:0)
```

> **Note**: The number of nodes is computed as `max(node_ids) + 1`.

> **Note**: If the `DGraph` is empty, `start_time` and `end_time` are `None`.

> **Note**: `len()` returns the number of timestamps (not the number of events) in the graph.

### Slicing: Creating new views

You can create a new `DGraph` view by slicing the underlying data. Currently, we support slicing by time, or by event index. Both operations are lightweight, as the storage is shared between `DGraph` instances. This makes it very fast to select subsets of your data.

You can slice temporal data using `slice_time()`. This returns a new `DGraph` containing only events within the specified time range (end time exclusive). Slicing is a lightweight operation since the underlying data storage is shared across `DGraph` instances.

> **Note**: These are both end-time *exclusive* operations.

Following from our previous code snippet:

```python
sliced_dg = dg.slice_time(start_time=5, end_time=10)
print(sliced_dg.start_time) # 5
print(sliced_dg.end_time) # 9, end time exclusive
print(sliced_dg.num_edges) # 1
print(sliced_dg.device) # still on gpu
```

## 7. Materialization, Iteration and `DGBatch`

In practice, the typical workflow will require you to feed data into your model for training. For this purpose, we need to *materialize* the view.

The method on `DGraph` looks like:

```python
def materialize(self, materialize_features: bool = True) -> DGBatch:
    """Materialize the current DGraph slice into a dense `DGBatch`.

    Args:
        materialize_features (bool, optional): If True, includes dynamic node
            features, node IDs/times, and edge features. Defaults to True.

    Returns:
        DGBatch: A batch containing src, dst, timestamps, and optionally
            features from the current slice.
    """
```

As described above, the output is a `DGBatch` object, which is nothing but a container of tensors corresponding to the materialized data of the `DGraph`, on device. By default, the `DGBatch` contains the following attributes:

```python
@dataclass
class DGBatch:
    """Container for a batch of events/materialized data from a DGraph.

    Each `DGBatch` holds edge and node information for a slice of a dynamic graph,
    including optional dynamic node features and edge features. Hooks read and write
    additional attributes to the container transparently during dataloading.

    Args:
        src (Tensor): Source node indices for edges in the batch. Shape `(E,)`.
        dst (Tensor): Destination node indices for edges in the batch. Shape `(E,)`.
        time (Tensor): Timestamps of each edge event. Shape `(E,)`.
        dynamic_node_feats (Tensor | None, optional): Dynamic node features for nodes
            in the batch. Typically sparse tensor of shape `(T x V x d_node_dynamic)`.
        edge_feats (Tensor | None, optional): Edge features for the batch. Typically
            sparse tensor of shape `(E x d_edge)` or `(T x V x V x d_edge)` depending
            on storage.
        node_times (Tensor | None, optional): Timestamps corresponding to dynamic node features.
        node_ids (Tensor | None, optional): Node IDs corresponding to dynamic node features.
    """
```

For example:

```python

# Our full graph view
dg_batch = dg.materialize(materialize_features=False) # Skip features
print(dg_batch.src) # torch.tensor([2, 2, 1], dtype=torch.long, device='cuda:0')
print(dg_batch.edge_feats) # None, because we skipped materializing features

# Our sliced graph view (from start_time=5, end_time=10)
sliced_dg_batch = sliced_dg.materialize()
print(dg_batch.src) # torch.tensor([5], dtype=torch.long, device='cuda:0')
print(dg_batch.edge_feats is None) # False, we matrialized our slice of edge features
```

> **Note**: Materializing a full graph view with features could be expensive, especially on large graphs.
> **Note**: The device of `DGraph` determines the device on which the `DGBatch` tensors are allocated.

### DGDataLoader

Internally, the `DGDataLoader` is responsible for materializing slices of graph data, using exactly the mechanics describe above. In particular, when you do something like:

```python
from tgm import DGraph
from tgm.loader import DGDataLoader

dg = DGraph(...)
loader = DGDataLoader(dg, ...)

for batch in loader:
    ...
```

the data loader computes offsets into the storage, performs slicing operations, materializes the sliced views, and the applies hooks on the materialized data. See our [hook management tutorial](../tutorials/hook_tutorial.md) for more details.

______________________________________________________________________

## Summary

We learned about how `DGData` is used for loading data and preprocessing. We discussed how to created data splits and discretize your dataset to coarser time granularities. Once your data is loaded, you cast to `DGraph`, which is an immutable view of a slice of data.We showed how to query various attributes from a `DGraph`, and how to slice the `DGraph` in temporal snapshots. Finally, we showed how to *materialize* the data in `DGBatch` for training, and how the `DGDataLoader` does this internally during iteration.

With this foundation, you're ready to explore [hook management](../tutorials/hook_tutorial.md) and get started with our examples. Please feel free to reach out to us if anything is unclear or unintuitive. We are happy to discuss and improve your experience with TGM.
