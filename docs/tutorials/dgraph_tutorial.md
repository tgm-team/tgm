# Constructing and Accessing Properties in DGraph

This tutorial shows how to construct a `DGraph` object in `tgm` and explore its properties.

The `DGraph` class is defined in [`tgm/graph.py`](https://github.com/tgm-team/tgm/blob/main/tgm/graph.py).

______________________________________________________________________

## Construct `DGraph` from TGB Datasets

The [Temporal Graph Benchmark (TGB)](https://tgb.complexdatalab.com/) provides a suite of temporal graph datasets with diverse scales, properties, and tasks.

TGM supports constructing `DGraph` objects from TGB [LinkPropPrediction](https://tgb.complexdatalab.com/docs/linkprop/) and [NodePropPrediction](https://tgb.complexdatalab.com/docs/nodeprop/) datasets. Temporal knowledge graphs (TKG) and temporal hypergraphs (THG) are not yet supported.

You can specify the time granularity with [`TimeDeltaDG`](https://github.com/tgm-team/tgm/blob/main/tgm/timedelta.py). For example, `'r'` (relative) means timestamps define only the ordering of edges and are not used for time conversion. Other granularities allow unit-based conversions. The default time granularity is `'r'`, and the default device is `'cpu'`.

```python
from tgm import DGraph

train_dg = DGraph.from_tgb('tgbl-wiki', time_delta='r', split='train', device='cpu')
```

> **Note:** Time granularity features are experimental and may change.

______________________________________________________________________

## Create a Custom `DGraph`

You can also define a `DGraph` from your own data.

### Construct `DGraph` From Raw Tensors

#### Define Temporal Edges

- `edge_index`: shape `[num_edge_events, 2]`
- `edge_timestamps`: shape `[num_edge_events]`
- `edge_feats`: shape `[num_edge_events, D_edge]` (optional)

```python
import torch

edge_index = torch.LongTensor([[2, 2], [2, 4], [1, 8]])
edge_timestamps = torch.LongTensor([1, 5, 20])
edge_feats = torch.rand(3, 5)  # optional edge features
```

#### Define Node Events (Optional)

- `node_timestamps`: shape `[num_node_events]`
- `node_ids`: shape `[num_node_events]`
- `dynamic_node_feats`: shape `[num_node_events, D_node_dynamic]`
- `static_node_feats`: shape `[num_nodes, D_node_static]` (optional)

```python
node_timestamps = torch.LongTensor([1, 2, 3])
node_ids = torch.LongTensor([2, 4, 6])
dynamic_node_feats = torch.rand([3, 5])
static_node_feats = torch.rand(9, 11)
```

#### Construct the `DGraph`

```python
from tgm import DGraph

dg = DGraph.from_raw(
    edge_timestamps=edge_timestamps,
    edge_index=edge_index,
    edge_feats=edge_feats,
    node_timestamps=node_timestamps,
    node_ids=node_ids,
    dynamic_node_feats=dynamic_node_feats,
    static_node_feats=static_node_feats,
    time_delta='s',  # second-wise granularity
    device='cuda', # move graph to GPU
)
```

### Construct `DGraph` from Pandas DataFrames

```python
import pandas as pd

edge_df = pd.DataFrame({
    'src': [2, 2, 1],
    'dst': [2, 4, 8],
    't': [1, 5, 10],
    'edge_feat': [torch.rand(5).tolist() for _ in range(3)],
})

dynamic_node_df = pd.DataFrame({
    'node': [2, 4, 6],
    't': [1, 2, 3],
    'dynamic_node_feat': [torch.rand(5).tolist() for _ in range(3)],
})

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
    device='cuda', # move graph to GPU
)
```

### Construct `DGraph` from CSV Files

To load graph data from CSV files, use `DGraph.from_csv()`.
See [`tgm/graph.py`](https://github.com/tgm-team/tgm/blob/main/tgm/graph.py) for details.

______________________________________________________________________

## Accessing `DGraph` Properties

`DGraph` objects act as views over the underlying data. You can access properties and perform slicing operations.

The number of nodes is computed as `max(node_ids) + 1`. If the `DGraph` is empty, `start_time` and `end_time` are `None`.

```python
print('=== Graph Properties ===')
print(f'Start time                : {dg.start_time}')
print(f'End time                  : {dg.end_time}')
print(f'Number of nodes           : {dg.num_nodes}')
print(f'Number of edge events     : {dg.num_edges}')
print(f'Number of timestamps      : {dg.num_timestamps}')
print(f'Total events (edge+node)  : {dg.num_events}') # or len(dg)
print(f'Edge feature dimension    : {dg.edge_feats_dim}')
print(f'Static node feature dim   : {dg.static_node_feats_dim}')
print(f'Dynamic node feature dim  : {dg.dynamic_node_feats_dim}')
print('==========================')
```

______________________________________________________________________

## Slicing `DGraph`

You can slice temporal data using `slice_time()`. This returns a new `DGraph` containing only events within the specified time range (end time exclusive). Slicing is a lightweight operation since the underlying data storage is shared across `DGraph` instances.

```python
start_time, end_time = 5, 10
sliced_dg = dg.slice_time(start_time, end_time)
```

______________________________________________________________________

## Using `DGDataLoader` and Hooks

TGM integrates operations like **negative sampling** and **neighbor sampling** into the data loader via hooks.

```python
from tgm.loader import DGDataLoader
from tgm.hooks import NegativeEdgeSamplerHook, RecencyNeighborHook

neg_hook = NegativeEdgeSamplerHook(low=0, high=train_dg.num_nodes)

# Sample 20 1-hop neighbors per node
nbr_hook = RecencyNeighborHook(num_nbrs=[20], num_nodes=train_dg.num_nodes)

train_loader = DGDataLoader(
    train_dg,
    hook=[neg_hook, nbr_hook],
    batch_size=200,
)
```

You can also iterate over time windows (instead of event counts) by specifying a `TimeDeltaDG` in the data loader constructor. For this to work, the underlying graph must be non-ordered (e.g. time granularity `'s'`).

______________________________________________________________________

## Iterating Over Batches with `DGBatch`

Each batch from `DGDataLoader` is a `DGBatch`.

```python
iter_loader = iter(train_loader)
batch = next(iter_loader)

print('=== Batch of 200 edges ===')
print(f'Source nodes shape         : {batch.src.shape}')
print(f'Destination nodes shape    : {batch.dst.shape}')
print(f'Timestamps shape           : {batch.time.shape}')
print(f'Negative destinations shape: {batch.neg.shape}')
print(f'1-hop neighbors shape      : {batch.nbr_nids[1].shape}')
print('===========================')
```

The materialized batch will contain, at a minimum the batch of edges, features and nodes on the appropriate device. Hooks inject additional attributes at runtime (e.g. `batch.neg`).

______________________________________________________________________

## Feedback

Please feel free to reach out to us if anything is unclear or unintuitive. We are happy to discuss and improve your experience with TGM.
