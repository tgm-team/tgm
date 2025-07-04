# Constructing and Accessing Properties in DGraph

This tutorial will show you how to construct a `DGraph` object in `tgm` and accessing its properties.

The `DGraph` object can be found in `tgm/graph.py`

Let's first learn how to make a `DGraph`by defining the graph ourselves.

Start by defining the temporal edges, the `edge_index` has shape $\[num_edge_events, 2\]$

The `edge_timestamps` tensor specifies the timestamps associated with each edge, must has shape $\[num_edge_events\]$

The `edge_feats` specifies the features associated with each edge, must has shape $\[num_edge_events, D_edge\]$ where $D_edge$ is the feature dimension of edge. Edge features are also optional

```python
import torch

edge_index = torch.LongTensor([[2, 2], [2, 4], [1, 8]])
edge_timestamps = torch.LongTensor([1, 5, 20])
edge_feats = torch.rand(3, 5)

assert edge_index.shape[0] == edge_timestamps.shape[0] == edge_feats.shape[0]
```

## Construct `DGraph` from Pandas

There are three ways you can initialize `DGraph`: from Pandas dataframes, from csv or from raw tensor (least recommend).

Here is how to construct your `DGraph` from pandas dataframes. See the example below:

```python
import pandas as pd

edge_dict = {
    'src': [2, 10],
    'dst': [3, 20],
    't': [1337, 1338],
    'edge_feat': [torch.rand(5).tolist(), torch.rand(5).tolist()],
}  # edge events

node_dict = {
    'node': [7, 8],
    't': [3, 6],
    'node_feat': [torch.rand(5).tolist(), torch.rand(5).tolist()],
}  # node events, optional

data = DGData.from_pandas(
    edge_df=pd.DataFrame(edge_dict),
    edge_src_col='src',
    edge_dst_col='dst',
    edge_time_col='t',
    edge_feats_col='edge_feat',
    node_df=pd.DataFrame(node_dict),
    node_id_col='node',
    node_time_col='t',
    dynamic_node_feats_col='node_feat',
)

dgraph = DGraph(dgdata)

```

## Construct `DGraph` from raw tensors

To construct a `DGraph` object, we first bundles related tensors into a `DGData` which then constructs the underlying storage of the data.

`DGData` can be found at `tgm/data.py` and can be constructed with the `from_raw` function. In TGM, we support both edge events and node events at the same time. The core assumption is that a temporal graph must have edge events and with optional node events as well.

The edge events are specified with the following arguments:

- `edge_timestamps`: Tensor  # \[num_edge_events\] (required)

- `edge_index`: Tensor  # \[num_edge_events, 2\] (required)

- `edge_feats`: Tensor | None = None  # \[num_edge_events, D_edge\] (optional)

The node events are specified with the following arguments:

- `node_timestamps`: Tensor | None = None  # \[num_node_events\] (optional)

- `node_ids`: Tensor | None = None, # \[num_node_events\] (optional)

- `dynamic_node_feats`: Tensor | None = None  # \[num_node_events, D_node_dynamic\]

Lastly, each node can also have a static node feature, i.e. node feature that doesn't change over time

- `static_node_feats`: Tensor | None = None  # \[num_nodes, D_node_static\]

In the next part, we will specify our node events

```python
from tgm.data import DGData
from tgm.graph import DGraph


node_timestamps = torch.LongTensor([1, 5, 10])
node_ids = torch.LongTensor([2, 4, 6])
dynamic_node_feats = torch.rand([3, 5])

assert node_timestamps.shape[0] == node_ids.shape[0] == dynamic_node_feats.shape[0]

static_node_feats = torch.rand(9, 11)

dgdata = DGData.from_raw(
    edge_timestamps,
    edge_index,
    edge_feats,
    node_timestamps,
    node_ids,
    dynamic_node_feats,
    static_node_feats,
) # initializing our DGData

our_dgraph = DGraph(dgdata) # initializing our DGraph
```

## Properties of `DGraph`

`DGraph` objects act as a view on the underlying graph data, allowing the user to access various properties as well as slicing and other operations. Additional properties are seen in the `tgm/graph.py`

```python
print('=== Graph Properties ===')
print(f'start time : {our_dgraph.start_time}')
print(f'end time : {our_dgraph.end_time}')
print(f'number of nodes : {our_dgraph.num_nodes}')
print(f'number of edge events : {our_dgraph.num_edges}')
print(f'number of timestamps : {our_dgraph.num_timestamps}')
print(f'number of edge and node events : {our_dgraph.num_events}')
print(f'edge feature dim : {our_dgraph.edge_feats_dim}')
print(f'static node feature dim : {our_dgraph.static_node_feats_dim}')
print(f'dynamic node feature dim : {our_dgraph.dynamic_node_feats_dim}')
print('======================')
```
