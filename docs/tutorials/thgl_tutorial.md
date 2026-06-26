# Temporal Heterogeneous Graphs tutorial

Temporal heterogeneous graphs are dynamic graphs in which each node and each relation between nodes is associated with a specific node type and edge type, respectively. They provide a practical framework for modeling real-world systems, such as social networks with multiple interaction types evolving over time, as well as communication and event streams. This tutorial aims to provide an overview of how to work with temporal heterogeneous graphs in `TGM`.

## 1. Constructing temporal heterogeneous graph data

`TGM` provides different APIs to constructing temporal heterogeneous graph data:

- Load temporal heterogeneous graph benchmark dataset from `TGB 2.0`
- Load custom temporal heterogeneous graph dataset from Tensors, Pandas and CSV

Temporal heterogeneous graph data is loaded and constructed with `DGGraph` APIs. For detailed tutorial about `DGGraph`, please see [`dgraph_tutorial.md`](./dgraph_tutorial.md)

### 1.1 `THGL` from TGB 2.0

`THGL` from `TGB 2.0` is a great starting point and most likely all you need. The [Temporal Graph Benchmark 2.0 (TGB)](https://openreview.net/forum?id=EADRzNJFn1#discussion) introduces a set of temporal heterogeneous graphs with diverse scales and properties.

```python
data = DGData.from_tgb('thgl-software')
min_node = data.edge_index.min().int()
max_node = data.edge_index.max().int()

print(data.time_delta) # TimeDelta('s', value=1)
print(data.edge_index.shape) # torch.Size([1489806, 2])
print(data.edge_type.shape) # torch.Size([1489806])
print(data.static_node_x) # None, no static node features in thgl-software

print(min_node) # 0
print(max_node) # 681926
print(data.node_x) # torch.Size([681927])
```

> **TIP**: You can `print(data)` to see which features and events exist within the dataset.

To evaluate model performance on validation and test set, `TGBTHGNegativeEdgeSamplerHook` is needed to load negative edges.

```python
hm = HookManager(keys=['test'])
hm.register(
    'test',
    TGBTHGNegativeEdgeSamplerHook(
        'thgl-software',
        split_mode='test',
        first_node_id=min_node, # The minimum node ID of the whole graph
        last_node_id=max_node, # The maximum node ID of the whole graph
        node_type=data.node_type, # Type of each node in the graph
    ),
)
```

*For detailed tutorial on Hooks and Hooks Manager, please check out [Hook Tutorial](./hook_tutorial.md).*

If you have our own dataset in TGM, you can create a `DGData` object either `from_csv`, `from_pandas`, or directly from tensors. A brief overview of each is given below, consult the API reference for more details.

### 1.2 From CSV

To load temporal heterogeneous graph from CSV, we can use `from_csv` API from `DGData` as described in [DGraph Tutorial](./dgraph_tutorial.md#from-csv) with additional attributes as follows:

| Attribute       | Description                                    | Type  |
| --------------- | ---------------------------------------------- | ----- |
| `edge_type_col` | Column name in edge file for type of each edge | `str` |
| `node_type_col` | Column name in node file for type of each node | `str` |

Internally, we perform various checks on the tensors shapes, node ranges, and timestamps values. If your data is well structured, everything should work. If you get an error message that is not intuitive, please let us know!

### 1.3 From Pandas

Similarly, to load temporal heterogeneous graph from Pandas, we can use `from_pandas` API from `DGData` as described in [DGraph Tutorial](./dgraph_tutorial.md#from-pandas) with additional attributes as described above.

```python
import pandas as pd

# Define Edge Data
edge_df = pd.DataFrame({
    'src': [2, 2, 1],
    'dst': [2, 4, 8],
    't': [1, 5, 10],
    'edge_type' : [0,1,0],
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
    'static_node_feat': [torch.rand(11).tolist() for _ in range(9)],
    'node_type' : torch.arange(9)
})

dg = DGData.from_pandas(
    edge_df=edge_df,
    edge_src_col='src',
    edge_dst_col='dst',
    edge_time_col='t',
    edge_x_col='edge_feat',
    node_df=dynamic_node_df,
    node_x_nids_col='node',
    node_x_time_col='t',
    node_x_col='dynamic_node_feat',
    static_node_x_df=static_node_df,
    static_node_x_col='static_node_feat',
    time_delta='s',  # second-wise granularity
    edge_type_col='edge_type',
    node_type_col='node_type'
)
```

### 1.4 From Tensor

If all your data is already in memory as `torch.Tensor` you can directly instantiate `DGdata` using the class method `DGData.from_raw`:

```python
import torch

# Define Edge Data
edge_index = torch.LongTensor([[2, 2], [2, 4], [1, 8]])
edge_time = torch.LongTensor([1, 5, 20])
edge_feats = torch.rand(3, 5)  # optional edge features
edge_type = torch.LongTensor([0,1,0])

# Define Dynamic Node Data (Optional)
node_x_time = torch.LongTensor([1, 2, 3])
node_x_nids = torch.LongTensor([2, 4, 6])
node_x = torch.rand([3, 5])
node_type = torch.arange(9)

# Define Static Node Features (Optional)
static_node_x = torch.rand(9, 11)

data = DGData.from_raw(
    edge_time=edge_time,
    edge_index=edge_index,
    edge_x=edge_feats,
    node_x_time=node_x_time,
    node_x_nids=node_x_nids,
    node_x=node_x,
    static_node_x=static_node_x,
    time_delta='s',  # second-wise granularity,
    edge_type=edge_type,
    node_type=node_type
)
```

## 2. Minimal example with `THGL`

Here’s a basic example demonstrating how to run `EdgeBank` for dynamic link property prediction on `thgl-software`:

```python

import numpy as np
import torch
from tgb.linkproppred.evaluate import Evaluator
from tqdm import tqdm

from tgm import DGraph
from tgm.constants import METRIC_TGB_LINKPROPPRED
from tgm.data import DGData, DGDataLoader
from tgm.hooks import HookManager, TGBTHGNegativeEdgeSamplerHook
from tgm.nn import EdgeBankPredictor

def eval(
    loader: DGDataLoader,
    model: EdgeBankPredictor,
    evaluator: Evaluator,
) -> float:
    perf_list = []
    for batch in tqdm(loader):
        for idx, neg_batch in enumerate(batch.neg_batch_list):
            query_src = batch.edge_src[idx].repeat(len(neg_batch) + 1)
            query_dst = torch.cat([batch.edge_dst[idx].unsqueeze(0), neg_batch])

            y_pred = model(query_src, query_dst)
            input_dict = {
                'y_pred_pos': y_pred[0],
                'y_pred_neg': y_pred[1:],
                'eval_metric': [METRIC_TGB_LINKPROPPRED],
            }
            perf_list.append(evaluator.eval(input_dict)[METRIC_TGB_LINKPROPPRED])
        model.update(batch.edge_src, batch.edge_dst, batch.edge_time)

    return float(np.mean(perf_list))


evaluator = Evaluator(name='')

data = DGData.from_tgb('thgl-software')
min_node = data.edge_index.min().int()
max_node = data.edge_index.max().int()

train_data, val_data, test_data = data.split()
train_dg = DGraph(train_data)
val_dg = DGraph(val_data)
test_dg = DGraph(test_data)

train_data = train_dg.materialize(materialize_features=False)

hm = HookManager(keys=['test'])
hm.register(
    'test',
    TGBTHGNegativeEdgeSamplerHook(
        'thgl-software',
        split_mode='test',
        first_node_id=min_node,
        last_node_id=max_node,
        node_type=data.node_type,
    ),
)

test_loader = DGDataLoader(test_dg, 200, hook_manager=hm)

model = EdgeBankPredictor(
    train_data.edge_src,
    train_data.edge_dst,
    train_data.edge_time,
)


with hm.activate('test'):
    test_mrr = eval(test_loader, model, evaluator)
```

A complete example can be viewed at [examples/thgl/edgebank.py](https://github.com/tgm-team/tgm/blob/main/examples/linkproppred/thgl/edgebank.py)

## References

1. [TGB 2.0: A Benchmark for Learning on Temporal Knowledge Graphs and Heterogeneous Graphs](https://openreview.net/forum?id=EADRzNJFn1#discussion)
