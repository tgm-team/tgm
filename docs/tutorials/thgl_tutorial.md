# Temporal Heterogeneous Graphs tutorial

Temporal heterogeneous graphs are dynamic graphs in which each node and each relation between nodes is associated with a specific node type and edge type, respectively. They provide a practical framework for modeling real-world systems, such as social networks with multiple interaction types evolving over time, as well as communication and event streams. This tutorial aims to provide an overview of how to work with temporal heterogeneous graphs in `TGM`.

## 1. Constructing temporal heterogeneous graph data

`TGM` supports different APIs to constructing temporal heterogeneous graph data:

- Load temporal heterogeneous graph benchmark dataset from `TGB 2.0`
- Load custom temporal heterogeneous graph dataset from Tensors, Pandas and CSV

### 1.1 `THGL` from TGB 2.0

### 1.2 From Pandas

### 1.3 From CSV

## 2. Minimal example with `THGL`

Here’s a basic example demonstrating how to experiment `EdgeBank` for dynamic link property prediction on `thgl-software`:

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

A complete example can be viewed at [examples/thgl/edgebank.py](../tgm/examples/thgl/edgebank.py)

## References

1. [TGB 2.0: A Benchmark for Learning on Temporal Knowledge Graphs and Heterogeneous Graphs](https://openreview.net/forum?id=EADRzNJFn1#discussion)
