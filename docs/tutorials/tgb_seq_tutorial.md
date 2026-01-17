# TGB-Seq tutorial

[`TGB-Seq`](https://arxiv.org/abs/2502.02975) is supported in `TGM`. This tutorial provides an overview of how to set up and run experiments on `TGB-Seq`.

## Running Pre-packaged examples

TGM includes pre-packaged example scripts to help you get started quickly. The examples require extra dependencies, including `TGB-Seq`, beyond the core library.

```
pip install -e .[examples]
```

*Please advise [TGB-Seq Github](https://github.com/TGB-Seq/TGB-Seq) for further information.*

After installing the dependencies, you can run any supported model on any `TGB-Seq` benchmark dataset. For instance, `EdgeBank` dynamic link prediction on [`GoogleLocal`](https://tgb-seq.github.io/datasets/#googlelocal):

```
python examples/linkproppred/tgb_seq/edgebank.py --dataset GoogleLocal --device cuda
```

To view the full list of available datasets, please visit the [`TGB-Seq` dataset page.](https://tgb-seq.github.io/datasets/)

> \[!NOTE\]
> By default, our link prediction examples on `TGB-Seq` default to `GoogleLocal`.
> Examples run on CPU by default; use the `--device` flag to override this as shown above.

## `TGB-Seq` dataload

`TGM` provides data loading I/O support for `TGB-Seq`. A `TGB-Seq` dataset can be loaded as follows:

```python
full_data= DGData.from_tgb_seq(
    'GoogleLocal', root='./data'
)

train_data, val_data, test_data = full_data.split()
```

Unlike `TGB`, which downloads dataset to `.env` by default, `TGB-Seq` requires you to explicitly specify the download destination (in this example, `./data`).

## `TGB-Seq` custom negative sampler hook

To evaluate to TGB-Seq on link prediction, we need a custom hook from `TGB-Seq` to sample negative edges:

```python
from tgm.hooks import StatelessHook

class TGBSEQ_NegativeEdgeSamplerHook(StatelessHook):
    produces = {'neg', 'neg_time'}

    def __init__(
        self, dataset_name: str, split_mode: str, dgraph: DGraph, root: str = './data'
    ) -> None:
        self.has_precomputed_negatives = split_mode == 'test'

        if self.has_precomputed_negatives:
            from tgb_seq.LinkPred.dataloader import TGBSeqLoader

            self.negs = torch.from_numpy(
                TGBSeqLoader(dataset_name, root=root).negative_samples
            )
            self.neg_idx = 0
        else:
            edge_dst = dgraph.edge_dst
            self.low, self.high = int(edge_dst.min()), int(edge_dst.max())
            self.num_negs = 100

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        batch_size = len(batch.edge_src)

        if self.has_precomputed_negatives:
            batch.neg = self.negs[self.neg_idx : self.neg_idx + batch_size]
            self.neg_idx += batch_size
        else:
            size = (self.num_negs, batch_size)
            batch.neg = torch.randint(
                self.low, self.high, size, dtype=torch.int32, device=dg.device
            )

        batch.neg_time = batch.edge_time.clone()
        return batch
```

## Minimal example

Here’s a basic example demonstrating how to run `EdgeBank` for dynamic link property prediction on `GoogleLocal`:

```python
import numpy as np
import torch
from tgb_seq.LinkPred.evaluator import Evaluator
from tqdm import tqdm

from tgm import DGBatch, DGraph
from tgm.data import DGData, DGDataLoader
from tgm.hooks import HookManager
from tgm.nn import EdgeBankPredictor


def eval(
    loader: DGDataLoader,
    model: EdgeBankPredictor,
    evaluator: Evaluator,
) -> float:
    perf_list = []
    for batch in tqdm(loader):
        negs_per_pos = len(batch.neg)

        for idx in range(negs_per_pos):
            query_src = batch.edge_src[idx].repeat(negs_per_pos + 1)
            query_dst = torch.cat([batch.edge_dst[idx].unsqueeze(0), batch.neg[idx]])

            y_pred = model(query_src, query_dst)
            y_pred_pos, y_pred_neg = y_pred[0].unsqueeze(0), y_pred[1:]
            perf_list.append(evaluator.eval(y_pred_pos, y_pred_neg))
        model.update(batch.edge_src, batch.edge_dst, batch.edge_time)

    return float(np.mean(perf_list))

evaluator = Evaluator()

train_data, val_data, test_data = DGData.from_tgb_seq(
    'GoogleLocal', root='./data'
).split()
train_dg = DGraph(train_data)
test_dg = DGraph(test_data)

edge_dst = test_dg.edge_dst
low, high = int(edge_dst.min()), int(edge_dst.max())

hm = HookManager(keys=['test'])
hm.register(
    'test',
    TGBSEQ_NegativeEdgeSamplerHook(
        'GoogleLocal', split_mode='test', dgraph=test_dg, root='./data'
    ),
)

test_loader = DGDataLoader(test_dg, batch_size=200, hook_manager=hm, drop_last=True)

train_data = train_dg.materialize(materialize_features=False)
model = EdgeBankPredictor(
    train_data.edge_src,
    train_data.edge_dst,
    train_data.edge_time,
)


with hm.activate('test'):
    test_mrr = eval(test_loader, model, evaluator)
    log_metric(f'Test {METRIC_TGB_LINKPROPPRED}', test_mrr)

```

## References

1. [TGB-Seq Benchmark: Challenging Temporal GNNs with Complex Sequential Dynamics.](https://openreview.net/forum?id=8e2LirwiJT)
