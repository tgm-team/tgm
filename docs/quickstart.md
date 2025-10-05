# Quick Start Guide

### Minimal Example

Here’s a basic example demonstrating how to train TGCN for dynamic node property prediction on `tgbl-trade`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from tgm import DGraph, DGBatch
from tgm.data import DGData, DGDataLoader
from tgm.nn import TGCN, NodePredictor

# Load TGB data splits
train_data, val_data, test_data = DGData.from_tgb("tgbn-trade").split()

# Construct a DGraph and setup iteration by yearly ('Y') snapshots
train_dg = DGraph(train_data)
train_loader = DGDataLoader(train_dg, batch_unit="Y")

# tgbl-trade has no static node features, so we create Gaussian ones (dim=64)
static_node_feats = torch.randn((train_dg.num_nodes, 64))

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.recurrent = TGCN(in_channels=node_dim, out_channels=embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(
        self, batch: DGBatch, node_feat: torch.tensor, h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_index = torch.stack([batch.src, batch.dst], dim=0)
        h_0 = self.recurrent(node_feat, edge_index, H=h)
        z = F.relu(h_0)
        z = self.linear(z)
        return z, h_0

# Initialize our model and optimizer
encoder = RecurrentGCN(node_dim=static_node_feats.shape[1], embed_dim=128)
decoder = NodePredictor(in_dim=128, out_dim=train_dg.dynamic_node_feats_dim)
opt = torch.optim.Adam(set(encoder.parameters()) | set(decoder.parameters()), lr=0.001)

# Training loop
h_0 = None
for batch in train_loader:
    opt.zero_grad()
    y_true = batch.dynamic_node_feats
    if y_true is None:
        continue

    z, h_0 = encoder(batch, static_node_feats, h_0)
    z_node = z[batch.node_ids]
    y_pred = decoder(z_node)

    loss = F.cross_entropy(y_pred, y_true)
    loss.backward()
    opt.step()
    h_0 = h_0.detach()
```

### Running Pre-packaged Examples

TGM includes pre-packaged example scripts to help you get started quickly. The examples require extra dependencies beyond the core library.

```sh
pip install -e .[examples]
```

After installing the dependencies, you can run any of our examples. For instance, TGAT dynamic link prediction on `tgbl-wiki`:

```sh
python examples/linkproppred/tgat.py --dataset tgbl-wiki --device cuda
```

> \[!NOTE\]
> By default, our link prediction examples default to `tgbl-wiki`, and node prediction use `tgbn-trade`.
> Examples run on CPU by default; use the `--device` flag to override this as shown above.

### Next steps

- Explore more of our [examples](../examples/)
- Dive deeper into TGM with our [tutorials](./tutorials/)
