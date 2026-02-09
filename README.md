<a id="readme-top"></a>

![image](./docs/img/logo.svg)

<div align="center">
<h3 style="font-size: 22px">Efficient and Modular ML on Temporal Graphs</h3>
<a href="https://tgm.readthedocs.io/en/latest"/><strong style="font-size: 18px;"/>Read Our Docs»</strong></a>
<a href="https://arxiv.org/abs/2510.07586"/><strong style="font-size: 18px;"/>Read Our Paper»</strong></a>
<br/>
<br/>

[![Stars](https://img.shields.io/github/stars/tgm-team/tgm?style=flat&label=Stars&labelColor=white&logo=github&logoColor=black)](https://github.com/tgm-team/tgm/stargazers)
[![PyPI](https://img.shields.io/pypi/v/tgm-lib?style=flat&label=PyPI&labelColor=white&logo=pypi&logoColor=black)](https://pypi.org/project/tgm-lib/)
[![Downloads](https://img.shields.io/pypi/dm/tgm-lib?style=flat&label=Downloads&labelColor=white&logo=pypi&logoColor=black)](https://pypi.org/project/tgm-lib/)
[![Tests](https://img.shields.io/github/actions/workflow/status/tgm-team/tgm/testing.yml?label=Tests&style=flat&labelColor=white&logo=github-actions&logoColor=black)](https://github.com/tgm-team/tgm/actions/workflows/testing.yml)
[![Docs](https://img.shields.io/readthedocs/tgm?style=flat&label=Docs&labelColor=white&logo=readthedocs&logoColor=black)](https://tgm.readthedocs.io/en/latest/?badge=latest)
[![Coverage](https://img.shields.io/codecov/c/github/tgm-team/tgm?style=flat&label=Coverage&labelColor=white&logo=codecov&logoColor=black)](https://codecov.io/gh/tgm-team/tgm)

</div>

## About The Project

TGM is a research library for temporal graph learning, designed to accelerate training on dynamic graphs while enabling rapid prototyping of new methods.
It provides a unified abstraction for both discrete and continuous-time graphs, supporting diverse tasks across link, node, and graph-level prediction.

> \[!IMPORTANT\]
> TGM is in beta, and may introduce breaking changes.

### Highlights

- **Unified Temporal API**: supports both continuous-time and discrete-time graphs, and graph discretization
- **Efficiency**: ~7.8× faster training and ~175× faster discretization vs. existing research libraries
- **Research-Oriented**: modular hook framework standardizes workflows for link, node, and graph-level tasks
- **Datasets**: built-in support for popular datasets (e.g., TGB[^12])

### Supported Methods

To request a method for prioritization, please [open an issue](https://github.com/tgm-team/tgm/issues) or [join the discussion](https://github.com/tgm-team/tgm/discussions).

| Status      | Methods                                                                                                                                                       |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Implemented | EdgeBank[^1], GCN[^2], GC-LSTM[^3], GraphMixer[^4], TGAT[^5], TGN[^6], DygFormer[^7], TPNet[^8], ROLAND [^13], PopTrack [^14], TNCN[^9], Base3[^15] CTAN[^16] |
| Planned     | DyGMamba[^10], NAT[^11]                                                                                                                                       |

## Installation

#### From Source (recommended)

```sh
pip install git+https://github.com/tgm-team/tgm.git@main
```

#### From PyPi

```
pip install tgm-lib
```

> \[!NOTE\]
> Windows is not directly tested in our CI. Additional setup may be required.
> For instance, for *cuda:12.4*, you will need to manually install the appropriate PyTorch wheels:
>
> ```sh
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
> ```

## Quick Tour for New Users

### System Design Overview

![image](./docs/img/architecture-dark.svg#gh-dark-mode-only)
![image](./docs/img/architecture-light.svg#gh-light-mode-only)

TGM is organized as a **three-layer architecture**:

1. **Data Layer**

   - Immutable, time-sorted coordinate-format graph storage with lightweight, concurrency-safe graph views.
   - Efficient time-based slicing and binary search over timestamps, enabling fast recent-neighbor retrieval.
   - Supports continuous-time and discrete-time loading, with vectorized snapshot creation.
   - Extensible backend allows alternative storage layouts for future models.

1. **Execution Layer**

   - The DataLoader is responsible for iterating through the temporal graph data stream by time or events based on the user-defined granularity.
   - HookManager orchestrates transformations during data loading (e.g., temporal neighbor sampling), dynamically adding relevant attributes to the Batch yielded by the dataloader.
   - Hooks can be combined and registered under specific conditions (analytics, training, etc.).
   - Pre-defined recipes simplify common setups (e.g. TGB link prediction) and prevent common pitfalls (e.g., mismanaging negatives).

1. **ML Layer**

   - Materializes batches directly on-device for model computation.
   - Supports node-, link-, and graph-level prediction.

> \[!TIP\]
> Check out [our paper](https://arxiv.org/abs/2510.07586) for technical details.

### Minimal Example

Here’s a basic example demonstrating how to train TGCN for dynamic node property prediction on `tgbn-trade`:

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

# tgbn-trade has no static node features, so we create Gaussian ones (dim=64)
static_node_x = torch.randn((train_dg.num_nodes, 64))

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.recurrent = TGCN(in_channels=node_dim, out_channels=embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(
        self, batch: DGBatch, node_feat: torch.tensor, h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_index = torch.stack([batch.edge_src, batch.edge_dst], dim=0)
        h_0 = self.recurrent(node_feat, edge_index, H=h)
        z = F.relu(h_0)
        z = self.linear(z)
        return z, h_0

# Initialize our model and optimizer
encoder = RecurrentGCN(node_dim=static_node_x.shape[1], embed_dim=128)
decoder = NodePredictor(in_dim=128, out_dim=train_dg.node_y_dim)
opt = torch.optim.Adam(set(encoder.parameters()) | set(decoder.parameters()), lr=0.001)

# Training loop
h_0 = None
for batch in train_loader:
    opt.zero_grad()
    y_true = batch.node_y
    if y_true is None:
        continue

    z, h_0 = encoder(batch, static_node_x, h_0)
    z_node = z[batch.node_y_nids]
    y_pred = decoder(z_node)

    loss = F.cross_entropy(y_pred, y_true)
    loss.backward()
    opt.step()
    h_0 = h_0.detach()
```

### Running Pre-packaged Examples

TGM includes pre-packaged example scripts to help you get started quickly. The examples require extra dependencies beyond the core library.

To get started, [follow our installation from source instructions](#installation) and then install the additional dependencies:

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

- Explore more of our [examples](../tgm/examples/)
- Dive deeper into TGM with our [tutorials](../tgm/docs/tutorials/)

## Citation

If you use TGM in your work, please cite [our paper](https://arxiv.org/abs/2510.07586):

```bibtex
@misc{chmura2025tgm,
  title  = {TGM: A Modular and Efficient Library for Machine Learning on Temporal Graphs},
  author = {Chmura, Jacob and Huang, Shenyang and Ngo, Tran Gia Bao and Parviz, Ali and Poursafaei, Farimah and Leskovec, Jure and Bronstein, Michael and Rabusseau, Guillaume and Fey, Matthias and Rabbany, Reihaneh},
  year   = {2025},
  note   = {arXiv:2510.07586}
}
```

## Contributing

We welcome contributions. If you encounter problems or would like to propose a new features, please open an [issue](https://github.com/tgm-team/tgm/issues) [and join the discussion](https://github.com/tgm-team/tgm/discussions). For details on contributing to TGM, see our [contribution guide](./.github/CONTRIBUTING.md).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## References

[^12]: [Temporal Graph Benchmark](https://tgb.complexdatalab.com/)

[^1]: [Towards Better Evaluation for Dynamic Link Prediction](https://arxiv.org/abs/2207.10128)

[^2]: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)

[^3]: [GC-LSTM: Graph Convolution Embedded LSTM for Dynamic Link Prediction](https://arxiv.org/abs/1812.04206)

[^4]: [Do We Really Need Complicated Model Architectures For Temporal Networks?](https://arxiv.org/abs/2302.11636)

[^5]: [Inductive Representation Learning on Temporal Graphs](https://arxiv.org/abs/2002.07962)

[^6]: [Temporal Graph Networks for Deep Learning on Dynamic Graphs](https://arxiv.org/abs/2006.10637)

[^7]: [Towards Better Dynamic Graph Learning: New Architecture and Unified Library](https://arxiv.org/abs/2303.13047)

[^8]: [Improving Temporal Link Prediction via Temporal Walk Matrix Projection](https://arxiv.org/abs/2410.04013)

[^13]: [ROLAND: Graph Learning Framework for Dynamic Graphs](https://arxiv.org/pdf/2208.07239)

[^14]: [Temporal Graph Models Fail to Capture Global Temporal Dynamics](https://openreview.net/pdf?id=9kLDrE5rsW)

[^9]: [Efficient Neural Common Neighbor for Temporal Graph Link Prediction](https://arxiv.org/abs/2406.07926)

[^15]: [Base3: a simple interpolation-based ensemble method for robust dynamic link prediction](https://www.arxiv.org/abs/2506.12764)

[^16]: [Long Range Propagation on Continuous-Time Dynamic Graphs](https://arxiv.org/abs/2406.02740)

[^10]: [DyGMamba: Efficiently Modeling Long-Term Temporal Dependency on Continuous-Time Dynamic Graphs with State Space Models](https://arxiv.org/abs/2408.04713)

[^11]: [Neighborhood-aware Scalable Temporal Network Representation Learning](https://arxiv.org/abs/2209.01084)
