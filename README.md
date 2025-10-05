<a id="readme-top"></a>

![image](./docs/img/logo.svg)

<div align="center">
<h3 style="font-size: 22px">Efficient and Modular ML on Temporal Graphs</h3>
<a href="https://tgm.readthedocs.io/en/latest"/><strong style="font-size: 18px;"/>Read Our Docs»</strong></a>
<a href="https://github.com/tgm-team/tgm"/><strong style="font-size: 18px;"/>Read Our Paper»</strong></a>
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

| Status      | Methods                                                                                         |
| ----------- | ----------------------------------------------------------------------------------------------- |
| Implemented | EdgeBank[^1], GCN[^2], GC-LSTM[^3], GraphMixer[^4], TGAT[^5], TGN[^6], DygFormer[^7], TPNet[^8] |
| Planned     | TNCN[^9], DyGMamba[^10], NAT[^11]                                                               |

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

TGM provides a modular and efficient workflow for temporal graph learning. In this quick tour, you'll get an overview of the system design, see a minimal example, and learn how to explore pre-packaged examples and tutorials.

### System Design Overview

![image](./docs/img/architecture-dark.svg#gh-dark-mode-only)
![image](./docs/img/architecture-light.svg#gh-light-mode-only)

TODO

### Minimal Example

Here’s a basic workflow demonstrating how to load a dataset, define a model, and run training:

```python

from tgm import DGData, DGraph
from tgm.nn import GCNEncoder, NodePredictor

train_data, val_data, test_data = DGData.from_tgb("tgbl-trade").split()
train_dg = DGraph(train_data)
train_loader = DGDataLoader(train_dg, batch_unit="s")

encoder = GCNEncoder(
    in_channels=train_dg.static_node_feats.shape[1],
    embed_dim=128,
    out_channels=128,
    num_layers=2,
)
decoder = NodePredictor(in_dim=args.embed_dim, out_dim=???)
opt = torch.optim.Adam(set(encoder.parameters()) | set(decoder.parameters()), lr=0.001)

for batch in loader:
    opt.zero_grad()
    y_true = batch.dynamic_node_feats
    if y_true is None:
        continue

    z = encoder(batch, static_node_feats)
    z_node = z[batch.node_ids]
    y_pred = decoder(z_node)

    loss = F.cross_entropy(y_pred, y_true)
    loss.backward()
    opt.step()
```

### Running Pre-packaged Examples

Start by syncing additional dependencies in our example scripts:

```sh
uv sync --group examples
```

For this example, we'll run [TGAT](https://arxiv.org/abs/2002.07962) dynamic link-prediction on [tgbl-wiki](https://tgb.complexdatalab.com/docs/leader_linkprop/#tgbl-wiki-v2). We'll use standard parameters on run on GPU. We show some explicit arguments for clarity:

```
python examples/linkproppred/tgat.py \
  --dataset tgbl-wiki \
  --bsize 200 \
  --device cuda \
  --epochs 1 \
  --n-nbrs 20 20 \
  --sampling recency
```

### Next steps

- Explore more of our [examples](../tgm/examples/)
- Dive deeper into TGM with our [tutorials](../tgm/docs/tutorials/)

> \[!TIP\]
> Refer to our [our docs](https://tgm.readthedocs.io/) for more information and TG example recipes.

## Citation

If you use TGM in your work, please cite [our paper](https://github.com/tgm-team/tgm):

```bibtex
@article{TODO,
  title   = "TODO",
  author  = "TODO",
  journal = "TODO",
  year    = "2025",
  url     = "TODO"
}
```

## Contributing

We welcome contributions. If you encounter problems or would like to propose a new features, please open an [issue](https://github.com/tgm-team/tgm/issues) [and join the discussion](https://github.com/tgm-team/tgm/discussions). For details on contributin to TGM, see our [contribution guide](./.github/CONTRIBUTING.md).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

[^12]: https://tgb.complexdatalab.com/

[^1]: https://arxiv.org/abs/2207.10128

[^2]: https://arxiv.org/abs/1609.02907

[^3]: https://arxiv.org/abs/1812.04206

[^4]: https://arxiv.org/abs/2302.11636

[^5]: https://arxiv.org/abs/2002.07962

[^6]: https://arxiv.org/abs/2006.10637

[^7]: https://arxiv.org/abs/2303.13047

[^8]: https://arxiv.org/abs/2410.04013

[^9]: https://arxiv.org/abs/2406.07926

[^10]: https://arxiv.org/abs/2408.04713

[^11]: https://arxiv.org/abs/2209.01084
