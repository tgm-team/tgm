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
- **Datasets**: built-in support for popular datasets (e.g., [TGB](https://tgb.complexdatalab.com/))
- **Validated Methods**: 8+ implemented methods across diverse architectures

### Supported Methods

TGM implements a range of temporal graph learning methods. To request a method for prioritization, please [open an issue](https://github.com/tgm-team/tgm/issues) or [join the discussion](https://github.com/tgm-team/tgm/discussions).

**Implemented:** EdgeBank[^1], GCN[^2], GC-LSTM[^3], GraphMixer[^4], TGAT[^5], TGN[^6], DygFormer[^7], TPNet[^8]

**Planned:** TNCN[^9], DyGMamba[^10], NAT[^11]

### Performance Benchmarks

Work in progress.

## Installation

#### From Source (recommended)

```sh
pip install git+https://github.com/tgm-team/tgm.git@main
```

#### From PyPi

```
pip install tgm-lib
```

### Windows

To enable GPU on non-linux platforms, you will need to manually install the appropriate torch wheels for your drivers. For instance, for *cuda:12.4*, follow the steps above and then issue:

```sh
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Quick Tour for New Users

![image](./docs/img/architecture-dark.svg#gh-dark-mode-only)
![image](./docs/img/architecture-light.svg#gh-light-mode-only)

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

> \[!TIP\]
> Refer to our [our docs](https://tgm.readthedocs.io/) for more information and TG example recipes.

### Creating a new model

Work in progress.

## Citation

Please cite [our paper](https://github.com/tgm-team/tgm) if your use this code in your own work:

```
@article{TODO,
title   = "TODO",
author  = "TODO"
journal = "TODO",
url     = "TODO"
year    = "2025",
}
```

## Contributing

If you notice anything unexpected, or would like to propose a new feature, please open an [issue](https://github.com/tgm-team/tgm/issues) and feel free [to discuss them with us](https://github.com/tgm-team/tgm/discussions).

To learn more about making a contribution to TGM see our [contribution guide](./.github/CONTRIBUTING.md).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

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
