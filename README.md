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

> \[!NOTE\]
> TGM is in beta, and may introduce breaking changes.

### Library Highlights

- **Unified Support**: handles both continuous-time and discrete-time graphs
- **Time-First API**: native support for time granularity, graph discretization, and snapshot iteration
- **Efficiency**: ~7.8× faster training and ~175× faster graph discretization vs. existing research libraries
- **Research-Oriented**: modular hook framework standardizes TG transforms for link, node, and graph-level tasks
- **Datasets**: built-in support for popular datasets (e.g., [TGB](https://tgb.complexdatalab.com/))
- **Validated Methods**: 8+ implemented methods across diverse architectures

### Supported Methods

We aim to support the following temporal graph learning methods. If you want us to prioritize a new method, please open an [issue](https://github.com/tgm-team/tgm/issues) and feel free [to discuss them with us](https://github.com/tgm-team/tgm/discussions).

- :white_check_mark: [EdgeBank](https://arxiv.org/abs/2207.10128)
- :white_check_mark: [GCN](https://arxiv.org/abs/1609.02907)
- :white_check_mark: [GC-LSTM](https://arxiv.org/abs/1812.04206)
- :white_check_mark: [GraphMixer](https://arxiv.org/abs/2302.11636)
- :white_check_mark: [TGAT](https://arxiv.org/abs/2002.07962)
- :white_check_mark: [TGN](https://arxiv.org/abs/2006.10637)
- :white_check_mark: [DygFormer](https://arxiv.org/abs/2303.13047)
- :white_check_mark: [TPNet](https://arxiv.org/abs/2410.04013)
- :white_large_square: [TNCN](https://arxiv.org/abs/2406.07926)
- :white_large_square: [DyGMamba](https://arxiv.org/abs/2408.04713)
- :white_large_square: [NAT](https://arxiv.org/abs/2209.01084)

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
