<a id="readme-top"></a>

![image](./docs/img/logo.svg)

<div align="center">
<h3 style="font-size: 22px">Efficient and Modular Training on Dynamic Graphs</h3>
<a href="https://tgm.readthedocs.io/en/latest"/><strong style="font-size: 18px;">Read Our Docs»</strong></a>
<a href="https://github.com/tgm-team/tgm"/><strong style="font-size: 18px;">Read Our Paper»</strong></a>
<br/>
<br/>

[![GitHub Repo stars](https://img.shields.io/github/stars/tgm-team/tgm)](https://github.com/tgm-team/tgm/stargazers)
[![Unit Tests](https://github.com/tgm-team/tgm/actions/workflows/testing.yml/badge.svg)](https://github.com/tgm-team/tgm/actions/workflows/testing.yml)
[![Linting](https://github.com/tgm-team/tgm/actions/workflows/ruff.yml/badge.svg)](https://github.com/tgm-team/tgm/actions/workflows/ruff.yml)

</div>

## About The Project

TGM is a research library designed to accelerate training workloads over dynamic graphs and facilitate prototyping of temporal graph learning methods.

Our main goal is to provide an efficient abstraction over dynamic graphs to enable new practitioners to quickly contribute to research in the field. We natively support both discrete and continuous-time graphs.

> \[!NOTE\]
> TGM is still alpha software, and may introduce breaking changes.

### Library Highlights

## Supported Methods

### Architecture Overview

## Quick Tour for New Users

## Installation

The current recommended way to install TGM is from source.

#### Using [uv](https://docs.astral.sh/uv/) (recommended)

```sh
# Create and activate your venv
uv venv my_venv --python 3.10 && source my_venv/bin/activate

# Install the wheels into the venv
uv pip install git+https://github.com/tgm-team/tgm.git

# Test the install
python -c 'import tgm; print(tgm.__version__)'
```

#### Using [pip](https://pip.pypa.io/en/stable/installation/)

```sh
# Create and activate your venv
python3.10 -m venv my_venv && source my_venv/bin/activate

# Install the wheels into the venv
pip install git+https://github.com/tgm-team/tgm.git

# Test the install
python -c 'import tgm; print(tgm.__version__)'
```

## Documentation

Documentation along with a quick start guide can be found on the [docs website](https://tgm.readthedocs.io/).

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
