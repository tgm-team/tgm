<a id="readme-top"></a>

<div align="center">
<h1> OpenDG </h1>
<h3 style="font-size: 22px">Efficient and modular training on dynamic graphs</h3>
<a href="https://opendg.readthedocs.io/en/latest"/><strong style="font-size: 18px;">Read Our Docs»</strong></a>
<a href="https://github.com/shenyangHuang/opendg"/><strong style="font-size: 18px;">Read Our Paper»</strong></a>
<br/>
<br/>

[![GitHub Repo stars](https://img.shields.io/github/stars/shenyangHuang/opendg)](https://github.com/shenyangHuang/opendg/stargazers)
[![Unit Tests](https://github.com/shenyangHuang/opendg/actions/workflows/testing.yml/badge.svg)](https://github.com/shenyangHuang/opendg/actions/workflows/testing.yml)
[![Linting](https://github.com/shenyangHuang/opendg/actions/workflows/ruff.yml/badge.svg)](https://github.com/shenuangHuang/opendg/actions/workflows/ruff.yml)

</div>

## About The Project

OpenDG is a research library designed to accelerate the training over dynamic graphs, and facilitate rapid prototyping of new temporal graph learning methods.

Our main goal is to provide an efficient and easy to use abstraction over dynamic graphs to enable new practitioners to quickly understand, train, and ulimately, contribute new research in the field. We natively support both discrete and continuous-time graphs.

> \[!NOTE\]
> OpenDG is still alpha software, and may introduce breaking changes.

### Library Highlights

### Architecture Overview

## Quick Tour for New Users

## Installation

The current recommended way to install OpenDG is from source.

#### Using [uv](https://docs.astral.sh/uv/) (recommended)

```sh
# Create and activate your venv
uv venv my_venv --python 3.10 && source my_venv/bin/activate

# Install the wheels into the venv
uv pip install git+https://github.com/shenyangHuang/opendg.git

# Test the install
python -c 'import opendg; print(opendg.__version__)'
```

#### Using [pip](https://pip.pypa.io/en/stable/installation/)

```sh
# Create and activate your venv
python3.10 -m venv my_venv && source my_venv/bin/activate

# Install the wheels into the venv
pip install git+https://github.com/shenyangHuang/opendg.git

# Test the install
python -c 'import opendg; print(opendg.__version__)'
```

## Documentation

Documentation along with a quick start guide can be found on the [docs website](https://opendg.readthedocs.io/).

## Citation

Please cite [our paper](https://github.com/shenyangHuang/opendg) if your use this code in your own work:

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

If you notice anything unexpected, or would like to propose a new feature, please open an [issue](https://github.com/shenyangHuang/opendg/issues) and feel free [to discuss them with us](https://github.com/shenyangHuang/opendg/discussions).

To learn more about making a contribution to OpenDG see our [contribution guide](./.github/CONTRIBUTING.md).

<p align="right">(<a href="#readme-top">back to top</a>)</p>
