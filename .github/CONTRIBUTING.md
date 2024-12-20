# Contributing to OpenDG

## Developping OpenDG

### Prerequisites

The project uses [uv](https://docs.astral.sh/uv/) to manage and lock project dependencies for a consistent and reproducible environment. If you do not have `uv` installed on your system, visit [this page](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.

**Note**: If you have `pip` installed globally you can just invoke:

```sh
pip install uv
```

#### Using uv

### Installation

```sh
# Clone the repo
git clone https://github.com/shenyangHuang/openDG.git
cd openDG

# Install core dependencies into an isolated environment
uv sync
```

#### Install pre-commit hooks:

```sh
uv run pre-commit install
```

## Unit Testing

The openDG test suite is located uner `test/`.
Run the entire test suite with

```sh
uv run pytest
```

## Continuous Integration

OpenDG uses [Github Actions](https://github.com/shenyangHuang/openDG/tree/main/.github/workflows) for continuous integration.

Everytime you send a Pull Request, your commit will be built and checked against the openDG guidelines:

1. Ensure that your code is formatted correctly. We use the [`Ruff-Formatter`](https://docs.astral.sh/ruff/formatter/).

   ```bash
   uv run ruff format .
   ```

   The pre-commit hooks will auto-format your code according to our style spec.

1. Ensure that the entire test suite passes and that code coverage roughly stays the same.

   ```bash
   uv run pytest --cov
   ```

## Building Documentation

To build the documentation:

1. [Build and install](#developing-pyg) openDG from source.
1. Install [Sphinx](https://www.sphinx-doc.org/en/master/) theme via
   ```bash
   pip install git+https://github.com/pyg-team/pyg_sphinx_theme.git
   ```
1. Generate the documentation via:
   ```bash
   cd docs
   uv run make html
   ```

The documentation is now available to view by opening `docs/build/html/index.html`.
