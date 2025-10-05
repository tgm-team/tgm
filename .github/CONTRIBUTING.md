# Contributing to TGM

Thank you for your interest in contributing! This guide covers setup, deveopment workflow and testing processes in TGM. We also invite you to join the [TGL slack group](https://tglworkshop.slack.com/ssb/redirect), where you can connect with the community and ask questions.

## Prerequisites

TGM requires **Python 3.10**.

We use [uv](https://docs.astral.sh/uv/) to manage dependencies and provide a reproducible environment. If you don't have `uv` installed:

```sh
pip install uv
```

> \[!NOTE\]
> `uv` creates an isolated environment in `.venv` based on the lockfile `uv.lock`.

### Using uv

#### Running commands

Most commands can be run by simply prepending `uv run` to the respective command:

- Instead of running `python <command>`, you will run `uv run python <command>`
- Instead of running `pytest`, you will run `uv run pytest`

#### Managing dependencies

_uv_ automatically resolves package versions while respecting existing dependencies.

**Add or remove a core dependency**:

```sh
uv add <package>
uv remove <package>
```

> \[!IMPORTANT\]
> This updates `pyproject.toml` and `uv.lock` automatically, which should be committed.

Core dependencies should be kept to an absolute minimum. To add optional dependencies, `uv` has the notion of _dependency groups_. For instance, the _dev_ group is the set of dependencies required for TGM development, but is not necessarily shipped to end-users of the library.

**Add or remove a group dependency**:

```sh
uv add --group <group> <package>
uv remove --group <group> <package>
```

> \[!TIP\]
> Any package on PyPI can be added, making `uv` a drop in replacement for `pip`. For complex use cases such as non-python dependencies, or installing specific package versions, consult the [uv documentation](https://docs.astral.sh/uv/).

#### Activating the Virtual Environment

Sometimes you may want to activate .venv manually (e.g., for IDE integration):

```sh
. .venv/bin/activate # bash
```

**Note**: after doing so, you will have direct access to all executables (e.g. Python) as usual.

## Development Setup

### Install TGM from source

```sh
# Clone the repo
git clone https://github.com/tgm-team/tgm.git
cd tgm

# Install core dependencies
uv sync
```

### Install pre-commit hooks:

TGM includes [pre-commit hooks](../.pre-commit-config.yaml) for formatting, linting, static type analysis, and more.

The hooks can be installed by issuing:

```sh
uv run pre-commit install
```

> \[!TIP\]
> Hooks can be bypassed with the `--no-verify` flag, but using them is strongly recommended.

## Testing

The TGM test suite is organized into:

- `test/unit`: unit tests
- `test/integration`: integration tests
- `test/performance`: performance tests

### Unit Tests

Run the entire (unit) test suite with

```sh
./scripts/run_unit_tests.sh
```

### Integration Tests

Integration tests run periodicaly on our CI cluster to validate example correctness, latency, throughput and GPU usage.

To trigger manually:

```sh
./scripts/triger_integration_tests.sh
```

> \[!IMPORTANT\]
> You will need to setup an auth key for permission to trigger remote jobs on our CI. Contact [Jacob Chmura](jacobpaul.chmura@gmail.com) for details. At minimum, run examples locally to ensure nothing breaks.

### Performance Tests

Performance tests also run on our CI cluster to stress test various aspect of our core library. They can be triggered similar to our integration tests, assuming you have the permissions:

```sh
./scripts/trigger_perf_tests.sh
```

### Building Documentation

To build (and serve) the documentation locally:

```sh
./scripts/build_docs.sh
```

## Proposing a new model

1. Start by writing a new example akin to our previous examples, outside of TGM core.
1. Next, add the example into our integration test harness. Request the minimal amount of resources needed.
1. Once the harness is setup, we'll report some performance and efficiency numbers, to confirm the implementation is correct.
1. If you have reusable components (hooks, layers, etc.) that are broadly useful, they may be migrated to TGM core. Note each addition must include unit tests and a justification for inclusion.

### Submitting PRs

Before opening a Pull Request, please first [open a new issue](https://github.com/tgm-team/tgm/issues) describing the feature you’d like to add or the bug you’d like to fix. Assign yourself to the issue so others know you’re working on it, and feel free to tag a core library developer if you’d like early feedback on your proposed approach.

Once you are ready to contribute code:

1. Fork the repository and create a feature brnach.
1. Implement your changes and ensure that all tests pass locally
1. Open a pull request referencing the related issue.

TGM uses [Github Actions](https://github.com/tgm-team/tgm/tree/main/.github/workflows) for continuous integration. Each PR will automatically be built and validated against the TGM guidelines.

Please ensure all the automatic checks pass and then tag members of the core team for review and further discussion.
