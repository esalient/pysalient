# Contributing to PySALIENT

Welcome! 

## Getting the code

In order to modify the source code, `git` is required.

1. Fork the `pysalient` repository
2. Clone your fork locally
3. Check out a new branch for your proposed changes
4. Push the changes to your fork
5. Open a pull request against `pysalient` from your forked repository

## Setup development environment

### Local install

To install the package locally for development, we recommend using [pixi](https://pixi.sh), which manages both conda and PyPI dependencies seamlessly.

#### Install pixi

If you don't have pixi installed yet:

```bash
# Linux & macOS
curl -fsSL https://pixi.sh/install.sh | bash

# Windows (PowerShell)
iwr -useb https://pixi.sh/install.ps1 | iex
```

#### Setup development environment

```bash
# Install the default development environment
pixi install

# Activate the environment (optional - pixi run commands work without activation)
pixi shell

# Or run commands directly without activating
pixi run test
pixi run lint
```

#### Available environments

- `default` - Core development tools (pytest, ruff, mypy, sphinx)
- `examples` - Development tools + Jupyter (for running notebooks)
- `plot` - Development tools + matplotlib
- `all` - All features combined

```bash
# Use a specific environment
pixi shell -e examples  # For Jupyter notebooks
pixi shell -e all       # For everything
```

## Tools

[//]: #a (review and add tox)
[//]: #b (review and add make for multiple test setups)

- [pytest](https://docs.pytest.org/en/latest/) to define, discover, and run tests
- [ruff](https://docs.astral.sh/ruff/) for code linting and formatting
- [mypy](https://mypy.readthedocs.io/en/stable/) for static type checking
- [pre-commit](https://pre-commit.com/) to easily run checks
- [changie](https://changie.dev/) to create Changelog entries, without merge conflicts
- [GitHub actions](https://github.com/features/actions) for the CI/CD and push to [PyPi](https://pypi.org/).

[//]: #c (Add Issue Tracking)

## CHANGELOG

The project uses [changie](https://changie.dev/) to generate `CHANGELOG` entries. Do **not** edit the `CHANGELOG.md` directly.

Changie needs to be installed on your host to use.

Once installed, please run the following command to add your change:

```bash
changie new
```

[//]: #d (Add release process)

## From the team

Thank you for your interest in the project and contributing.

