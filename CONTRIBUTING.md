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

To install the package locally for development, create an environment (python 3.11).

```bash
python -m pip install -e pysalient/
```

We recommend using [mamba](https://mamba.readthedocs.io/en/latest/) to create the Python environment.

## Prerequisites

For [turbODBC](https://turbodbc.readthedocs.io/en/latest/pages/getting_started.html#installation) dependency, the following is required to be installed on the host: 

| Requirement               | Linux (apt)        | Linux (dnf) *  | OSX                    |
|:--------------------------|:-------------------|:---------------|:-----------------------|
| C++11 compiler            | `gcc`              | `gcc`          | clang with OSX 10.9+   |
| Boost library + header(1) | `libboost-all-dev` | `boost-devel`  | `boost`                |
| ODBC library              | `python-dev`       | `python-devel` | use `pyenv` to install |

* Ensure EPEL is enabled.

### Tools

[//]: #a (review and add tox)
[//]: #b (review and add make for multiple test setups)

- [pytest](https://docs.pytest.org/en/latest/) to define, discover, and run tests
- [flake8](https://flake8.pycqa.org/en/latest/) for code linting
- [black](https://github.com/psf/black) for coding formatting
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

