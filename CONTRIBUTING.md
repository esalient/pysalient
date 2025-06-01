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

To install the package locally for development, create an environment (python 3.11), we recommend using [uv](https://astral.sh/blog/uv) with [venv](https://docs.astral.sh/uv/pip/environments/)

```bash
uv venv --python 3.11
# activate
source .venv/bin/activate # windows: .\.venv\Scripts\activate
uv pip install -e ".[dev]"
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

