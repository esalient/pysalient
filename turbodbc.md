# turbODBC Installation Requirements

For future integration of [turbODBC](https://turbodbc.readthedocs.io/en/latest/pages/getting_started.html#installation) dependency, the following will be required to be installed on the host:

| Requirement               | Linux (apt)        | Linux (dnf) *  | OSX                    |
|:--------------------------|:-------------------|:---------------|:-----------------------|
| C++11 compiler            | `gcc`              | `gcc`          | clang with OSX 10.9+   |
| Boost library + header(1) | `libboost-all-dev` | `boost-devel`  | `boost`                |
| ODBC library              | `python-dev`       | `python-devel` | use `pyenv` to install |

* Ensure EPEL is enabled.

> **Note:** This dependency is not currently required for pySALIENT but may be added in future releases.