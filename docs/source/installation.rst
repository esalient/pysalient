=============
Installation
=============

pySALIENT requires Python 3.11 or later.

The recommended way to install pySALIENT for general use is directly from the source repository.

1.  **Clone the repository:**
    Open your terminal and run the following command:

    .. code-block:: bash

        git clone https://github.com/esalient/pysalient # Replace with the correct URL if different
        cd pysalient

2.  **Install the package:**
    You can use either `uv` or `pip` to install the package. Navigate into the cloned directory (`pysalient`) if you haven't already.

    *   **Using `uv`:**
        If you have `uv` installed, you can create a virtual environment and install the package in one step:

        .. code-block:: bash

            uv venv --python 3.11  # Creates .venv and ensures Python 3.11
            uv pip install .       # Installs pySALIENT into the environment

        Activate the environment:
        *   Linux/macOS: ``source .venv/bin/activate``
        *   Windows: ``.\.venv\Scripts\activate``

    *   **Using `pip`:**
        If you prefer using `pip`, first create and activate a virtual environment using your preferred method (e.g., `venv`):

        .. code-block:: bash

            python3 -m venv .venv   # Or python -m venv .venv
            # Activate the environment (see commands above)
            pip install .           # Installs pySALIENT