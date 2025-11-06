=============
Installation
=============

pySALIENT requires Python 3.11 or later.

The recommended way to install pySALIENT is using `pixi <https://pixi.sh>`_, which manages both conda and PyPI dependencies seamlessly.

1.  **Install pixi:**
    If you don't have pixi installed yet:

    .. code-block:: bash

        # Linux & macOS
        curl -fsSL https://pixi.sh/install.sh | bash

        # Windows (PowerShell)
        iwr -useb https://pixi.sh/install.ps1 | iex

2.  **Clone the repository:**
    Open your terminal and run the following command:

    .. code-block:: bash

        git clone https://github.com/esalient/pysalient
        cd pysalient

3.  **Install with pixi:**
    Simply run:

    .. code-block:: bash

        pixi install

    This will create an isolated environment with all dependencies installed.

4.  **Using pySALIENT:**

    You can either activate the pixi environment or run commands directly:

    .. code-block:: bash

        # Activate the environment (optional)
        pixi shell

        # Or run Python directly without activation
        pixi run python your_script.py

Alternative: Using pip
----------------------

If you prefer using ``pip``, you can still install pySALIENT in a traditional virtual environment:

.. code-block:: bash

    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
    pip install .