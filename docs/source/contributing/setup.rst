=============================
Development Environment Setup
=============================

To make changes to the pySALIENT codebase or documentation, you'll need to set up a development environment.

Prerequisites
-------------
*   Git
*   Python 3.11 or later
*   `uv` (Recommended, but `pip` can also be used)

Setup Steps
-----------

1.  **Clone the repository** (if you haven't already):
    If you plan to contribute, it's best to fork the repository first on GitHub and clone your fork.

    .. code-block:: bash

        git clone https://github.com/your-username/pysalient # Replace with your fork URL
        cd pysalient

    If you only need to build the documentation or run tests locally without contributing back, you can clone the main repository:

    .. code-block:: bash

        git clone https://github.com/esalient/pysalient
        cd pysalient

2.  **Set up the virtual environment using `uv`:**
    This command creates a virtual environment (named ``.venv`` by default) and ensures Python 3.11 is used.

    .. code-block:: bash

        uv venv --python 3.11

3.  **Activate the virtual environment:**
    *   On Linux/macOS:

        .. code-block:: bash

            source .venv/bin/activate

    *   On Windows:

        .. code-block:: bash

            .\.venv\Scripts\activate

    *(Note: While activation is recommended for shell convenience, subsequent `uv` commands will automatically detect and use the ``.venv`` even if it's not activated.)*

4.  **Install dependencies:**
    Install the project in editable mode along with development dependencies:

    .. code-block:: bash

        uv pip install -e ".[dev]"

    This installs pySALIENT itself in a way that your code changes are immediately reflected, plus all the tools needed for testing, linting, formatting, and building documentation (like Sphinx, pytest, ruff, black).

Building and Viewing Documentation Locally
------------------------------------------

This project uses Sphinx to generate documentation from docstrings and ``.rst`` files located in the ``docs/source/`` directory.

To build and view the documentation locally:

1.  Make sure your development environment is activated.
2.  Navigate to the documentation directory:

    .. code-block:: bash

        cd docs

3.  Build the HTML documentation:

    .. code-block:: bash

        make html
        # Or on Windows: make.bat html

4.  Open the main page in your browser:

    *   On macOS: ``open build/html/index.html``
    *   On Linux: ``xdg-open build/html/index.html``
    *   On Windows: ``start build/html/index.html`` (or navigate via File Explorer)

Alternatively, use the ``serve`` target for automatic rebuilding and easy viewing via a local web server:

1.  Navigate to the ``docs`` directory.
2.  Run the serve command:

    .. code-block:: bash

        make serve
        # Or on Windows: make.bat serve

3.  Open ``http://localhost:8000`` in your web browser. Press ``Ctrl+C`` in the terminal to stop the server.