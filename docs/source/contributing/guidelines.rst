========================
Contribution Guidelines
========================

First off, thank you for considering contributing to pySALIENT! 

Ways to Contribute
------------------

There are many ways to contribute, including:

*   **Reporting Bugs:** If you find a bug, please open an issue on GitHub describing the problem, how to reproduce it, and what you expected to happen.
*   **Suggesting Enhancements:** Have an idea for a new feature or an improvement to an existing one? Open an issue on GitHub to discuss it.
*   **Contributing Code or Documentation:** If you'd like to fix a bug, implement a feature, or improve the documentation, please follow the workflow described below.

For significant changes, it's often best to open an issue first to discuss the proposed changes and ensure they align with the project's goals.

Contribution Workflow
---------------------

We use a standard GitHub workflow for contributions:

1.  **Fork & Clone:** Fork the `pySALIENT repository <https://github.com/esalient/pysalient>`_ to your own GitHub account and then clone your fork locally:

    .. code-block:: bash

        git clone https://github.com/your-username/pysalient.git # Use your fork URL
        cd pysalient

2.  **Branch:** Create a new branch for your changes. Choose a descriptive name (e.g., ``fix-data-loading-bug``, ``add-new-metric-feature``):

    .. code-block:: bash

        git checkout -b your-branch-name

3.  **Code:** Make your desired changes to the code or documentation. Ensure you follow the project's coding style (e.g., using `black` for formatting).

4.  **Test & Lint:** Run the tests and linters to ensure code quality and prevent regressions:

    .. code-block:: bash

        pytest tests/
        ruff check .
        black . --check  # Check formatting without changing files

    Fix any issues reported by these tools. You can apply formatting automatically with ``black .``.

5.  **Create a Changelog Entry:** Before committing, record your change using ``changie``. Run the following command in your terminal and follow the prompts to describe your contribution (e.g., added feature, fixed bug, security update):

    .. code-block:: bash

        changie new

    This creates a small ``.yaml`` file in the ``.changes/`` directory. This file *must* be included in your commit in the next step.

6.  **Commit:** Commit your changes with a clear and concise commit message describing the work you've done:

    .. code-block:: bash

        git add .
        git commit -m "Brief description of your changes"

7.  **Push:** Push your branch to your fork on GitHub:

    .. code-block:: bash

        git push origin your-branch-name

8.  **Pull Request (PR):** Go to the original pySALIENT repository on GitHub and open a new Pull Request (PR) from your fork's branch to the main repository's primary branch (e.g., ``main``).

9.  **Describe:** In the PR description, clearly outline the changes you've made. Explain the purpose of the changes (e.g., "Fixes bug #123", "Adds feature XYZ as discussed in issue #456"). If your PR addresses a specific GitHub issue, please link to it (e.g., ``Closes #123``).

Review Process
--------------

Once you submit your Pull Request, the project maintainers will review it.

*   **Feedback:** Maintainers may provide feedback or request changes. Please address the feedback promptly. You can push additional commits to your branch; the PR will update automatically.
*   **Approval & Merge:** Once the PR is approved, a maintainer will merge it into the main codebase.

We aim to review contributions in a timely manner.

Community Expectations
----------------------

Please be respectful and constructive in all interactions within the project community (issues, pull requests, discussions). We value collaboration and a positive environment.

Questions?
----------

If you have questions about contributing, feel free to open an issue on GitHub.

Thank you again for your interest in contributing to pySALIENT!