"""Stub module to run the documentation CLI from the root of the repository.

There is nothing here, actual implementation is in ``doc/cli.py``. This module is only
used to allow running the documentation CLI from the root of the repository, e.g.::

    uv run doc build

"""

import runpy
from pathlib import Path


def main() -> int:
    """Run the documentation CLI located in ``doc/cli.py``."""

    cli_path = Path(__file__).resolve().parents[1] / "doc" / "cli.py"
    namespace = runpy.run_path(str(cli_path), run_name="_doc_cli")

    return namespace["main"]()
