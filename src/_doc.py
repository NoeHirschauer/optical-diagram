"""Entry point for the documentation CLI.

Actual implementation is located in ``doc/cli.py``. This module is just a thin wrapper to
allow the CLI to be run from the installed package.

Note that running this script will sync the environment with the `doc` extra dependencies
in order to build the documentation.

The ``uv run doc`` command is registered in the ``pyproject.toml`` file, which points to
this module. When the command is run, this module is executed, which in turn runs the
actual CLI implementation in ``doc/cli.py``. This is a workaround for the fact that uv
cannot run scripts that are not in src/ or installed packages without providing the full
path to the script.
"""

import runpy
import subprocess
from pathlib import Path


def main() -> int:
    """Run the documentation CLI located in ``doc/cli.py``."""
    # Step 1: Sync the environment with the `doc` extra dependencies if not installed
    try:
        import sphinx  # type: ignore
    except ModuleNotFoundError:
        sync_doc_command = ["uv", "sync", "--group", "doc"]
        subprocess.run(sync_doc_command, check=True)

    # Step 2: Run the documentation CLI
    cli_path = Path(__file__).resolve().parents[1] / "doc" / "cli.py"
    namespace = runpy.run_path(str(cli_path), run_name="_doc_cli")

    return namespace["main"]()
