# /// script
# requires-python = ">=3.13,<3.15"
# dependencies = [
#     "pydata-sphinx-theme>=0.24.0",
#     "sphinx>=9.2.0",
#     "sphinx-autoapi>=3.9.0",
#     "sphinx-design>=0.7.0",
#     "sphinx-gallery>=0.22.1",
#     "sphinx-togglebutton>=0.4.5",
# ]
# ///


"""Command line interface for managing the documentation build"""

__all__ = [
    "main",
]

import argparse
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from importlib import metadata
from pathlib import Path


def get_doc_dependencies(package_name: str) -> list[str]:
    doc_deps = [
        dep.split()[0].strip()
        for dep in metadata.requires(package_name)
        if dep.endswith("extra == 'doc'")
    ]
    # clean up the dependency names to remove any version specifiers or extras
    delimiters = ["<", ">", "=", "[", "]"]
    for i, dep in enumerate(doc_deps):
        for delimiter in delimiters:
            if delimiter in dep:
                dep = dep.split(delimiter)[0].strip()
        doc_deps[i] = dep

    return doc_deps


def check_dependencies_installed(dependencies: list[str]) -> bool:
    for dep in dependencies:
        try:
            metadata.distribution(dep)
        except metadata.PackageNotFoundError:
            return False
    return True


def _doc_dir() -> Path:
    return Path(__file__).resolve().parent


def _generated_doc_index() -> Path:
    return _doc_dir() / "build" / "html" / "index.html"


def _clean_doc_outputs(include_autoexamples: bool = False) -> None:
    doc_dir = _doc_dir()
    relative_paths = ["build", "source/autoapi"]
    if include_autoexamples:
        relative_paths.append("source/_autoexamples")

    for relative_path in relative_paths:
        full_path = doc_dir / relative_path
        if full_path.exists():
            print(f"Removing doc/{relative_path}...")
            shutil.rmtree(full_path, ignore_errors=True)
    print("Documentation cleanup complete.")


def _spinner_animation(stop_event: threading.Event) -> None:
    """Display a simple spinner animation."""
    # spinner_chars = "⠤⠧⠇⠏⠋⠉⠹⠸⠼⠴"
    # spinner_chars = "⣷⣯⣟⡿⢿⣻⣽⣾"
    # spinner_chars = "⠄⠆⠇⠋⠙⠸⠰⠠⠰⠸⠙⠋⠇⠆"  # dots bounce
    # spinner_chars = "⠁⠂⠄⠌⠔⠤⠥⠦⠮⠶⠷⠿⠟⠛⠫⠋⠍⠉⠑⠡"  # sand 6
    spinner_chars = "⠁⠂⠄⡀⡈⡐⡠⣀⣁⣂⣄⣌⣔⣤⣥⣦⣮⣶⣷⣿⡿⠿⢟⠟⡛⠛⠫⢋⠋⠍⡉⠉⠑⠡⢁"  # sand 8
    idx = 0
    while not stop_event.is_set():
        sys.stdout.write(
            f"\rBuilding HTML documentation {spinner_chars[idx % len(spinner_chars)]} "
        )
        sys.stdout.flush()
        idx += 1
        time.sleep(0.08)
    sys.stdout.write("\rBuilding HTML documentation... Done!\n")
    sys.stdout.flush()


def _build_doc(quiet: bool = False, force: bool = False) -> None:
    """Build the documentation using Sphinx.

    This runs Sphinx through the active Python interpreter in the local environment,
    which generates the HTML documentation in `doc/build/html`.
    """
    doc_dir = _doc_dir()
    cmd = [sys.executable, "-m", "sphinx", "-M", "html", "source", "build"]
    if force:
        cmd.append("-a")  # write all files
        cmd.append("-E")  # don't use cached environment
    if quiet:
        cmd.append("-q")
        stop_event = threading.Event()
        spinner_thread = threading.Thread(
            target=_spinner_animation, args=(stop_event,), daemon=True
        )
        spinner_thread.start()
        try:
            subprocess.run(
                cmd,
                cwd=doc_dir,
                check=True,
            )
        finally:
            stop_event.set()
            spinner_thread.join(timeout=1)
    else:
        subprocess.run(
            cmd,
            cwd=doc_dir,
            check=True,
        )


def _clean_doc(*, include_autoexamples: bool = False) -> None:
    """Clean the documentation build.

    This removes the generated documentation outputs directly from the local
    documentation tree.
    """
    if include_autoexamples:
        print("Cleaning documentation (including _autoexamples gallery)...")
    else:
        print("Cleaning documentation...")
    _clean_doc_outputs(include_autoexamples=include_autoexamples)


def _view_doc() -> None:
    """Open the generated documentation in the default browser.

    This opens the generated HTML index directly from the local documentation tree.
    """
    doc_index = _generated_doc_index()
    if not doc_index.is_file():
        raise FileNotFoundError(f"{doc_index} not found - run 'uv run doc build' first.")

    webbrowser.open(doc_index.resolve().as_uri())


def _make_doc_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="doc",
        description="Documentation commands for the project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="{ build | clean | view }")

    build_parser = subparsers.add_parser(
        "build",
        help="Build the HTML documentation using Sphinx.",
    )
    build_parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress most output during the build.",
    )

    build_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force a full rebuild of the documentation (excluding autoexamples).",
    )

    clean_parser = subparsers.add_parser(
        "clean",
        help="Remove generated documentation outputs.",
    )
    clean_parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Also remove the generated _autoexamples gallery.",
    )

    subparsers.add_parser(
        "view",
        help="Open the built HTML documentation in your browser.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run documentation subcommands.

    Parameters
    ----------
    argv : list[str] | None, optional
        Command-line arguments to parse. If not provided, the current process
        arguments are used.

    Returns
    -------
    int
        Exit status code. Returns ``0`` after successfully handling the command
        or printing the help message.

    Examples
    --------
    To build the documentation::

        uv run doc build

    To clean the generated documentation outputs::

        uv run doc clean

    To clean the generated documentation outputs including the autoexamples gallery::

        uv run doc clean --all

    To open the built documentation in the default web browser::

        uv run doc view

    Notes
    -----
    When invoked without a subcommand, the help text is shown::

        uv run doc

    To see the flags available for a specific subcommand, use the help flag with that
    subcommand::

        uv run doc build --help
        uv run doc build -h
    """
    package_name = "optical_diagram"
    if not check_dependencies_installed(get_doc_dependencies(package_name)):
        print("Documentation dependencies are not installed. Installing...")
        sync_doc_command = ["uv", "sync", "--group", "doc"]
        subprocess.run(sync_doc_command, check=True)

    parser = _make_doc_parser()
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        parser.print_help()
        return 0

    args = parser.parse_args(argv)

    if args.command == "build":
        _build_doc(quiet=args.quiet, force=args.force)
    elif args.command == "clean":
        _clean_doc(include_autoexamples=args.all)
    elif args.command == "view":
        _view_doc()

    return 0


if __name__ == "__main__":
    main()
