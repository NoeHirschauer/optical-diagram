import runpy
import subprocess
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


def main() -> int:
    """Run the documentation CLI located in ``doc/cli.py``."""

    # Step 1: check if the `doc` extra dependencies are installed. If not, install them
    package_name = "optical_diagram"
    if not check_dependencies_installed(get_doc_dependencies(package_name)):
        print("Documentation dependencies are not installed. Installing...")
        sync_doc_command = ["uv", "sync", "--group", "doc"]
        subprocess.run(sync_doc_command, check=True)

    # Step 2: Run the documentation CLI
    cli_path = Path(__file__).resolve().parents[1] / "doc" / "cli.py"
    namespace = runpy.run_path(str(cli_path), run_name="_doc_cli")

    return namespace["main"]()
