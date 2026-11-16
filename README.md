# optical-diagram

An attempt at an object-oriented python module to trace simple optical diagrams, including basic optical elements and a utility for beam tracing

## Installation

This package is managed with [uv](https://github.com/astral-sh/uv).
Install `uv`, then run the following from the project root:

```bash
uv sync
```

This creates or updates the project environment from `uv.lock`, installs the
package in editable mode, and includes the development dependencies. To also
install the documentation dependencies explicitly, use the `doc` dependency
group:

```bash
uv sync --group doc
```

## Basic usage

- Run one of the example scripts in the `examples/` folder to see typical usage (e.g. the Galilean telescope example):

  ```bash
  uv run python -m examples.galilean_telescope
  ```

- Or import the package in Python:

  ```python
  from optical_diagram import OpticalTable, ConvergingLens, RayTracedBeam
  # ... create elements and render with OpticalTable ...
  ```

## Running tests

Run the test suite in the uv-managed environment:

```bash
uv run pytest
```

## Building the documentation

The documentation is managed through the `doc` project script. From the
repository root:

```bash
uv run doc build
```

The command uses the `doc` dependency group when the documentation
dependencies are not already installed. Other available commands are:

```bash
uv run doc clean       # Remove generated documentation
uv run doc clean --all # Also remove generated examples
uv run doc view        # Open the built documentation in a browser
```

Building the docs executes the examples to generate the example gallery under
`doc/build/html/_autoexamples`.
