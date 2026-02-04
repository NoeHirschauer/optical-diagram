# optical-diagram

An attempt at an object-oriented python module to trace simple optical diagrams, including basic optical elements and a utility for beam tracing

## Installation

Install from the project root.

- Editable/development install:
  
  ```bash
  pip install -e .
  ```

- Standard install:

  ```bash
  pip install .
  ```

- Install the documentation/build dependencies (recommended when building the docs):
 
  ```bash
  pip install .[docs]
  ```
  The available extras are:
  
  - `docs`: Sphinx and related packages for building the documentation.
  - `tests`: Testing packages like pytest for running the test suite.
  - `dev`: Combines `docs` and `tests` for development purposes.

## Basic usage

- Run one of the example scripts in the `examples/` folder to see typical usage (e.g. the Galilean telescope example):

  ```bash
  python -m examples.galilean_telescope
  ```
- Or import the package in Python:

  ```python
  from optical_diagram import OpticalTable, ConvergingLens, RayTracedBeam
  # ... create elements and render with OpticalTable ...
  ```

## Building the documentation

From the repository root you can build the Sphinx docs in two common ways:

- Using the Makefile (requires `make`):
  
  ```bash
  cd docs
  make html
  ```

- Using sphinx-build directly:
  
  ```bash
  sphinx-build -b html docs/source docs/build/html
  ```

### Notes

- The docs use `sphinx_gallery` to generate example galleries; building the docs will execute the examples and place generated output under `docs/build/html/auto_examples`.
- If you installed the docs extras with `pip install .[docs]`, the required Sphinx packages will be available locally.