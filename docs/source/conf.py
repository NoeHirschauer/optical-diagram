# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# make sure that the package is importable
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# %% -- Path setup --------------------------------------------------------------

# define the package path
PKG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# define the output path for the generated API docs. it will be docs/source/auto_api
APIDOC_PATH = os.path.join(os.path.dirname(__file__), "auto_api")

# %% -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "optical-diagram"
copyright = "2026, Noé Hirschauer"
author = "Noé Hirschauer"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
]

# %% sphinx-gallery configuration

sphinx_gallery_conf = {
    "examples_dirs": os.path.join(PKG_PATH, "examples"),  # path to example scripts
    "gallery_dirs": "auto_examples",  # path to save generated examples
    "filename_pattern": r"\.py",  # include all .py files
    "line_numbers": False,  # do not add line numbers to code blocks
    "download_all_examples": False,  # do not add a "download all" link
    "write_computation_times": False,  # do not write computation times
    "remove_config_comments": True,  # remove comments that start with sphinx_gallery_
    "show_signature": False,  # do not show function signatures in the gallery
}

# %% autodoc configuration

templates_path = ["_templates"]
exclude_patterns = [
    "**test/",
    "**docs/",
]

autodoc_default_options = {
    "member-order": "bysource",
    "show-inheritance": True,
    "inherited-members": True,
    "imported-members": False,
}

# Specify what content will be inserted into the main body of an autoclass directive.
# One of "class", "init", "both".
autoclass_content = "both"

# How to represent typehints in the documentation. "signature", "description" or "none".
autodoc_typehints = "description"

# Do not show param = Literal["a", "b"] but only param = "a" | "b"
python_display_short_literal_types = True

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If True, create TOC entries for domain objects (functions, classes, attributes, ...).
toc_object_entries = True

# add line numbers to source code
viewcode_line_numbers = True

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "autolink"

# %% Napoleon configuration

napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_include_init_with_doc = False  # already merged with class doc (autoclass)
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_rtype = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_theme_options = {
    "show_nav_level": 1,    # Only shows the top level by default
    "navigation_depth": 3,  # Stops the sidebar from expanding to level 3
    "show_toc_level": 1,    # show 1 level in page toc by default
    "collapse_navigation": False,
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "footer_end": ["theme-version"],
}


# define a custom setup function that runs apidoc before building the docs
def _run_apidoc(_):
    from sphinx.ext.apidoc import main as apidoc_main

    # Generate the API documentation
    apidoc_main(
        [
            "--no-toc",
            "--force",
            "--separate",
            "--module-first",
            "-o",
            APIDOC_PATH,
            PKG_PATH,
        ]
    )


def setup(app):
    app.connect("builder-inited", _run_apidoc)
