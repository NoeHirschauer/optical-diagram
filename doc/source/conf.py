# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# make sure that the package is importable
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("../.."))

# %% -- Path setup --------------------------------------------------------------

# define the package path
ROOT_PATH = Path(os.path.abspath(os.path.join(os.path.dirname(__file__)))).parent.parent
EXAMPLES_PATH = ROOT_PATH / "examples"
PKG_PATH = ROOT_PATH / "src"

# if __name__ == "__main__":
#     print(f"ROOT_PATH     : {ROOT_PATH}")
#     print(f"PKG_PATH      : {PKG_PATH}")
#     print(f"EXAMPLES_PATH : {EXAMPLES_PATH}")

# %% -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "optical-diagram"
copyright = "2026, Noé Hirschauer"
author = "Noé Hirschauer"
github_repo = "https://github.com/NoeHirschauer/optical-diagram"

# read the version from the package __init__.py
from optical_diagram import __version__

release = version = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
]

# %% sphinx-gallery configuration

sphinx_gallery_conf = {
    "examples_dirs": EXAMPLES_PATH,  # path to example scripts
    "gallery_dirs": "_autoexamples",  # path to save generated examples
    "filename_pattern": r"\.py",  # include all .py files
    "line_numbers": False,  # do not add line numbers to code blocks
    "download_all_examples": False,  # do not add a "download all" link
    "write_computation_times": False,  # do not write computation times
    "remove_config_comments": True,  # remove comments that start with sphinx_gallery_
    "show_signature": False,  # do not show function signatures in the gallery
}


templates_path = ["_templates"]
exclude_patterns = [
    "**test/",
    "**docs/",
]

# Do not show param = Literal["a", "b"] but only param = "a" | "b"
python_display_short_literal_types = True

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If True, create TOC entries for domain objects (functions, classes, attributes, ...).
toc_object_entries = True

# How to display the members in the TOC. One of ['domain'] | 'hide' | 'all'.
toc_object_entries_show_parents = "hide"

# add line numbers to source code
viewcode_line_numbers = False

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "autolink"

# %% Napoleon configuration

napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_preprocess_types = True
napoleon_use_rtype = True
napoleon_type_aliases = {
    "np.ndarray": "~numpy.ndarray",
    "ndarray": "~numpy.ndarray",
    "ArrayLike": "~numpy.typing.ArrayLike",
    "plt.Axes": "~matplotlib.axes.Axes",
    "Literal": "~typing.Literal",
}
napoleon_custom_sections = [("See also", "See Also")]



# %% -- autoapi settings ------------------------------------------------------
# https://sphinx-autoapi.readthedocs.io/en/latest/index.html
autoapi_dirs = [PKG_PATH]
autoapi_ignore = ["**/tests/*", "**/doc/**", "**/examples/**"]
autoapi_template_dir = templates_path[0] + "/_autoapi/"
autoapi_python_class_content = "both"  # include both class and __init__ docstrings
autoapi_include_summaries = True  # include the first line of docstrings
autoapi_options = [
    "members",
    # "undoc-members",
    # "private-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]
autoapi_own_page_level = "class"
autoapi_member_order = "bysource"
# autoapi_keep_files = True  # wether to keep the generated rst files (for debugging)

# Where to show the type hints. {["signature"], "description", "none", "both"}
autodoc_typehints = "description"

# %% -- intersphinx mapping ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_theme_options = {
    "show_nav_level": 1,  # Only shows the top level by default
    "navigation_depth": 3,  # Stops the sidebar from expanding to level 3
    "show_toc_level": 2,  # show 2 levels in page toc by default
    "collapse_navigation": False,
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "footer_end": ["theme-version"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": github_repo,
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
    ],
}

# add css files to the build
html_css_files = [
    "custom_monospace_font.css",
    "remove_sphinx_gallery_download_links.css",
    "wider_article_content.css",
]
