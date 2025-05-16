# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
# Add the source directory to the path so Sphinx can find the modules
import os
import sys

project = "Jua Python SDK"
copyright = "2025, Jua.ai"
author = "Jua.ai"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add the _ext directory to sys.path so that our custom extensions can be found
sys.path.insert(0, os.path.abspath("."))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "autoapi.extension",
    "sphinx_markdown_builder",
    "myst_parser",
    "sphinx_autodoc_typehints",
    # Custom extension for better function signature formatting in Markdown
    "_ext.function_sig",
]

templates_path = ["_templates"]
exclude_patterns = []

sys.path.insert(0, os.path.abspath("../../src"))

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable/", None),
}

# -- Options for autodoc extension ------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
}

# -- Options for autoapi extension ------------------------------------------
autoapi_type = "python"
autoapi_dirs = ["../../src/jua"]
autoapi_ignore = ["*/scripts/*"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
    "group-by-class",
]
autoapi_add_toctree_entry = True
autoapi_template_dir = "_templates/autoapi"
autoapi_own_page_level = "class"
autoapi_root = "sdk"

# Configure how parent names are shown in documentation TOC
toc_object_entries_show_parents = "hide"

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# -- Enable line wrapping for long function signatures -----------------------
# Configures automatic line-breaking for function signatures that exceed
# the specified length limit. Each parameter will be displayed on a new line.
maximum_signature_line_length = 68

# Python domain-specific configuration to improve signature formatting
python_maximum_signature_line_length = 68
python_display_short_literal_types = True

# -- Markdown builder configuration ------------------------------------------
# Configure any specific settings for the Markdown builder
markdown_anchor_signatures = True
markdown_anchor_sections = True
markdown_http_base = None
markdown_docinfo = True
markdown_bullet = "*"

# Create CSS file for custom styles if it doesn't exist yet
html_css_files = ["css/custom.css"]
