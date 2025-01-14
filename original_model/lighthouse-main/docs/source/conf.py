"""Configuration file for the Sphinx documentation builder.

Author: Etienne de Montalivet
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

from lighthouse import __version__

project = "lighthouse"
copyright = "2024, Lighthouse collaborators"
author = "Lighthouse collaborators"
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# poetry add sphinx sphinx-rtd-theme sphinx-copybutton numpydoc
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "numpydoc",
    "sphinx.ext.autosectionlabel",
]

# copy code button settings
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\n"

# numpydoc_class_members_toctree = True
# numpydoc_show_class_members = True
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 10

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autosummary_imported_members = False
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
exclude_patterns = ["_build", "_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
source_suffix = [".rst", ".md"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/WyssCenter/lighthouse",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
