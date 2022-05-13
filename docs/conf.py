# Configuration file for the Sphinx documentation builder.
# - Documentation: <http://www.sphinx-doc.org/en/master/config>

# Append path to include source code.
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# Project information.
project = "MESMO"
copyright = "2018-2022, TUMCREATE"
author = "MESMO authors"

# Extensions.
# - Add any Sphinx extension module names here, as strings. They can be
#   extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
#   ones.
extensions = [
    "myst_parser",  # Markdown parser.
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_multiversion",
]

# Extension settings.
# - sphinx.ext.autodoc: <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>
# - sphinx.ext.napoleon: <https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>
autodoc_default_options = {"members": None, "show-inheritance": None, "member-order": "bysource"}
autodoc_typehints = "description"
autodoc_mock_imports = [
    # Please note: Do not remove deprecated dependencies, because these are still needed for docs of previous versions.
    "cobmo",
    "cvxpy",
    "cv2",
    "diskcache",  # Deprecated.
    "dill",  # Deprecated.
    "gurobipy",
    "kaleido",
    "matplotlib",
    "multimethod",
    "multiprocess",  # Deprecated.
    "networkx",
    "natsort",
    "numpy",
    "opendssdirect",
    "pandas",
    "plotly",
    "pyyaml",
    "pyomo",  # Deprecated.
    "ray",
    "scipy",
    "tqdm",
]
napoleon_use_ivar = True

# Source settings.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"

# Exclude settings.
# - List of patterns, relative to source directory, that match files and
#   directories to ignore when looking for source files.
#   This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

# HTML theme settings.
# - The theme to use for HTML and HTML Help pages.  See the documentation for
#   a list of builtin themes.
html_theme = "furo"
html_title = "MESMO"
html_logo = "assets/mesmo_icon.png"
html_favicon = "assets/favicon.ico"
templates_path = ["templates"]
html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/versions.html",
        "sidebar/scroll-end.html",
    ]
}
html_static_path = ["static"]
html_css_files = ["css/custom.css"]

# Sphinx multiversion settings.
# <https://holzhaus.github.io/sphinx-multiversion/master/configuration.html>
# - Explicitly include all branches, tags from all remotes.
smv_tag_whitelist = r"^.*$"
smv_branch_whitelist = r"^.*$"
smv_remote_whitelist = r"^.*$"

# MyST markdown parser settings.
# <https://myst-parser.readthedocs.io/en/latest/using/intro.html#sphinx-configuration-options>
myst_heading_anchors = 4
myst_enable_extensions = [
    # 'amsmath',
    "colon_fence",
    # 'deflist',
    # 'dollarmath',
    # 'html_admonition',
    "html_image",
    "linkify",
    # 'replacements',
    # 'smartquotes',
    # 'substitution',
    # 'tasklist',
]

# Recommonmark settings.
# - Deprecated, but kept here for backwards compatibility with docs.
from recommonmark.transform import AutoStructify


def setup(app):
    app.add_transform(AutoStructify)
