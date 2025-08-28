# Configuration file for the Sphinx documentation builder.
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "LLM-LoRA Framework"
copyright = "2025, Timur Komolov"
author = "Timur Komolov"
release = "0.1.0"
version = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "autodocsumm",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.githubpages",
]

# Autodoc configuration
autodoc_default_options = {
    "autosummary": True,
    "members": True,
    "show-inheritance": True,
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Napoleon configuration for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "transformers": ("https://huggingface.co/docs/transformers/main/en/", None),
}

# Templates and exclusions
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Source file suffixes
source_suffix = {
    ".rst": None,
    ".md": "myst_parser",
}

# Master document
master_doc = "index"

# Language
language = "en"

# TODO extension
todo_include_todos = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Theme options
html_theme_options = {
    "canonical_url": "",
    "analytics_id": "",
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "#2980B9",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Custom sidebar
html_sidebars = {
    "**": [
        "relations.html",  # needs 'show_related': True theme option to display
        "searchbox.html",
    ]
}

# HTML context
html_context = {
    "display_github": True,
    "github_user": "your-username",
    "github_repo": "llm-lora",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# Additional HTML files
html_additional_pages = {}

# HTML title
html_title = f"{project} v{version} Documentation"

# HTML short title
html_short_title = f"{project} v{version}"

# HTML favicon
# html_favicon = "_static/favicon.ico"

# HTML logo
# html_logo = "_static/logo.png"

# Show source link
html_show_sourcelink = True

# Show sphinx link
html_show_sphinx = True

# Show copyright
html_show_copyright = True

# Last updated format
html_last_updated_fmt = "%b %d, %Y"

# Use index
html_use_index = True

# Split index
html_split_index = False

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    "papersize": "letterpaper",
    # The font size ('10pt', '11pt' or '12pt').
    "pointsize": "10pt",
    # Additional stuff for the LaTeX preamble.
    "preamble": "",
    # Latex figure (float) alignment
    "figure_align": "htbp",
}

# Grouping the document tree into LaTeX files
latex_documents = [
    (master_doc, "llm-lora.tex", f"{project} Documentation", author, "manual"),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page
man_pages = [(master_doc, "llm-lora", f"{project} Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files
texinfo_documents = [
    (
        master_doc,
        "llm-lora",
        f"{project} Documentation",
        author,
        "llm-lora",
        "Advanced Language Model Fine-tuning Framework",
        "Miscellaneous",
    ),
]

# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
epub_exclude_files = ["search.html"]

# -- Extension configuration -------------------------------------------------

# autodoc configuration
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# autosummary configuration
autosummary_generate = True
autosummary_imported_members = False

# viewcode configuration
viewcode_import = True
