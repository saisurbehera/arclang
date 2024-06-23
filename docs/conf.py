"""Sphinx configuration."""
project = "Arclang"
author = "Sai Surbehera"
copyright = "2024, Sai Surbehera"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
