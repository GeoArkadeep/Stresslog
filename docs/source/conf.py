import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath('../../'))
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'stresslog'
copyright = 'ROCK LAB PRIVATE LIMITED'
author = 'Arkadeep Ghosh'
release = '1.6.12'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',        # Automatically document from docstrings
    'sphinx.ext.napoleon',       # Support for NumPy and Google-style docstrings
    #'sphinx.ext.viewcode',       # Add links to source code
    ]


templates_path = ['_templates']
exclude_patterns = []

# Autodoc settings
autodoc_preserve_defaults = True  # Shows string literals instead of evaluated values
autodoc_default_options = {
    'members': True,
    'undoc-members': False,       # Exclude undocumented members
    'private-members': False,     # Exclude private members
    'special-members': False,     # Exclude special methods like __init__
    'inherited-members': False,   # Exclude inherited methods
    'imported-members': False,    # Exclude members not defined in the module
}
autodoc_inherit_docstrings = False  # Avoid inheriting docstrings from parents



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = 'furo'
