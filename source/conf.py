# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information




project = 'Demeter'
copyright = '2023, Kanishka B. Narayan'
author = 'Kanishka B. Narayan'
author = 'Chris R. Vernon'
release = '1.3.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


import os
import sys
sys.path.insert(0, os.path.abspath('../../demeter/'))

extensions = ['sphinx.ext.autodoc','sphinx.ext.napoleon','m2r2']

templates_path = ['_templates']
exclude_patterns = []

source_suffix = [".rst", ".md"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
