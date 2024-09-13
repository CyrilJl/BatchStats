# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path

# Récupérer le chemin absolu du dossier parent du dossier actuel
parent_dir = Path(__file__).resolve().parent.parent.parent
# Ajouter le dossier parent au chemin
sys.path.insert(0, str(parent_dir))


project = 'batchstats'
copyright = '2024, Cyril Joly'
author = 'Cyril Joly'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.napoleon', 'sphinx_copybutton', 'sphinx_favicon']

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_logo = "_static/logo_batchstats.svg"
html_context = {"default_mode": "light"}
html_sidebars = {"**": []}

pygments_style = 'vs'
