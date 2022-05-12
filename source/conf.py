# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))

# -- Project information -----------------------------------------------------

project = 'Machine learning toolbox for WPT and EEMD'
copyright = '2020, Melih C. Yesilli, Firas A. Khasawneh, Andreas Otto'
author = 'Melih C. Yesilli, Firas A. Khasawneh, Andreas Otto'

# The short X.Y version
version = ''
# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------
extensions = ['sphinx.ext.autodoc', 
              'sphinx.ext.coverage', 
              'sphinx.ext.napoleon',
              'sphinx_rtd_theme',
              'matplotlib.sphinxext.mathmpl',
              'matplotlib.sphinxext.plot_directive',
              'sphinx.ext.autosummary',
              'sphinx.ext.doctest',
              'sphinx.ext.intersphinx',
              'sphinx.ext.todo',
              'sphinx.ext.mathjax',
              'sphinx.ext.ifconfig',
              'sphinx.ext.viewcode',
              'sphinx.ext.githubpages',
              'sphinxcontrib.bibtex',]
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
bibtex_bibfiles = ['references.bib']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
# html_logo = 'logo.png'
html_theme_options = {
    'canonical_url': '',
    'analytics_id': 'UA-XXXXXXX-1',  #  Provided by Google in your dashboard
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'style_nav_header_background': 'white',
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 3,
    'includehidden': True,
    'titles_only': False,

}
html_static_path = ['_static']
# enable numbering figures
numfig = True