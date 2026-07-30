import os
import sys
sys.path.insert(0, os.path.abspath('../../..'))
from pathlib import Path
# Load tomllib (built-in for Python 3.11+) or tomli (for Python < 3.11)
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
with open(pyproject_path, "rb") as f:
    pyproject_data = tomllib.load(f)
# Extract PEP 621 standardized fields
project_meta = pyproject_data.get("project", {})
project = project_meta.get("name", "scaleo")
author = ", ".join([a.get("name", "") for a in project_meta.get("authors", [])])
release = project_meta.get("version", "0.0.1")
version = release

# project = 'scaleo'
copyright = '2026, Denis Savitskiy'
# author = 'Denis Savitskiy'
# release = '0.0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    # 'sphinx.ext.viewcode',
    'sphinx.ext.napoleon'
]

autosummary_generate = True
# Custom settings for Napoleon
napoleon_google_docstring = True
napoleon_numpy_docstring = False  # Turn off NumPy style
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'haiku'
html_static_path = ['_static']
