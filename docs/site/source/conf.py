"""Configuration file for the Sphinx documentation builder."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

project = "mlflow-monitor"
copyright = "2026, Zheng Shi"
author = "Zheng Shi"
release = "0.1.0"

autodoc_mock_imports = ["haystack"]

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "furo"
html_static_path = ["_static"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

autodoc_member_order = "bysource"
