"""FIAE — Feature Intelligence & Architecture Engine.

Core imports require only the Python standard library (NFR-002, doc 01).
Heavy capabilities are optional adapters behind dependency tiers.

Purple identity: #C084FC → #6D28D9
"""

__version__ = "0.0.1"

# Expose the CLI color system for branded outputs
from . import cli_colors
