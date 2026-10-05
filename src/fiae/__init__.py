"""FIAE — Feature Intelligence & Architecture Engine.

Core imports require only the Python standard library (NFR-002, doc 01).
Heavy capabilities are optional adapters behind dependency tiers.

Purple identity: #C084FC → #6D28D9
"""

__version__ = "0.1.0"

# Expose the CLI color system for branded outputs
from . import cli_colors

# Public entry points for library use. Both are standard-library-only at
# import time; scikit-learn is detected, never required (NFR-002).
from .fitted_pipeline import FittedPipeline
from .search.triggers import FeatureProposal
from .sklearn_api import SklearnFeatureTransformer

__all__ = [
    "FittedPipeline",
    "FeatureProposal",
    "SklearnFeatureTransformer",
    "cli_colors",
]
