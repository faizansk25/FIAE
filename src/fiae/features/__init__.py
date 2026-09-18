"""Feature engineering subsystem — docs 04, 05, 12.

Public API
----------
- ``FeatureOperator`` / ``register`` / ``get_operator`` / ``all_operators`` /
  ``check_domain`` — the operator registry contract.
- ``make_feature_node`` / ``canonical_inputs`` / ``feature_signature`` /
  ``signature_hash`` / ``drop_identity_inputs`` — canonicalization helpers.
- ``FeatureDAG`` — acyclic feature graph with duplicate collapse.

Importing this package triggers registration of the built-in operator catalog
(numeric, datetime, temporal) so that ``all_operators()`` is immediately
usable.
"""

from __future__ import annotations

from . import ops_categorical
from . import ops_datetime
from . import ops_fitted
from . import ops_group
from . import ops_numeric
from . import ops_temporal
from . import ops_text
try:
    from . import ops_sklearn
    from . import ops_model_informed
except ImportError:
    pass  # optional tier1 dependency not installed
from .canonical import (
    canonical_inputs,
    drop_identity_inputs,
    feature_signature,
    make_feature_node,
    signature_hash,
)
from .dag import FeatureDAG
from .registry import (
    FeatureOperator,
    all_operators,
    check_domain,
    get_operator,
    register,
)

__all__ = [
    "FeatureDAG",
    "FeatureOperator",
    "all_operators",
    "canonical_inputs",
    "check_domain",
    "drop_identity_inputs",
    "feature_signature",
    "get_operator",
    "make_feature_node",
    "register",
    "signature_hash",
]
