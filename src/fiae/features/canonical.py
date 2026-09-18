"""Canonicalization of feature expressions (doc 04).

Rules:
- ``a+b`` and ``b+a`` share a signature (commutative operand fold);
- ``min(a,b)`` and ``min(b,a)`` share a signature;
- ``x+0 -> x``; ``x*1 -> x``; ``abs(abs(x)) -> abs(x)`` (identity collapse is
  applied by the DAG which can see the referenced nodes);
- exact duplicate DAG nodes collapse (same signature -> same feature id);
- canonical signature is deterministic under declared inputs (NFR-004).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ..contracts import FeatureNode
from ..ids import content_hash
from .registry import FeatureOperator, get_operator

if TYPE_CHECKING:  # pragma: no cover
    pass

# Literal constants recognized for the x+0 / x*1 identity rules.
_ZERO_LITERALS = frozenset({"0", "0.0", "-0.0"})
_ONE_LITERALS = frozenset({"1", "1.0"})


def canonical_inputs(op: FeatureOperator, inputs: list[str]) -> list[str]:
    """Fold commutative operand order into a canonical form."""
    if op.commutative:
        return sorted(inputs)
    return list(inputs)


def canonical_params(params: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Stable parameter mapping (content_hash performs canonical serialization)."""
    return dict(sorted((params or {}).items()))


def feature_signature(
    op_name: str, inputs: list[str], params: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Canonical semantic signature of a feature expression (doc 04)."""
    op = get_operator(op_name)
    return {
        "op": op.name,
        "inputs": canonical_inputs(op, inputs),
        "params": canonical_params(params),
    }


def signature_hash(op_name: str, inputs: list[str], params: Optional[dict[str, Any]] = None) -> str:
    return content_hash(feature_signature(op_name, inputs, params))


def make_feature_node(
    op_name: str,
    inputs: list[str],
    params: Optional[dict[str, Any]] = None,
) -> FeatureNode:
    """Build a canonical FeatureNode from an operation declaration.

    The feature id is the content hash of the canonical signature, so equal
    expressions collapse to one node and identical inputs always produce the
    same identifier (doc 04 FeatureNode contract, NFR-004).
    """
    op = get_operator(op_name)
    sig = feature_signature(op_name, inputs, params)
    node = FeatureNode(
        feature_id="f_" + content_hash(sig)[:16],
        operator=op.name,
        inputs=list(sig["inputs"]),
        params=dict(sig["params"]),
        fit_scope=op.fit_scope,
        target_permission=op.target_permission,
        time_semantics=op.time_semantics,
        null_policy=op.null_policy,
        cost_hint={
            "cost_shape": op.cost_shape,
            "output_type": op.output_type,
            "estimated_cpu_seconds_per_1k_rows": _estimate_cpu(op),
            "estimated_peak_ram_mb": 16,
        },
    )
    node.lineage_hash = signature_hash(op.name, node.inputs, node.params)
    return node


def _estimate_cpu(op: FeatureOperator) -> float:
    """Coarse O(n)-family cost estimate used by the static resource gate (F1)."""
    shape = op.cost_shape.upper()
    if "N LOG N" in shape or "NLOGN" in shape:
        return 0.05
    if shape.startswith("O(") and "K" in shape:
        return 0.2
    return 0.01


def drop_identity_inputs(
    op_name: str, inputs: list[str]
) -> Optional[list[str]]:
    """Apply ``x+0 -> x`` and ``x*1 -> x``.

    Returns the reduced input list, the same list when no identity applies,
    or ``None`` when every input is a neutral constant (degenerate).
    """
    op = get_operator(op_name)
    if op.name == "sum":
        kept = [i for i in inputs if i not in _ZERO_LITERALS]
        # a sum of only zeros is not neutral -> keep as-is
        return kept or list(inputs)
    if op.name == "product":
        kept = [i for i in inputs if i not in _ONE_LITERALS]
        return kept or list(inputs)
    return list(inputs)
