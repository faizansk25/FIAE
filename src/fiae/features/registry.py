"""Feature operation registry (docs 04, 05, FR-007).

Every operation declares the catalog contract:
input semantic types, output semantic type, purpose, preconditions, learned
fit state, exact transform, null/invalid behavior, leakage class, cost shape,
generation trigger, rejection conditions, validation requirement, inference
state requirement, and its mandatory tests are defined per doc 12.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..contracts import FitScope, LeakageClass, TargetPermission
from ..errors import ErrorCode, FIAEError


@dataclass(frozen=True)
class FeatureOperator:
    """Declaration of one feature operation (doc 05 catalog contract)."""

    name: str
    family: str
    arity: str  # "unary" | "binary" | "datetime" | "temporal"
    input_types: tuple[str, ...]
    output_type: str
    purpose: str
    preconditions: tuple[str, ...]
    fit_scope: FitScope
    null_policy: str
    leakage_class: LeakageClass
    target_permission: TargetPermission
    cost_shape: str
    generation_trigger: str
    rejection_conditions: tuple[str, ...]
    validation: str
    inference_requirement: str
    transform: Callable[..., list]
    mandatory_tests: tuple[str, ...] = ()
    commutative: bool = False
    # Optional static domain check: (values..., params) -> rejection reason | None
    domain_check: Optional[Callable[..., Optional[str]]] = None
    # Temporal semantics: None | "sequential" (sorted-time ops, no-future invariant)
    time_semantics: Optional[str] = None


_REGISTRY: dict[str, FeatureOperator] = {}


def register(op: FeatureOperator) -> FeatureOperator:
    if op.name in _REGISTRY:
        raise FIAEError(
            code=ErrorCode.FEATURE_PRECONDITION_FAILED,
            safe_message=f"duplicate operator registration: {op.name}",
            component="features.registry",
        )
    _REGISTRY[op.name] = op
    return op


def get_operator(name: str) -> FeatureOperator:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise FIAEError(
            code=ErrorCode.FEATURE_PRECONDITION_FAILED,
            safe_message=f"unknown feature operator: {name}",
            component="features.registry",
            subject_id=name,
        ) from None


def all_operators() -> list[FeatureOperator]:
    return sorted(_REGISTRY.values(), key=lambda o: o.name)


def check_domain(op: FeatureOperator, values: tuple, params: dict[str, Any]) -> Optional[str]:
    """Run the static domain/precondition check; return rejection reason or None."""
    if op.domain_check is None:
        return None
    return op.domain_check(*values, **(params or {}))
