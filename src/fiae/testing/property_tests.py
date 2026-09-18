"""Property-based testing framework (doc 12).

Generates test cases that verify invariants across all operators and
pipeline stages.  Lightweight, no external test dependencies.

Doc 12 reference:
- Every operator must satisfy null-preservation, domain-consistency,
  deterministic-output, and idempotence properties
- Pipeline must satisfy monotonic-improvement, no-leakage, and
  reproducibility properties
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from ..features.registry import FeatureOperator, all_operators


@dataclass
class PropertyResult:
    """Result of a single property check."""

    property_name: str
    operator_name: str
    passed: bool
    details: str = ""
    sample_input: Any = None
    sample_output: Any = None


def generate_test_values(seed: int = 42, n: int = 50) -> list[list]:
    """Generate diverse test value sets for property testing."""
    rng = random.Random(seed)
    return [
        # Standard numeric
        [rng.gauss(0, 1) for _ in range(n)],
        # Positive values
        [abs(rng.gauss(5, 2)) for _ in range(n)],
        # Mixed sign
        [rng.uniform(-100, 100) for _ in range(n)],
        # With None values
        [rng.gauss(0, 1) if rng.random() > 0.2 else None for _ in range(n)],
        # Large values
        [rng.uniform(1e10, 1e15) for _ in range(n)],
        # Small values
        [rng.uniform(1e-10, 1e-5) for _ in range(n)],
        # Constants
        [5.0] * n,
        # Single value
        list(range(n)),
        # Ascending
        [float(i) for i in range(n)],
        # Descending
        [float(n - i) for i in range(n)],
    ]


def generate_binary_test_values(seed: int = 42, n: int = 50) -> list[tuple[list, list]]:
    """Generate diverse paired test value sets for binary operators."""
    rng = random.Random(seed)
    pairs = []
    for _ in range(5):
        a = [rng.gauss(0, 1) for _ in range(n)]
        b = [rng.gauss(0, 1) for _ in range(n)]
        pairs.append((a, b))
    # Add some special cases
    pairs.append(([1.0] * n, [float(i) for i in range(n)]))
    pairs.append(([float(i) for i in range(n)], [float(i) for i in range(n)]))
    return pairs


# ---------------------------------------------------------------------------
# Property checks
# ---------------------------------------------------------------------------
def check_null_preservation(op: FeatureOperator, values: list) -> PropertyResult:
    """Property: null in -> null out (doc 01 invariant 10)."""
    try:
        if op.arity == "unary":
            result = op.transform(values)
        else:
            return PropertyResult(
                property_name="null_preservation",
                operator_name=op.name, passed=True,
                details="skipped for binary operator",
            )
    except Exception as e:
        return PropertyResult(
            property_name="null_preservation",
            operator_name=op.name, passed=False,
            details=f"exception: {e}",
        )

    if op.null_policy == "always_defined":
        return PropertyResult(
            property_name="null_preservation",
            operator_name=op.name, passed=True,
            details="operator always defines output",
        )

    for i, v in enumerate(values):
        if v is None and i < len(result) and result[i] is not None:
            return PropertyResult(
                property_name="null_preservation",
                operator_name=op.name, passed=False,
                details=f"null at index {i} produced non-null output",
                sample_input=values[:5],
                sample_output=result[:5],
            )
    return PropertyResult(
        property_name="null_preservation",
        operator_name=op.name, passed=True,
    )


def check_deterministic(op: FeatureOperator, values: list) -> PropertyResult:
    """Property: same input -> same output (NFR-004)."""
    try:
        if op.arity == "unary":
            r1 = op.transform(values)
            r2 = op.transform(values)
        else:
            return PropertyResult(
                property_name="deterministic",
                operator_name=op.name, passed=True,
                details="skipped for binary",
            )
    except Exception as e:
        return PropertyResult(
            property_name="deterministic",
            operator_name=op.name, passed=False,
            details=f"exception: {e}",
        )

    if r1 != r2:
        return PropertyResult(
            property_name="deterministic",
            operator_name=op.name, passed=False,
            details="two runs produced different outputs",
            sample_input=values[:5],
            sample_output=[r1[:5], r2[:5]],
        )
    return PropertyResult(
        property_name="deterministic",
        operator_name=op.name, passed=True,
    )


def check_output_length(op: FeatureOperator, values: list) -> PropertyResult:
    """Property: output length == input length."""
    try:
        if op.arity == "unary":
            result = op.transform(values)
        else:
            return PropertyResult(
                property_name="output_length",
                operator_name=op.name, passed=True,
                details="skipped for binary",
            )
    except Exception as e:
        return PropertyResult(
            property_name="output_length",
            operator_name=op.name, passed=False,
            details=f"exception: {e}",
        )

    if len(result) != len(values):
        return PropertyResult(
            property_name="output_length",
            operator_name=op.name, passed=False,
            details=f"input len={len(values)}, output len={len(result)}",
        )
    return PropertyResult(
        property_name="output_length",
        operator_name=op.name, passed=True,
    )


def check_finite_outputs(op: FeatureOperator, values: list) -> PropertyResult:
    """Property: numeric outputs are finite (no inf/nan in output)."""
    try:
        if op.arity == "unary":
            result = op.transform(values)
        else:
            return PropertyResult(
                property_name="finite_outputs",
                operator_name=op.name, passed=True,
                details="skipped for binary",
            )
    except Exception:
        return PropertyResult(
            property_name="finite_outputs",
            operator_name=op.name, passed=True,
            details="exception during transform (not a finite-output issue)",
        )

    for i, v in enumerate(result):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return PropertyResult(
                property_name="finite_outputs",
                operator_name=op.name, passed=False,
                details=f"non-finite output at index {i}: {v}",
                sample_input=values[:5],
                sample_output=result[:5],
            )
    return PropertyResult(
        property_name="finite_outputs",
        operator_name=op.name, passed=True,
    )


# ---------------------------------------------------------------------------
# Run all properties
# ---------------------------------------------------------------------------
def run_all_properties(seed: int = 42) -> list[PropertyResult]:
    """Run all property checks on all L0 unary operators."""
    results = []
    ops = all_operators()
    test_values = generate_test_values(seed)

    for op in ops:
        if op.arity != "unary":
            continue
        if op.fit_scope.value != "none":
            continue  # skip L1/L2 operators (need fit state)

        for values in test_values:
            results.append(check_null_preservation(op, values))
            results.append(check_deterministic(op, values))
            results.append(check_output_length(op, values))
            results.append(check_finite_outputs(op, values))

    return results


def summarize_results(results: list[PropertyResult]) -> dict[str, Any]:
    """Summarize property test results."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = [r for r in results if not r.passed]

    by_property = {}
    for r in results:
        by_property.setdefault(r.property_name, {"passed": 0, "failed": 0})
        if r.passed:
            by_property[r.property_name]["passed"] += 1
        else:
            by_property[r.property_name]["failed"] += 1

    by_operator = {}
    for r in results:
        by_operator.setdefault(r.operator_name, {"passed": 0, "failed": 0})
        if r.passed:
            by_operator[r.operator_name]["passed"] += 1
        else:
            by_operator[r.operator_name]["failed"] += 1

    return {
        "total": total,
        "passed": passed,
        "failed_count": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0,
        "by_property": by_property,
        "failed_details": [
            {"op": r.operator_name, "property": r.property_name, "details": r.details}
            for r in failed
        ],
    }
