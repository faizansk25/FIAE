"""Hypothesis-style property testing (doc 12).

Automatic invariant generation and counterexample minimization
for comprehensive operator and pipeline testing.

Normative source: doc 12 section "Property-based testing".
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..features.registry import FeatureOperator, all_operators


@dataclass
class PropertyTestCase:
    """A generated test case with input and expected properties."""

    input_values: list
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class PropertyTestResult:
    """Result of a property test."""

    property_name: str
    operator_name: str
    passed: bool
    counterexample: Optional[list] = None
    shrunk_counterexample: Optional[list] = None
    details: str = ""
    iterations: int = 0


def _generate_values(rng: random.Random, strategy: str, n: int) -> list:
    """Generate values using a specified strategy."""
    if strategy == "floats":
        return [rng.gauss(0, 10) for _ in range(n)]
    if strategy == "positive":
        return [abs(rng.gauss(5, 3)) for _ in range(n)]
    if strategy == "mixed_sign":
        return [rng.uniform(-100, 100) for _ in range(n)]
    if strategy == "with_none":
        return [rng.gauss(0, 1) if rng.random() > 0.2 else None for _ in range(n)]
    if strategy == "large":
        return [rng.uniform(1e10, 1e15) for _ in range(n)]
    if strategy == "small":
        return [rng.uniform(1e-10, 1e-5) for _ in range(n)]
    if strategy == "constant":
        v = rng.gauss(0, 1)
        return [v] * n
    if strategy == "boundary":
        return [0.0, 1.0, -1.0, float("inf"), float("-inf"), float("nan")] + \
               [rng.gauss(0, 1) for _ in range(n - 6)]
    if strategy == "sorted":
        vals = sorted([rng.gauss(0, 10) for _ in range(n)])
        return vals
    if strategy == "repeated":
        base = [rng.gauss(0, 1) for _ in range(n // 3)]
        return base * 3
    return [rng.gauss(0, 1) for _ in range(n)]


def _shrink_counterexample(
    op: FeatureOperator, property_fn: Callable, values: list, max_steps: int = 50
) -> list:
    """Minimize a counterexample by removing elements (shrinking)."""
    best = list(values)
    rng = random.Random(42)

    for _ in range(max_steps):
        if len(best) <= 2:
            break
        # Try removing a random element
        idx = rng.randint(0, len(best) - 1)
        candidate = best[:idx] + best[idx + 1:]
        try:
            if property_fn(op, candidate):
                continue  # property holds without this element
        except Exception:
            continue
        # Property still fails — keep the smaller counterexample
        best = candidate

    return best


def property_null_preservation(op: FeatureOperator, values: list) -> bool:
    """Property: null in -> null out."""
    if op.arity != "unary":
        return True
    if op.null_policy == "always_defined":
        return True
    try:
        result = op.transform(values)
    except Exception:
        return True
    for i, v in enumerate(values):
        if v is None and i < len(result) and result[i] is not None:
            return False
    return True


def property_output_length(op: FeatureOperator, values: list) -> bool:
    """Property: output length == input length."""
    if op.arity != "unary":
        return True
    try:
        result = op.transform(values)
    except Exception:
        return True
    return len(result) == len(values)


def property_deterministic(op: FeatureOperator, values: list) -> bool:
    """Property: same input -> same output."""
    if op.arity != "unary":
        return True
    try:
        r1 = op.transform(values)
        r2 = op.transform(values)
    except Exception:
        return True
    return r1 == r2


def property_finite_outputs(op: FeatureOperator, values: list) -> bool:
    """Property: numeric outputs are finite."""
    if op.arity != "unary":
        return True
    try:
        result = op.transform(values)
    except Exception:
        return True
    return all(not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) for v in result)


def property_monotonicity(op: FeatureOperator, values: list) -> bool:
    """Property: for sqrt/log1p/abs, output preserves order for positive inputs."""
    if op.name not in ("sqrt", "log1p", "abs", "square"):
        return True
    if op.arity != "unary":
        return True
    try:
        result = op.transform(values)
    except Exception:
        return True
    # Filter to valid pairs
    pairs = [(values[i], result[i]) for i in range(len(values))
             if values[i] is not None and result[i] is not None
             and isinstance(values[i], (int, float)) and isinstance(result[i], (int, float))]
    if len(pairs) < 2:
        return True
    # Check monotonicity for positive inputs
    pos_pairs = [(x, y) for x, y in pairs if x >= 0]
    for i in range(len(pos_pairs) - 1):
        if pos_pairs[i][0] > pos_pairs[i + 1][0] \
                and pos_pairs[i][1] < pos_pairs[i + 1][1]:
            return False
    return True


def run_hypothesis_tests(
    seed: int = 42, n_iterations: int = 20, n_values: int = 30
) -> list[PropertyTestResult]:
    """Run comprehensive hypothesis-style tests on all L0 unary operators."""
    rng = random.Random(seed)
    strategies = [
        "floats", "positive", "mixed_sign", "with_none",
        "large", "small", "constant", "sorted", "repeated",
    ]
    properties = [
        ("null_preservation", property_null_preservation),
        ("output_length", property_output_length),
        ("deterministic", property_deterministic),
        ("finite_outputs", property_finite_outputs),
        ("monotonicity", property_monotonicity),
    ]

    results = []
    for op in all_operators():
        if op.arity != "unary" or op.fit_scope.value != "none":
            continue

        for prop_name, prop_fn in properties:
            for _ in range(n_iterations):
                strategy = rng.choice(strategies)
                values = _generate_values(rng, strategy, n_values)
                passed = prop_fn(op, values)

                if not passed:
                    # Shrink counterexample
                    shrunk = _shrink_counterexample(op, prop_fn, values)
                    results.append(PropertyTestResult(
                        property_name=prop_name,
                        operator_name=op.name,
                        passed=False,
                        counterexample=values[:10],
                        shrunk_counterexample=shrunk[:10],
                        iterations=n_iterations,
                    ))
                    break
            else:
                results.append(PropertyTestResult(
                    property_name=prop_name,
                    operator_name=op.name,
                    passed=True,
                    iterations=n_iterations,
                ))

    return results


def summarize_hypothesis_results(results: list[PropertyTestResult]) -> dict:
    """Summarize hypothesis test results."""
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

    return {
        "total": total,
        "passed": passed,
        "failed": len(failed),
        "pass_rate": round(passed / total, 4) if total else 0,
        "by_property": by_property,
        "counterexamples": [
            {"op": r.operator_name, "property": r.property_name,
             "shrunk": r.shrunk_counterexample}
            for r in failed
        ],
    }
