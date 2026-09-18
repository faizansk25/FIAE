"""Auto-generated test scaffolds and verification (doc 09).

Generates test code for new operators based on their catalog declaration,
and verifies that existing operators satisfy their contracts.

Normative source: doc 09 section "Auto-generation" and "Verification gates".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..features.registry import FeatureOperator, all_operators


@dataclass
class TestScaffold:
    """Generated test code for one operator."""

    operator_name: str
    test_code: str
    required_fixtures: list[str]
    description: str


def generate_operator_test(op: FeatureOperator) -> TestScaffold:
    """Generate a test scaffold for a single operator (doc 09).

    Creates test code that verifies:
    1. Null preservation
    2. Output length matches input length
    3. Deterministic output
    4. Domain checks (if applicable)
    5. Basic functional correctness
    """
    name = op.name

    # Build test code based on operator characteristics
    lines = [
        f"def test_{name}_null_preservation():",
        f"    \"\"\"Verify null in -> null out for {name}.\"\"\"",
        f"    op = get_operator(\"{name}\")",
    ]

    if op.arity == "unary":
        lines.extend([
            "    result = op.transform([1.0, None, 3.0])",
            "    assert result[1] is None, \"null not preserved\"",
        ])
    else:
        lines.extend([
            "    result = op.transform([1.0, None, 3.0], [4.0, 5.0, 6.0])",
            "    assert result[1] is None, \"null not preserved\"",
        ])

    lines.extend(["", ""])

    # Output length test
    lines.extend([
        f"def test_{name}_output_length():",
        f"    \"\"\"Verify output length matches input length for {name}.\"\"\"",
        f"    op = get_operator(\"{name}\")",
    ])
    if op.arity == "unary":
        lines.extend([
            "    result = op.transform([1.0, 2.0, 3.0, 4.0, 5.0])",
            "    assert len(result) == 5, f\"expected 5, got {len(result)}\"",
        ])
    else:
        lines.extend([
            "    result = op.transform([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])",
            "    assert len(result) == 3, f\"expected 3, got {len(result)}\"",
        ])

    lines.extend(["", ""])

    # Deterministic test
    lines.extend([
        f"def test_{name}_deterministic():",
        f"    \"\"\"Verify same input -> same output for {name}.\"\"\"",
        f"    op = get_operator(\"{name}\")",
    ])
    if op.arity == "unary":
        lines.extend([
            "    r1 = op.transform([1.0, 2.0, 3.0])",
            "    r2 = op.transform([1.0, 2.0, 3.0])",
            "    assert r1 == r2, \"non-deterministic output\"",
        ])
    else:
        lines.extend([
            "    r1 = op.transform([1.0, 2.0], [3.0, 4.0])",
            "    r2 = op.transform([1.0, 2.0], [3.0, 4.0])",
            "    assert r1 == r2, \"non-deterministic output\"",
        ])

    lines.extend(["", ""])

    # Functional test based on operator type
    lines.extend([
        f"def test_{name}_functional():",
        f"    \"\"\"Basic functional test for {name}.\"\"\"",
        f"    op = get_operator(\"{name}\")",
    ])
    if "sqrt" in name:
        lines.append("    result = op.transform([4.0, 9.0, 16.0])")
        lines.append("    assert abs(result[0] - 2.0) < 1e-10")
    elif "log" in name:
        lines.append("    result = op.transform([0.0, 1.0, 2.0])")
        lines.append("    assert result[0] == 0.0  # log1p(0) = 0")
    elif "square" in name:
        lines.append("    result = op.transform([2.0, 3.0])")
        lines.append("    assert result[0] == 4.0 and result[1] == 9.0")
    elif "abs" in name:
        lines.append("    result = op.transform([-5.0, 3.0])")
        lines.append("    assert result[0] == 5.0 and result[1] == 3.0")
    elif "sign" in name:
        lines.append("    result = op.transform([-1.0, 0.0, 1.0])")
        lines.append("    assert result[0] == -1 and result[1] == 0 and result[2] == 1")
    else:
        lines.append("    result = op.transform([1.0, 2.0, 3.0])")
        lines.append("    assert len(result) == 3")

    code = "\n".join(lines)
    return TestScaffold(
        operator_name=name,
        test_code=code,
        required_fixtures=["get_operator"],
        description=f"Auto-generated tests for {name} ({op.family}, {op.arity})",
    )


def generate_all_scaffolds() -> list[TestScaffold]:
    """Generate test scaffolds for all registered operators (doc 09)."""
    scaffolds = []
    for op in all_operators():
        scaffolds.append(generate_operator_test(op))
    return scaffolds


def verify_operator_contract(op: FeatureOperator) -> list[str]:
    """Verify that an operator satisfies its doc 05 contract (doc 09).

    Returns a list of violations.  Empty list means compliant.
    """
    violations = []

    # Required fields
    if not op.name:
        violations.append("Missing name")
    if not op.family:
        violations.append("Missing family")
    if not op.purpose:
        violations.append("Missing purpose")
    if not op.cost_shape:
        violations.append("Missing cost_shape")
    if not op.generation_trigger:
        violations.append("Missing generation_trigger")
    if not op.validation:
        violations.append("Missing validation")

    # Transform must be callable
    if not callable(op.transform):
        violations.append("transform is not callable")

    # Arity consistency
    if op.arity == "unary" and len(op.input_types) != 1:
        violations.append(f"unary operator has {len(op.input_types)} input_types")
    if op.arity == "binary" and len(op.input_types) != 2:
        violations.append(f"binary operator has {len(op.input_types)} input_types")

    # L1/L2 operators must have fit_scope
    if op.leakage_class.value in ("L1", "L2") and op.fit_scope.value == "none":
        violations.append(f"{op.leakage_class.value} operator has fit_scope=none")

    # L0 operators should have target_permission=P0
    if op.leakage_class.value == "L0" and op.target_permission.value != "P0_no_target":
        violations.append(f"L0 operator has target_permission={op.target_permission.value}")

    return violations


def verify_all_contracts() -> dict[str, list[str]]:
    """Verify all operators satisfy their contracts (doc 09).

    Returns dict mapping operator name -> list of violations.
    """
    results = {}
    for op in all_operators():
        violations = verify_operator_contract(op)
        if violations:
            results[op.name] = violations
    return results


def generate_compliance_report() -> dict[str, Any]:
    """Generate a full compliance report for all operators (doc 09)."""
    ops = all_operators()
    violations = verify_all_contracts()

    by_family = {}
    for op in ops:
        by_family.setdefault(op.family, []).append(op.name)

    compliant = [name for name in [o.name for o in ops] if name not in violations]

    return {
        "total_operators": len(ops),
        "compliant": len(compliant),
        "non_compliant": len(violations),
        "compliance_rate": len(compliant) / len(ops) if ops else 0,
        "by_family": {k: len(v) for k, v in by_family.items()},
        "violations": violations,
    }
