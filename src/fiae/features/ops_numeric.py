"""Numeric feature operations (doc 05 catalog slice 1-32, L0 row-wise).

Exact semantics per entry: null preserved, invalid domain -> explicit missing
(never a misleading numeric value, doc 01 invariant 10), overflow guarded.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from ..contracts import FitScope, LeakageClass, TargetPermission
from .registry import FeatureOperator, register

# Overflow guard thresholds: x*x overflows float64 above ~1.3e154.
_SQ_OVERFLOW = 1e154
_CUBE_OVERFLOW = 1e102


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _f(v: Any) -> Optional[float]:
    """Coerce to float, treating non-numeric junk as missing."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if _num(v):
        return float(v)
    return None


# --------------------------------------------------------------------------
# unary transforms (values: list, **params) -> list
# --------------------------------------------------------------------------
def tf_identity(values: list, **_: Any) -> list:
    return list(values)


def tf_log1p(values: list, **_: Any) -> list:
    out: list[Optional[float]] = []
    for v in values:
        x = _f(v)
        # x >= -1 domain; below is invalid -> explicit missing
        out.append(math.log1p(x) if x is not None and x >= -1.0 else None)
    return out


def tf_signed_log1p(values: list, **_: Any) -> list:
    out: list[Optional[float]] = []
    for v in values:
        x = _f(v)
        out.append(
            math.copysign(math.log1p(abs(x)), x) if x is not None else None
        )
    return out


def tf_sqrt(values: list, **_: Any) -> list:
    out: list[Optional[float]] = []
    for v in values:
        x = _f(v)
        out.append(math.sqrt(x) if x is not None and x >= 0.0 else None)
    return out


def tf_cbrt(values: list, **_: Any) -> list:
    out: list[Optional[float]] = []
    for v in values:
        x = _f(v)
        out.append(math.copysign(abs(x) ** (1.0 / 3.0), x) if x is not None else None)
    return out


def tf_square(values: list, **_: Any) -> list:
    out: list[Optional[float]] = []
    for v in values:
        x = _f(v)
        out.append(x * x if x is not None and abs(x) <= _SQ_OVERFLOW else None)
    return out


def tf_cube(values: list, **_: Any) -> list:
    out: list[Optional[float]] = []
    for v in values:
        x = _f(v)
        out.append(x * x * x if x is not None and abs(x) <= _CUBE_OVERFLOW else None)
    return out


def tf_abs(values: list, **_: Any) -> list:
    out: list[Optional[float]] = []
    for v in values:
        x = _f(v)
        out.append(abs(x) if x is not None else None)
    return out


def tf_sign(values: list, **_: Any) -> list:
    out: list[Optional[int]] = []
    for v in values:
        x = _f(v)
        out.append(None if x is None else (-1 if x < 0 else (0 if x == 0 else 1)))
    return out


def tf_reciprocal(values: list, eps: float = 1e-12, **_: Any) -> list:
    out: list[Optional[float]] = []
    for v in values:
        x = _f(v)
        out.append(1.0 / x if x is not None and abs(x) > eps else None)
    return out


def tf_exp_clip(values: list, lo: float = -20.0, hi: float = 20.0, **_: Any) -> list:
    out: list[Optional[float]] = []
    for v in values:
        x = _f(v)
        out.append(math.exp(min(hi, max(lo, x))) if x is not None else None)
    return out


# --------------------------------------------------------------------------
# indicators
# --------------------------------------------------------------------------
def tf_zero_indicator(values: list, **_: Any) -> list:
    out: list[Optional[bool]] = []
    for v in values:
        x = _f(v)
        out.append(None if x is None else x == 0.0)
    return out


def tf_positive_indicator(values: list, **_: Any) -> list:
    out: list[Optional[bool]] = []
    for v in values:
        x = _f(v)
        out.append(None if x is None else x > 0.0)
    return out


def tf_missing_indicator(values: list, **_: Any) -> list:
    # always defined, missing-token consistent (doc 05 entry 19)
    return [v is None or (isinstance(v, float) and math.isnan(v)) for v in values]


def tf_finite_indicator(values: list, **_: Any) -> list:
    # always defined; NaN/inf/not-a-number -> False
    out: list[bool] = []
    for v in values:
        x = _f(v)
        out.append(x is not None and math.isfinite(x))
    return out


# --------------------------------------------------------------------------
# pairwise transforms (a: list, b: list, **params) -> list
# --------------------------------------------------------------------------
def tf_sum(a: list, b: list, **_: Any) -> list:
    out: list[Optional[float]] = []
    for va, vb in zip(a, b):
        x, y = _f(va), _f(vb)
        out.append(x + y if x is not None and y is not None else None)
    return out


def tf_difference(a: list, b: list, **_: Any) -> list:
    out: list[Optional[float]] = []
    for va, vb in zip(a, b):
        x, y = _f(va), _f(vb)
        out.append(x - y if x is not None and y is not None else None)
    return out


def tf_product(a: list, b: list, **_: Any) -> list:
    out: list[Optional[float]] = []
    for va, vb in zip(a, b):
        x, y = _f(va), _f(vb)
        if x is None or y is None:
            out.append(None)
        elif x != 0 and y != 0 and abs(x * y) > _SQ_OVERFLOW:
            out.append(None)  # overflow guard
        else:
            out.append(x * y)
    return out


def tf_safe_ratio(a: list, b: list, eps: float = 1e-12, **_: Any) -> list:
    out: list[Optional[float]] = []
    for va, vb in zip(a, b):
        x, y = _f(va), _f(vb)
        if x is None or y is None:
            out.append(None)
        elif abs(y) <= eps:
            out.append(None)  # zero/near-zero denominator -> explicit missing
        else:
            out.append(x / y)
    return out


def tf_relative_difference(a: list, b: list, eps: float = 1e-12, **_: Any) -> list:
    out: list[Optional[float]] = []
    for va, vb in zip(a, b):
        x, y = _f(va), _f(vb)
        out.append(
            (x - y) / (abs(y) + eps) if x is not None and y is not None else None
        )
    return out


def tf_min_pair(a: list, b: list, **_: Any) -> list:
    out: list[Optional[float]] = []
    for va, vb in zip(a, b):
        x, y = _f(va), _f(vb)
        out.append(min(x, y) if x is not None and y is not None else None)
    return out


def tf_max_pair(a: list, b: list, **_: Any) -> list:
    out: list[Optional[float]] = []
    for va, vb in zip(a, b):
        x, y = _f(va), _f(vb)
        out.append(max(x, y) if x is not None and y is not None else None)
    return out


def tf_mean_pair(a: list, b: list, **_: Any) -> list:
    out: list[Optional[float]] = []
    for va, vb in zip(a, b):
        x, y = _f(va), _f(vb)
        out.append((x + y) / 2.0 if x is not None and y is not None else None)
    return out


def tf_harmonic_mean_pair(a: list, b: list, **_: Any) -> list:
    out: list[Optional[float]] = []
    for va, vb in zip(a, b):
        x, y = _f(va), _f(vb)
        if x is None or y is None or x <= 0 or y <= 0 or (x + y) == 0:
            out.append(None)  # domain violation -> explicit missing
        else:
            out.append(2.0 * x * y / (x + y))
    return out


def tf_geometric_mean_pair(a: list, b: list, **_: Any) -> list:
    out: list[Optional[float]] = []
    for va, vb in zip(a, b):
        x, y = _f(va), _f(vb)
        if x is None or y is None or x < 0 or y < 0:
            out.append(None)
        else:
            out.append(math.sqrt(x * y))
    return out


def tf_euclidean_norm_pair(a: list, b: list, **_: Any) -> list:
    out: list[Optional[float]] = []
    for va, vb in zip(a, b):
        x, y = _f(va), _f(vb)
        out.append(math.hypot(x, y) if x is not None and y is not None else None)
    return out


def tf_absolute_difference(a: list, b: list, **_: Any) -> list:
    out: list[Optional[float]] = []
    for va, vb in zip(a, b):
        x, y = _f(va), _f(vb)
        out.append(abs(x - y) if x is not None and y is not None else None)
    return out


# --------------------------------------------------------------------------
# static domain checks: (values..., **params) -> reason | None
# --------------------------------------------------------------------------
def dc_log1p(values: list, max_invalid_rate: float = 0.5, **_: Any) -> Optional[str]:
    total = sum(1 for v in values if _f(v) is not None)
    invalid = sum(1 for v in values if (x := _f(v)) is not None and x < -1.0)
    if total and invalid / total > max_invalid_rate:
        return f"log1p domain violated for {invalid}/{total} rows (x < -1)"
    return None


def dc_sqrt(values: list, max_negative_rate: float = 0.05, **_: Any) -> Optional[str]:
    total = sum(1 for v in values if _f(v) is not None)
    neg = sum(1 for v in values if (x := _f(v)) is not None and x < 0)
    if total and neg / total > max_negative_rate:
        return f"sqrt domain violated for {neg}/{total} rows (x < 0)"
    return None


def dc_safe_ratio(
    a: list, b: list, eps: float = 1e-12, max_zero_rate: float = 0.3, **_: Any
) -> Optional[str]:
    total = sum(1 for va, vb in zip(a, b) if _f(va) is not None and _f(vb) is not None)
    zero = sum(1 for va, vb in zip(a, b) if (y := _f(vb)) is not None and _f(va) is not None and abs(y) <= eps)
    if total and zero / total > max_zero_rate:
        return f"safe_ratio zero denominators {zero}/{total} exceed tolerance"
    return None


def dc_harmonic(a: list, b: list, **_: Any) -> Optional[str]:
    total = sum(1 for va, vb in zip(a, b) if _f(va) is not None and _f(vb) is not None)
    bad = sum(1 for va, vb in zip(a, b) if (x := _f(va)) is not None and (y := _f(vb)) is not None and (x <= 0 or y <= 0))
    if total and bad / total > 0.3:
        return f"harmonic_mean_pair non-positive values {bad}/{total}"
    return None


# --------------------------------------------------------------------------
# operator registrations (doc 05 catalog slice 1-32)
# --------------------------------------------------------------------------
def _unary(
    name: str,
    purpose: str,
    tf,
    trigger: str,
    rejects: tuple,
    tests: tuple,
    *,
    domain=None,
    output_type: str = "numeric",
    null_policy: str = "preserve",
    input_types: tuple = ("continuous_numeric",),
) -> None:
    register(
        FeatureOperator(
            name=name,
            family="numeric",
            arity="unary",
            input_types=input_types,
            output_type=output_type,
            purpose=purpose,
            preconditions=("finite/parseable numeric or configured missing",),
            fit_scope=FitScope.NONE,
            null_policy=null_policy,
            leakage_class=LeakageClass.L0,
            target_permission=TargetPermission.P0_NONE,
            cost_shape="O(n)",
            generation_trigger=trigger,
            rejection_conditions=rejects,
            validation="incremental CV",
            inference_requirement="input available",
            transform=tf,
            mandatory_tests=tests,
            domain_check=domain,
        )
    )


def _binary(
    name: str,
    purpose: str,
    tf,
    trigger: str,
    rejects: tuple,
    tests: tuple,
    *,
    domain=None,
    commutative: bool = False,
) -> None:
    register(
        FeatureOperator(
            name=name,
            family="numeric interaction",
            arity="binary",
            input_types=("continuous_numeric", "continuous_numeric"),
            output_type="numeric",
            purpose=purpose,
            preconditions=("compatible units/evidence",),
            fit_scope=FitScope.NONE,
            null_policy="propagate",
            leakage_class=LeakageClass.L0,
            target_permission=TargetPermission.P0_NONE,
            cost_shape="O(n)",
            generation_trigger=trigger,
            rejection_conditions=rejects,
            validation="incremental CV",
            inference_requirement="both available",
            transform=tf,
            mandatory_tests=tests,
            commutative=commutative,
            domain_check=domain,
        )
    )




# Unary operators (doc 05 entries 1-19)
_unary("identity", "retain raw baseline signal", tf_identity,
       "all allowed numeric raw features",
       ("constant", "leakage-rejected", "identifier semantics"),
       ("roundtrip", "null preservation"))
_unary("log1p", "compress right skew and multiplicative scale", tf_log1p,
       "non-negative skew/heavy tail",
       ("domain violation", "no useful variation"),
       ("domain", "monotonicity", "finite outputs"), domain=dc_log1p)
_unary("signed_log1p", "compress positive and negative heavy tails",
       tf_signed_log1p, "mixed-sign heavy tail",
       ("no gain", "constant"),
       ("sign preservation", "inverse-order around zero"))
_unary("sqrt", "stabilize count-like/non-negative skew", tf_sqrt,
       "counts/non-negative values",
       ("negative rate > tolerance",),
       ("domain", "sqrt(0)"), domain=dc_sqrt)
_unary("cbrt", "mild signed tail compression", tf_cbrt,
       "signed skew", ("no gain",),
       ("negative values supported",))
_unary("square", "capture symmetric quadratic relationship", tf_square,
       "linear model residual curvature",
       ("overflow", "extreme scale", "no gain"),
       ("overflow guard",))
_unary("cube", "capture odd nonlinear relationship", tf_cube,
       "strong residual evidence",
       ("depth/cost", "no gain"),
       ("overflow guard",))
_unary("abs", "capture magnitude regardless of sign", tf_abs,
       "signed deviations",
       ("sign carries most signal", "no gain"),
       ("idempotence",))
_unary("sign", "separate negative/zero/positive regimes", tf_sign,
       "mixed-sign sparse features",
       ("single sign", "no variance"),
       ("three-way mapping",), output_type="categorical")
_unary("reciprocal", "model inverse rate/time relationships", tf_reciprocal,
       "domain suggests inverse",
       ("too many near-zero values",),
       ("zero policy",))
_unary("exp_clip", "capture exponential response under bounded input", tf_exp_clip,
       "strong domain/prior evidence only",
       ("wide unbounded range", "no evidence"),
       ("finite output",))
_unary("zero_indicator", "capture structural zeros", tf_zero_indicator,
       "excess zeros",
       ("zero rare", "no variance"),
       ("zero/missing distinction",), output_type="boolean")
_unary("positive_indicator", "capture sign regime", tf_positive_indicator,
       "threshold/regime behavior",
       ("no variation",),
       ("boundary",), output_type="boolean")
_unary("missing_indicator", "capture informative missingness", tf_missing_indicator,
       "any missing feature",
       ("no missing values",),
       ("missing-token consistency",), output_type="boolean",
       input_types=("any",), null_policy="always_defined")
_unary("finite_indicator", "detect invalid numeric values", tf_finite_indicator,
       "messy numerical sources",
       ("all finite", "no variance"),
       ("NaN/inf handling",), output_type="boolean", null_policy="always_defined")

# Binary operators (doc 05 entries 20-32)
_binary("sum", "add compatible quantities", tf_sum,
        "semantic pair/prior",
        ("unit mismatch", "arbitrary combinatorial pair"),
        ("commutative canonicalization",), commutative=True)
_binary("difference", "measure gap/net/change", tf_difference,
        "actual-plan, price-cost, end-start",
        ("no meaningful direction",),
        ("non-commutative signature",))
_binary("product", "capture multiplicative interaction", tf_product,
        "residual/domain interaction",
        ("overflow", "combinatorial explosion"),
        ("overflow",), commutative=True)
_binary("safe_ratio", "normalize numerator by denominator", tf_safe_ratio,
        "amount/count, distance/time",
        ("too many zero denominators",),
        ("division zero", "sign"), domain=dc_safe_ratio)
_binary("relative_difference", "scale difference by reference magnitude",
        tf_relative_difference, "relative performance",
        ("near-zero reference dominates",), ("zero policy",))
_binary("min_pair", "lower envelope/bottleneck", tf_min_pair,
        "capacity/bottleneck prior",
        ("high redundancy", "no evidence"),
        ("commutative",), commutative=True)
_binary("max_pair", "upper envelope/capacity", tf_max_pair,
        "maximum regime",
        ("redundant", "no evidence"),
        ("commutative",), commutative=True)
_binary("mean_pair", "combine repeated measurements", tf_mean_pair,
        "redundant measurements",
        ("unit mismatch",),
        ("symmetry",), commutative=True)
_binary("harmonic_mean_pair", "emphasize lower of two rates",
        tf_harmonic_mean_pair, "rate/bottleneck semantics",
        ("domain violation",),
        ("zero denominator",), domain=dc_harmonic, commutative=True)
_binary("geometric_mean_pair", "multiplicative central tendency",
        tf_geometric_mean_pair, "scale ratios/multiplicative domains",
        ("domain", "no evidence"),
        ("overflow-safe",), commutative=True)
_binary("euclidean_norm_pair", "2D magnitude", tf_euclidean_norm_pair,
        "coordinates/deviation",
        ("scale mismatch",),
        ("stable hypot",), commutative=True)
_binary("absolute_difference", "distance between values", tf_absolute_difference,
        "agreement/deviation",
        ("direction important", "no evidence"),
        ("symmetry",), commutative=True)
