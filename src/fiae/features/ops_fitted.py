"""Fitted-state numeric operations (doc 05 entries 12-16, L1).

These operators learn parameters from the training fold and apply them at
inference.  Fit state must be split_id-keyed to prevent cross-fold leakage
(doc 13 FittedStateRef contract).

Operators:
- standardize: zero mean / unit variance
- robust_scale: median / IQR scaling
- minmax_scale: map to [0, 1]
- winsorize: clip extreme quantiles
- quantile_normal: map empirical distribution toward normal
"""

from __future__ import annotations

import math
from typing import Any, Optional

from ..contracts import FitScope, LeakageClass, TargetPermission
from .registry import FeatureOperator, register


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _f(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return None


def _sorted_valid(values: list) -> list[float]:
    return sorted(x for v in values if (x := _f(v)) is not None)


# ---------------------------------------------------------------------------
# standardize (entry 12)
# ---------------------------------------------------------------------------
def tf_standardize_fit(values: list, **_: Any) -> dict:
    """Learn mean and std from training fold."""
    valid = [x for v in values if (x := _f(v)) is not None]
    if len(valid) < 2:
        return {"mean": 0.0, "std": 1.0}
    mean = sum(valid) / len(valid)
    var = sum((x - mean) ** 2 for x in valid) / len(valid)
    std = math.sqrt(var) if var > 0 else 1.0
    return {"mean": mean, "std": std}


def tf_standardize_transform(values: list, state: dict, **_: Any) -> list:
    mean = state.get("mean", 0.0)
    std = state.get("std", 1.0)
    out: list[Optional[float]] = []
    for v in values:
        x = _f(v)
        if x is None:
            out.append(None)
        elif std == 0:
            out.append(0.0)
        else:
            out.append((x - mean) / std)
    return out


# ---------------------------------------------------------------------------
# robust_scale (entry 13)
# ---------------------------------------------------------------------------
def _median(data: list[float]) -> float:
    n = len(data)
    s = sorted(data)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _iqr(data: list[float]) -> float:
    s = sorted(data)
    n = len(s)
    q1_idx = n // 4
    q3_idx = (3 * n) // 4
    q1 = s[q1_idx]
    q3 = s[min(q3_idx, n - 1)]
    return q3 - q1


def tf_robust_scale_fit(values: list, **_: Any) -> dict:
    """Learn median and IQR from training fold."""
    valid = _sorted_valid(values)
    if len(valid) < 2:
        return {"median": 0.0, "iqr": 1.0}
    med = _median(valid)
    iqr_val = _iqr(valid)
    return {"median": med, "iqr": iqr_val if iqr_val > 0 else 1.0}


def tf_robust_scale_transform(values: list, state: dict, **_: Any) -> list:
    median = state.get("median", 0.0)
    iqr = state.get("iqr", 1.0)
    out: list[Optional[float]] = []
    for v in values:
        x = _f(v)
        if x is None:
            out.append(None)
        else:
            out.append((x - median) / iqr)
    return out


# ---------------------------------------------------------------------------
# minmax_scale (entry 14)
# ---------------------------------------------------------------------------
def tf_minmax_scale_fit(values: list, **_: Any) -> dict:
    """Learn min and max from training fold."""
    valid = _sorted_valid(values)
    if not valid:
        return {"min": 0.0, "max": 1.0}
    return {"min": valid[0], "max": valid[-1]}


def tf_minmax_scale_transform(values: list, state: dict, **_: Any) -> list:
    lo = state.get("min", 0.0)
    hi = state.get("max", 1.0)
    span = hi - lo
    out: list[Optional[float]] = []
    for v in values:
        x = _f(v)
        if x is None:
            out.append(None)
        elif span == 0:
            out.append(0.5)
        else:
            out.append((x - lo) / span)
    return out


# ---------------------------------------------------------------------------
# winsorize (entry 15)
# ---------------------------------------------------------------------------
def tf_winsorize_fit(
    values: list, lower_quantile: float = 0.01, upper_quantile: float = 0.99, **_: Any
) -> dict:
    """Learn clip thresholds from training fold quantiles."""
    valid = _sorted_valid(values)
    n = len(valid)
    if n == 0:
        return {"lower": 0.0, "upper": 0.0}
    lo_idx = max(0, int(lower_quantile * (n - 1)))
    hi_idx = min(n - 1, int(upper_quantile * (n - 1)))
    return {"lower": valid[lo_idx], "upper": valid[hi_idx]}


def tf_winsorize_transform(values: list, state: dict, **_: Any) -> list:
    lower = state.get("lower", 0.0)
    upper = state.get("upper", 0.0)
    out: list[Optional[float]] = []
    for v in values:
        x = _f(v)
        if x is None:
            out.append(None)
        else:
            out.append(max(lower, min(upper, x)))
    return out


# ---------------------------------------------------------------------------
# quantile_normal (entry 16)
# ---------------------------------------------------------------------------
def _ppf_normal(p: float) -> float:
    """Rational approximation to inverse normal CDF (Abramowitz & Stegun 26.2.23).

    Accurate to about 4.5e-4 which is sufficient for feature engineering.
    """
    if p <= 0:
        return -6.0
    if p >= 1:
        return 6.0
    if p > 0.5:
        return -_ppf_normal(1.0 - p)
    t = math.sqrt(-2.0 * math.log(p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return -(t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t))


def tf_quantile_normal_fit(values: list, **_: Any) -> dict:
    """Learn empirical quantile map from training fold."""
    valid = _sorted_valid(values)
    n = len(valid)
    if n < 3:
        return {"values": valid, "n": n}
    return {"values": valid, "n": n}


def tf_quantile_normal_transform(values: list, state: dict, **_: Any) -> list:
    ref_values = state.get("values", [])
    n = len(ref_values)
    out: list[Optional[float]] = []
    for v in values:
        x = _f(v)
        if x is None or n == 0:
            out.append(None)
        else:
            # Binary search for rank
            lo, hi = 0, n
            while lo < hi:
                mid = (lo + hi) // 2
                if ref_values[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid
            rank = lo  # 0-indexed rank
            p = (rank + 0.5) / n
            out.append(_ppf_normal(p))
    return out


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------
def _fitted_op(
    name: str, purpose: str, fit_fn, transform_fn, trigger: str,
    rejects: tuple, tests: tuple, cost_shape: str = "O(n)",
) -> None:
    register(
        FeatureOperator(
            name=name,
            family="numeric",
            arity="unary",
            input_types=("continuous_numeric",),
            output_type="numeric",
            purpose=purpose,
            preconditions=("non-constant",),
            fit_scope=FitScope.TRAINING_FOLD,
            null_policy="preserve",
            leakage_class=LeakageClass.L1,
            target_permission=TargetPermission.P0_NONE,
            cost_shape=cost_shape,
            generation_trigger=trigger,
            rejection_conditions=rejects,
            validation="inside-fold",
            inference_requirement="stored fitted state",
            transform=transform_fn,
            mandatory_tests=tests,
        )
    )


_fitted_op(
    "standardize",
    "zero mean / unit variance for scale-sensitive models",
    tf_standardize_fit, tf_standardize_transform,
    "linear/SVM/KNN/PCA/neural models",
    ("constant", "tiny sample"),
    ("train-only fit", "serialization"),
)

_fitted_op(
    "robust_scale",
    "scale by median / IQR to resist outliers",
    tf_robust_scale_fit, tf_robust_scale_transform,
    "outlier-heavy scale-sensitive path",
    ("degenerate", "tiny sample"),
    ("fold isolation",),
)

_fitted_op(
    "minmax_scale",
    "map training range to [0, 1]",
    tf_minmax_scale_fit, tf_minmax_scale_transform,
    "bounded-input algorithms",
    ("extreme outlier", "no need"),
    ("unknown-range behavior",),
)

_fitted_op(
    "winsorize",
    "clip extreme quantile tails",
    tf_winsorize_fit, tf_winsorize_transform,
    "heavy tails / outlier sensitivity",
    ("extremes meaningful", "tiny data"),
    ("fold-only quantiles",),
)

_fitted_op(
    "quantile_normal",
    "map empirical distribution toward normal",
    tf_quantile_normal_fit, tf_quantile_normal_transform,
    "non-Gaussian + compatible model",
    ("small n", "high cost"),
    ("monotonic mapping",),
    cost_shape="O(n log n)",
)
