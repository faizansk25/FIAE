"""Group aggregate feature operations (doc 05 entries 73-80).

These operators learn per-group statistics from training data and apply them
at inference.  Each group key maps to a single aggregate value which is then
broadcast back to every row belonging to that group.

Fit state must be split_id-keyed to prevent cross-fold leakage (doc 13).

Operators:
- group_count:    support count per group key
- group_mean:     mean value per group key
- group_std:      standard deviation per group key
- group_min:      minimum value per group key
- group_max:      maximum value per group key
- group_median:   median value per group key
- group_nunique:  distinct value count per group key
- group_missing_rate: missing fraction per group key
"""

from __future__ import annotations

import math
from collections import defaultdict
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


def _s(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    return s if s else None


# ---------------------------------------------------------------------------
# Fit functions: learn per-group statistics from training fold
# ---------------------------------------------------------------------------
def _group_fit(
    values: list, keys: list, agg: str, default: float = 0.0
) -> dict:
    """Generic group aggregate fit.

    agg: one of 'count', 'mean', 'std', 'min', 'max', 'median', 'nunique',
         'missing_rate'
    """
    # For nunique and missing_rate, preserve raw values; otherwise coerce
    coerce = agg not in ("nunique", "missing_rate")
    groups: dict[str, list] = defaultdict(list)
    for k, v in zip(keys, values):
        ks = _s(k)
        if ks is not None:
            groups[ks].append(_f(v) if coerce else v)

    result: dict[str, Any] = {"default": default, "agg": agg}

    if agg == "count":
        result["map"] = {k: len(vals) for k, vals in groups.items()}
    elif agg == "mean":
        m = {}
        for k, vals in groups.items():
            valid = [x for x in vals if x is not None]
            m[k] = sum(valid) / len(valid) if valid else default
        result["map"] = m
    elif agg == "std":
        m = {}
        for k, vals in groups.items():
            valid = [x for x in vals if x is not None]
            if len(valid) < 2:
                m[k] = 0.0
            else:
                mean = sum(valid) / len(valid)
                var = sum((x - mean) ** 2 for x in valid) / len(valid)
                m[k] = math.sqrt(var)
        result["map"] = m
    elif agg == "min":
        m = {}
        for k, vals in groups.items():
            valid = [x for x in vals if x is not None]
            m[k] = min(valid) if valid else default
        result["map"] = m
    elif agg == "max":
        m = {}
        for k, vals in groups.items():
            valid = [x for x in vals if x is not None]
            m[k] = max(valid) if valid else default
        result["map"] = m
    elif agg == "median":
        m = {}
        for k, vals in groups.items():
            valid = sorted(x for x in vals if x is not None)
            if not valid:
                m[k] = default
            else:
                n = len(valid)
                if n % 2 == 1:
                    m[k] = valid[n // 2]
                else:
                    m[k] = (valid[n // 2 - 1] + valid[n // 2]) / 2.0
        result["map"] = m
    elif agg == "nunique":
        m = {}
        for k, vals in groups.items():
            # Raw values: None is missing, str/num is distinct
            unique = set()
            for v in vals:
                if v is None:
                    continue
                if isinstance(v, float) and math.isnan(v):
                    continue
                unique.add(str(v))
            m[k] = len(unique)
        result["map"] = m
    elif agg == "missing_rate":
        m = {}
        for k, vals in groups.items():
            total = len(vals)
            missing = 0
            for v in vals:
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    missing += 1
            m[k] = missing / total if total > 0 else default
        result["map"] = m

    return result


# ---------------------------------------------------------------------------
# Transform functions: map group keys to learned aggregates
# ---------------------------------------------------------------------------
def _group_transform(keys: list, state: dict) -> list:
    agg_map = state.get("map", {})
    default = state.get("default", 0.0)
    out: list[Optional[float]] = []
    for k in keys:
        ks = _s(k)
        if ks is None:
            out.append(None)
        else:
            out.append(agg_map.get(ks, default))
    return out


# ---------------------------------------------------------------------------
# Public fit/transform pairs
# ---------------------------------------------------------------------------
def tf_group_count_fit(keys: list, **_: Any) -> dict:
    groups: dict[str, int] = {}
    for k in keys:
        ks = _s(k)
        if ks is not None:
            groups[ks] = groups.get(ks, 0) + 1
    return {"map": groups, "default": 0, "agg": "count"}


def tf_group_count_transform(keys: list, state: dict, **_: Any) -> list:
    return _group_transform(keys, state)


def tf_group_mean_fit(values: list, keys: list, **_: Any) -> dict:
    return _group_fit(values, keys, "mean")


def tf_group_mean_transform(keys: list, state: dict, **_: Any) -> list:
    return _group_transform(keys, state)


def tf_group_std_fit(values: list, keys: list, **_: Any) -> dict:
    return _group_fit(values, keys, "std")


def tf_group_std_transform(keys: list, state: dict, **_: Any) -> list:
    return _group_transform(keys, state)


def tf_group_min_fit(values: list, keys: list, **_: Any) -> dict:
    return _group_fit(values, keys, "min")


def tf_group_min_transform(keys: list, state: dict, **_: Any) -> list:
    return _group_transform(keys, state)


def tf_group_max_fit(values: list, keys: list, **_: Any) -> dict:
    return _group_fit(values, keys, "max")


def tf_group_max_transform(keys: list, state: dict, **_: Any) -> list:
    return _group_transform(keys, state)


def tf_group_median_fit(values: list, keys: list, **_: Any) -> dict:
    return _group_fit(values, keys, "median")


def tf_group_median_transform(keys: list, state: dict, **_: Any) -> list:
    return _group_transform(keys, state)


def tf_group_nunique_fit(values: list, keys: list, **_: Any) -> dict:
    return _group_fit(values, keys, "nunique")


def tf_group_nunique_transform(keys: list, state: dict, **_: Any) -> list:
    return _group_transform(keys, state)


def tf_group_missing_rate_fit(values: list, keys: list, **_: Any) -> dict:
    return _group_fit(values, keys, "missing_rate")


def tf_group_missing_rate_transform(keys: list, state: dict, **_: Any) -> list:
    return _group_transform(keys, state)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
_GROUP_OPS = [
    ("group_count", "group support count", "count", "repeated entity/group"),
    ("group_mean", "group mean value", "mean", "repeated entity/group"),
    ("group_std", "group standard deviation", "std", "repeated entity/group"),
    ("group_min", "group minimum", "min", "repeated entity/group"),
    ("group_max", "group maximum", "max", "repeated entity/group"),
    ("group_median", "group robust center (median)", "median", "repeated entity/group"),
    ("group_nunique", "group distinct value count", "nunique", "repeated entity/group"),
    ("group_missing_rate", "group missing data fraction", "missing_rate", "repeated entity/group"),
]

for _name, _purpose, _agg, _trigger in _GROUP_OPS:
    _fit_fn = globals()[f"tf_{_name}_fit"]
    _tf_fn = globals()[f"tf_{_name}_transform"]
    register(
        FeatureOperator(
            name=_name,
            family="group aggregate",
            arity="binary",
            input_types=("entity_key", "continuous_numeric"),
            output_type="numeric",
            purpose=_purpose,
            preconditions=("repeated meaningful groups",),
            fit_scope=FitScope.TRAINING_FOLD,
            null_policy="unknown group -> 0 / global prior",
            leakage_class=LeakageClass.L1,
            target_permission=TargetPermission.P0_NONE,
            cost_shape="O(n) + group state",
            generation_trigger=_trigger,
            rejection_conditions=("mostly unique group", "memory/leakage risk"),
            validation="inside-fold",
            inference_requirement="stored aggregate map",
            transform=_tf_fn,
            mandatory_tests=("unknown group handling", "fold isolation"),
        )
    )
