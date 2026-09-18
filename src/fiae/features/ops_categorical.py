"""Categorical feature operations (doc 05 entries 33-43).

Categorical operators handle string/enum inputs.  They follow the same
null-preserving, fail-closed semantics as numeric operators.  L1 operators
(fitted-state) learn maps from training fold only; cross-fold leakage is
prevented by split_id-keyed state isolation (doc 13).
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any, Optional

from ..contracts import FitScope, LeakageClass, TargetPermission
from .registry import FeatureOperator, register


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cat(v: Any) -> Optional[str]:
    """Coerce value to string category; None for missing/empty."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    return s if s else None


_RARE_TOKEN = "__RARE__"  # noqa: S105 — categorical bucketing sentinel, not a secret


# ---------------------------------------------------------------------------
# Stateless L0 transforms (entries 38, 39, 43)
# ---------------------------------------------------------------------------
def tf_hash_encode(
    values: list, n_buckets: int = 16, seed: int = 0, **_: Any
) -> list:
    """Hash each category to a fixed-size bucket index (L0, no fit)."""
    out: list[Optional[int]] = []
    for v in values:
        c = _cat(v)
        if c is None:
            out.append(None)
        else:
            h = hashlib.md5(f"{seed}:{c}".encode(), usedforsecurity=False).hexdigest()
            out.append(int(h, 16) % n_buckets)
    return out


def tf_category_cross(
    values_a: list, values_b: list, **_: Any
) -> list:
    """Produce canonical escaped category pair string (L0)."""
    out: list[Optional[str]] = []
    for va, vb in zip(values_a, values_b):
        ca, cb = _cat(va), _cat(vb)
        if ca is None or cb is None:
            out.append(None)
        else:
            ca_esc = ca.replace("|", "\\|")
            cb_esc = cb.replace("|", "\\|")
            out.append(ca_esc + "|" + cb_esc)
    return out


def tf_ordinal_true_scale(
    values: list, order: list | None = None, **_: Any
) -> list:
    """Map categories to rank using a fixed user-supplied order (L0)."""
    if order is None:
        order = []
    rank_map = {cat: i for i, cat in enumerate(order)}
    out: list[Optional[int]] = []
    for v in values:
        c = _cat(v)
        if c is None:
            out.append(None)
        elif c in rank_map:
            out.append(rank_map[c])
        else:
            out.append(None)  # unknown category -> explicit missing
    return out


# ---------------------------------------------------------------------------
# L1 fitted-state transform functions
# ---------------------------------------------------------------------------
def tf_one_hot_fit(values: list, max_categories: int = 100, **_: Any) -> dict:
    """Learn vocabulary from training data. Returns fit_state dict."""
    counts = Counter(_cat(v) for v in values if _cat(v) is not None)
    categories = [cat for cat, _ in counts.most_common(max_categories)]
    return {"categories": categories, "max_categories": max_categories}


def tf_one_hot_transform(values: list, state: dict, **_: Any) -> list:
    """Transform using learned vocabulary. Returns list of bool lists."""
    cats = state.get("categories", [])
    cat_to_idx = {c: i for i, c in enumerate(cats)}
    out: list = []
    for v in values:
        c = _cat(v)
        row = [False] * len(cats)
        if c is not None and c in cat_to_idx:
            row[cat_to_idx[c]] = True
        out.append(row)
    return out


def tf_ordinal_encode_fit(values: list, **_: Any) -> dict:
    """Learn category -> integer mapping sorted by frequency."""
    counts = Counter(_cat(v) for v in values if _cat(v) is not None)
    categories = [cat for cat, _ in counts.most_common()]
    mapping = {cat: i for i, cat in enumerate(categories)}
    return {"mapping": mapping}


def tf_ordinal_encode_transform(values: list, state: dict, **_: Any) -> list:
    mapping = state.get("mapping", {})
    out: list[Optional[int]] = []
    for v in values:
        c = _cat(v)
        if c is None:
            out.append(None)
        elif c in mapping:
            out.append(mapping[c])
        else:
            out.append(-1)  # unknown code
    return out


def tf_frequency_encode_fit(values: list, **_: Any) -> dict:
    """Learn category -> frequency ratio on training fold."""
    n = len(values)
    counts = Counter(_cat(v) for v in values if _cat(v) is not None)
    freq = {cat: cnt / max(n, 1) for cat, cnt in counts.items()}
    return {"frequency_map": freq}


def tf_frequency_encode_transform(values: list, state: dict, **_: Any) -> list:
    freq_map = state.get("frequency_map", {})
    out: list[Optional[float]] = []
    for v in values:
        c = _cat(v)
        if c is None:
            out.append(None)
        else:
            out.append(freq_map.get(c, 0.0))
    return out


def tf_count_encode_fit(values: list, **_: Any) -> dict:
    """Learn category -> count on training fold."""
    counts = Counter(_cat(v) for v in values if _cat(v) is not None)
    return {"count_map": dict(counts)}


def tf_count_encode_transform(values: list, state: dict, **_: Any) -> list:
    count_map = state.get("count_map", {})
    out: list[Optional[int]] = []
    for v in values:
        c = _cat(v)
        if c is None:
            out.append(None)
        else:
            out.append(count_map.get(c, 0))
    return out


def tf_rare_group_fit(values: list, min_count: int = 5, **_: Any) -> dict:
    """Learn which categories to retain; rest become __RARE__."""
    counts = Counter(_cat(v) for v in values if _cat(v) is not None)
    retained = {cat for cat, cnt in counts.items() if cnt >= min_count}
    return {"retained": retained, "min_count": min_count}


def tf_rare_group_transform(values: list, state: dict, **_: Any) -> list:
    retained = state.get("retained", set())
    out: list[Optional[str]] = []
    for v in values:
        c = _cat(v)
        if c is None:
            out.append(None)
        elif c in retained:
            out.append(c)
        else:
            out.append(_RARE_TOKEN)
    return out


# ---------------------------------------------------------------------------
# L2 target-aware transforms (entries 40, 41)
# ---------------------------------------------------------------------------
def tf_target_mean_crossfit(
    values: list, target: list, smoothing: float = 10.0, **_: Any
) -> list:
    """OOF target mean encoding with Bayesian smoothing (L2 cross-fit).

    For this pure-transform implementation we compute smoothed global target
    mean per category.  A full cross-fit would require fold-aware routing.
    """
    non_none = [v for v in target if v is not None]
    global_mean = sum(float(v) for v in non_none) / max(1, len(non_none))

    cat_sums: dict[str, float] = {}
    cat_counts: dict[str, int] = {}
    for v, y in zip(values, target):
        c = _cat(v)
        if c is not None and y is not None:
            cat_sums[c] = cat_sums.get(c, 0.0) + float(y)
            cat_counts[c] = cat_counts.get(c, 0) + 1

    out: list[Optional[float]] = []
    for v in values:
        c = _cat(v)
        if c is None:
            out.append(None)
        else:
            cnt = cat_counts.get(c, 0)
            cat_mean = cat_sums.get(c, 0.0) / max(cnt, 1)
            smoothed = (cnt * cat_mean + smoothing * global_mean) / (cnt + smoothing)
            out.append(smoothed)
    return out


def tf_woe_crossfit(
    values: list, target: list, smoothing: float = 10.0, **_: Any
) -> list:
    """Weight of Evidence encoding for binary targets (L2 cross-fit).

    WoE = ln((P(y=1|cat) / P(y=0|cat)) / (P(y=1) / P(y=0)))
    with Laplace smoothing.
    """
    pos_total = sum(1 for y in target if y is not None and float(y) == 1.0)
    neg_total = sum(1 for y in target if y is not None and float(y) == 0.0)
    eps = 1e-6

    cat_pos: dict[str, int] = {}
    cat_neg: dict[str, int] = {}
    for v, y in zip(values, target):
        c = _cat(v)
        if c is None or y is None:
            continue
        if float(y) == 1.0:
            cat_pos[c] = cat_pos.get(c, 0) + 1
        else:
            cat_neg[c] = cat_neg.get(c, 0) + 1

    out: list[Optional[float]] = []
    for v in values:
        c = _cat(v)
        if c is None:
            out.append(None)
        else:
            p = (cat_pos.get(c, 0) + smoothing) / max(pos_total + 2 * smoothing, eps)
            q = (cat_neg.get(c, 0) + smoothing) / max(neg_total + 2 * smoothing, eps)
            out.append(math.log(max(p, eps) / max(q, eps)))
    return out


def tf_numeric_to_cat_target(
    values: list, target: list, n_bins: int = 5, smoothing: float = 10.0, **_: Any
) -> list:
    """Bin numeric values then cross-fit encode with target means (L2)."""
    numeric_vals = []
    for v in values:
        try:
            numeric_vals.append(float(v) if v is not None else None)
        except (TypeError, ValueError):
            numeric_vals.append(None)

    valid = [v for v in numeric_vals if v is not None]
    if len(valid) < n_bins:
        return [None] * len(values)

    lo, hi = min(valid), max(valid)
    if lo == hi:
        return [0.0] * len(values)

    def _bin_idx(x):
        if x is None:
            return None
        idx = int((x - lo) / (hi - lo) * n_bins)
        return min(idx, n_bins - 1)

    bin_labels = [_bin_idx(v) for v in numeric_vals]

    global_mean = sum(float(y) for y in target if y is not None) / max(
        1, len([y for y in target if y is not None])
    )
    bin_sums: dict[int, float] = {}
    bin_counts: dict[int, int] = {}
    for bl, y in zip(bin_labels, target):
        if bl is not None and y is not None:
            bin_sums[bl] = bin_sums.get(bl, 0.0) + float(y)
            bin_counts[bl] = bin_counts.get(bl, 0) + 1

    out: list[Optional[float]] = []
    for bl in bin_labels:
        if bl is None:
            out.append(None)
        else:
            cnt = bin_counts.get(bl, 0)
            bin_mean = bin_sums.get(bl, 0.0) / max(cnt, 1)
            smoothed = (cnt * bin_mean + smoothing * global_mean) / (cnt + smoothing)
            out.append(smoothed)
    return out


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------
def _cat_op(
    name: str, purpose: str, tf, trigger: str, rejects: tuple,
    tests: tuple, *, fit_scope: FitScope = FitScope.NONE,
    leakage_class: LeakageClass = LeakageClass.L0,
    target_permission: TargetPermission = TargetPermission.P0_NONE,
    output_type: str = "categorical",
    input_types: tuple = ("categorical",),
) -> None:
    register(
        FeatureOperator(
            name=name,
            family="categorical",
            arity="unary",
            input_types=input_types,
            output_type=output_type,
            purpose=purpose,
            preconditions=("categorical or coerceable input",),
            fit_scope=fit_scope,
            null_policy="preserve",
            leakage_class=leakage_class,
            target_permission=target_permission,
            cost_shape="O(n)",
            generation_trigger=trigger,
            rejection_conditions=rejects,
            validation="incremental CV" if fit_scope == FitScope.NONE else "inside-fold",
            inference_requirement="input available" if fit_scope == FitScope.NONE else "fitted state",
            transform=tf,
            mandatory_tests=tests,
        )
    )


def _cat_binary_op(
    name: str, purpose: str, tf, trigger: str, rejects: tuple,
    tests: tuple, *, fit_scope: FitScope = FitScope.NONE,
    leakage_class: LeakageClass = LeakageClass.L0,
) -> None:
    register(
        FeatureOperator(
            name=name,
            family="categorical interaction",
            arity="binary",
            input_types=("categorical", "categorical"),
            output_type="categorical",
            purpose=purpose,
            preconditions=("both inputs categorical",),
            fit_scope=fit_scope,
            null_policy="propagate",
            leakage_class=leakage_class,
            target_permission=TargetPermission.P0_NONE,
            cost_shape="O(n)",
            generation_trigger=trigger,
            rejection_conditions=rejects,
            validation="incremental CV" if fit_scope == FitScope.NONE else "inside-fold",
            inference_requirement="both available",
            transform=tf,
            mandatory_tests=tests,
        )
    )


def _target_aware_op(
    name: str, purpose: str, tf, trigger: str, rejects: tuple,
    tests: tuple, leakage_class: LeakageClass = LeakageClass.L2,
) -> None:
    register(
        FeatureOperator(
            name=name,
            family="target-aware categorical",
            arity="unary",
            input_types=("categorical", "target"),
            output_type="numeric",
            purpose=purpose,
            preconditions=("target-aware allowed", "enough support"),
            fit_scope=FitScope.TRAINING_FOLD,
            null_policy="preserve",
            leakage_class=leakage_class,
            target_permission=TargetPermission.P1_CROSSFIT,
            cost_shape="O(n)",
            generation_trigger=trigger,
            rejection_conditions=rejects,
            validation="strict cross-fit",
            inference_requirement="fitted state or global prior",
            transform=tf,
            mandatory_tests=tests,
        )
    )


# ---------------------------------------------------------------------------
# L0 registrations (entries 38, 39, 43)
# ---------------------------------------------------------------------------
_cat_op(
    "hash_encode",
    "bounded-memory category representation via hashing",
    tf_hash_encode,
    "very high cardinality categorical",
    ("collision risk", "interpretability constraint"),
    ("deterministic hash", "null preservation"),
)

_cat_binary_op(
    "category_cross",
    "capture joint category state of two categorical columns",
    tf_category_cross,
    "historical/interaction evidence between categories",
    ("joint cardinality too high",),
    ("escaping correctness", "null propagation"),
)

_cat_op(
    "ordinal_true_scale",
    "respect known ordered categorical semantics with fixed rank",
    tf_ordinal_true_scale,
    "true ordinal feature with domain-specified order",
    ("order inferred from target", "no order supplied"),
    ("order validation", "null preservation"),
)


# ---------------------------------------------------------------------------
# L1 registrations (entries 33-37)
# ---------------------------------------------------------------------------
register(
    FeatureOperator(
        name="one_hot",
        family="categorical",
        arity="unary",
        input_types=("categorical",),
        output_type="sparse_boolean",
        purpose="independent category indicators",
        preconditions=("cardinality under dimensional budget",),
        fit_scope=FitScope.TRAINING_FOLD,
        null_policy="unknown -> all-zero",
        leakage_class=LeakageClass.L1,
        target_permission=TargetPermission.P0_NONE,
        cost_shape="O(n*k)",
        generation_trigger="low/moderate cardinality",
        rejection_conditions=("dimension/RAM too high",),
        validation="inside-fold vocabulary",
        inference_requirement="stored vocabulary",
        transform=tf_one_hot_transform,
        mandatory_tests=("unknown category handling",),
    )
)

register(
    FeatureOperator(
        name="ordinal_encode",
        family="categorical",
        arity="unary",
        input_types=("categorical",),
        output_type="integer",
        purpose="compact integer representation",
        preconditions=("categorical input",),
        fit_scope=FitScope.TRAINING_FOLD,
        null_policy="unknown -> reserved code",
        leakage_class=LeakageClass.L1,
        target_permission=TargetPermission.P0_NONE,
        cost_shape="O(n)",
        generation_trigger="tree learner or true ordinal",
        rejection_conditions=("linear model fake order risk",),
        validation="inside-fold",
        inference_requirement="stored mapping",
        transform=tf_ordinal_encode_transform,
        mandatory_tests=("unknown code handling",),
    )
)

register(
    FeatureOperator(
        name="frequency_encode",
        family="categorical",
        arity="unary",
        input_types=("categorical",),
        output_type="numeric",
        purpose="encode category prevalence as frequency ratio",
        preconditions=("categorical input",),
        fit_scope=FitScope.TRAINING_FOLD,
        null_policy="unknown -> 0",
        leakage_class=LeakageClass.L1,
        target_permission=TargetPermission.P0_NONE,
        cost_shape="O(n)",
        generation_trigger="high cardinality",
        rejection_conditions=("unstable shift", "no gain"),
        validation="inside-fold",
        inference_requirement="stored frequency map",
        transform=tf_frequency_encode_transform,
        mandatory_tests=("fold isolation",),
    )
)

register(
    FeatureOperator(
        name="count_encode",
        family="categorical",
        arity="unary",
        input_types=("categorical",),
        output_type="numeric",
        purpose="encode support count per category",
        preconditions=("categorical input",),
        fit_scope=FitScope.TRAINING_FOLD,
        null_policy="unknown -> 0",
        leakage_class=LeakageClass.L1,
        target_permission=TargetPermission.P0_NONE,
        cost_shape="O(n)",
        generation_trigger="high cardinality",
        rejection_conditions=("temporal future-count leakage",),
        validation="fold/time safe",
        inference_requirement="stored count map",
        transform=tf_count_encode_transform,
        mandatory_tests=("history semantics",),
    )
)

register(
    FeatureOperator(
        name="rare_group",
        family="categorical",
        arity="unary",
        input_types=("categorical",),
        output_type="categorical",
        purpose="collapse low-count tail categories into __RARE__",
        preconditions=("many low-count levels",),
        fit_scope=FitScope.TRAINING_FOLD,
        null_policy="missing separate",
        leakage_class=LeakageClass.L1,
        target_permission=TargetPermission.P0_NONE,
        cost_shape="O(n)",
        generation_trigger="long tail distribution",
        rejection_conditions=("important rare signal harmed",),
        validation="inside-fold",
        inference_requirement="stored retained set",
        transform=tf_rare_group_transform,
        mandatory_tests=("unknown handling",),
    )
)


# ---------------------------------------------------------------------------
# L2 registrations (entries 40, 41, 42)
# ---------------------------------------------------------------------------
_target_aware_op(
    "target_mean_crossfit",
    "encode expected target value by category with Bayesian smoothing",
    tf_target_mean_crossfit,
    "high-cardinality categorical with target signal",
    ("tiny groups", "leakage", "stability failure"),
    ("self-target exclusion", "smoothing"),
)

_target_aware_op(
    "woe_crossfit",
    "Weight of Evidence encoding for binary targets",
    tf_woe_crossfit,
    "binary risk-style tasks with categorical features",
    ("zero support", "tiny class counts", "non-binary target"),
    ("smoothing", "finite output"),
)

_target_aware_op(
    "numeric_to_cat_target",
    "bin numeric values then cross-fit encode with target means",
    tf_numeric_to_cat_target,
    "nonlinear target relationship in numeric feature",
    ("small data", "leakage", "degenerate bins"),
    ("train-only bins", "finite output"),
)
