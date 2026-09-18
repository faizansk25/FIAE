"""F4 progressive evaluation + F6 final stability (doc 04).

F4 re-evaluates candidates on progressively larger row budgets: the gain must
stay positive at every stage and must not collapse relative to the first
stage. F6 re-measures the gain across independent CV seeds; a feature is
stable only if it helps in every seed with bounded coefficient of variation.
Both append StageVerdicts and reuse the pure-stdlib probe machinery from
`probe.py` (NFR-002).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .contracts import Task
from .probe import ProbePolicy, _cv_metric
from .problem.splits import kfold_indexes
from .search.records import FeatureAcceptanceRecord, StageVerdict


@dataclass
class EvaluatePolicy:
    """F4/F6 policy thresholds (policy, not universal constants)."""

    f4_stages: tuple = (0.25, 0.5, 1.0)  # row-budget fractions
    f4_min_gain: float = 1e-9
    f4_min_gain_retention: float = 0.5
    f6_seeds: tuple = (0, 1, 2)
    f6_max_gain_cv: float = 0.5
    f6_min_gain: float = 1e-9
    probe: Optional[ProbePolicy] = None

    def __post_init__(self):
        if self.probe is None:
            self.probe = ProbePolicy()


def _gain_at_rows(base_columns, candidate, y, m, pp, task) -> float:
    folds = kfold_indexes(m, pp.n_folds, pp.seed)
    base_m = _cv_metric(
        [list(c)[:m] for c in base_columns], y[:m], folds, task, pp.ridge_lambda
    )
    with_m = _cv_metric(
        [*[list(c)[:m] for c in base_columns], list(candidate)[:m]],
        y[:m], folds, task, pp.ridge_lambda,
    )
    return base_m - with_m


def f4_progressive_eval(
    record: FeatureAcceptanceRecord,
    base_columns,
    candidate,
    y: list,
    task: Task,
    policy: Optional[EvaluatePolicy] = None,
) -> StageVerdict:
    """F4: measure the candidate's gain at increasing row budgets."""
    policy = policy or EvaluatePolicy()
    n = len(y)
    metrics: dict = {"stages": []}
    if n == 0 or any(len(c) != n for c in [*base_columns, candidate]):
        v = StageVerdict("F4", False, "length mismatch or empty target", metrics)
        record.add_stage(v)
        record.final_reason = v.reason or "F4 rejected"
        return v

    first_gain = None
    final_gain = None
    for frac in policy.f4_stages:
        m = max(8, min(n, round(n * frac)))
        gain = _gain_at_rows(base_columns, candidate, y, m, policy.probe, task)
        metrics["stages"].append({"rows": m, "gain": round(gain, 6)})
        if first_gain is None:
            first_gain = gain
        final_gain = gain
        if gain <= policy.f4_min_gain:
            v = StageVerdict(
                "F4", False, f"gain vanished at rows={m} ({gain:.3e})", metrics
            )
            record.add_stage(v)
            record.final_reason = v.reason or "F4 rejected"
            return v

    retention = (final_gain / first_gain) if first_gain and first_gain > 0 else 0.0
    metrics["gain_retention"] = round(retention, 4)
    if retention < policy.f4_min_gain_retention:
        v = StageVerdict(
            "F4", False, f"gain collapsed to {retention:.0%} of first stage", metrics
        )
        record.add_stage(v)
        record.final_reason = v.reason or "F4 rejected"
        return v

    v = StageVerdict("F4", True, f"gain held ({retention:.0%} retained)", metrics)
    record.add_stage(v)
    return v


def f6_final_stability(
    record: FeatureAcceptanceRecord,
    base_columns,
    candidate,
    y: list,
    task: Task,
    policy: Optional[EvaluatePolicy] = None,
) -> StageVerdict:
    """F6: re-measure the gain across independent CV seeds."""
    policy = policy or EvaluatePolicy()
    n = len(y)
    metrics: dict = {"seeds": []}
    if n == 0 or any(len(c) != n for c in [*base_columns, candidate]):
        v = StageVerdict("F6", False, "length mismatch or empty target", metrics)
        record.add_stage(v)
        record.final_reason = v.reason or "F6 rejected"
        return v

    gains = []
    for seed in policy.f6_seeds:
        folds = kfold_indexes(n, policy.probe.n_folds, seed)
        base_m = _cv_metric(
            list(base_columns), y, folds, task, policy.probe.ridge_lambda
        )
        with_m = _cv_metric(
            [*base_columns, list(candidate)], y, folds, task,
            policy.probe.ridge_lambda,
        )
        gain = base_m - with_m
        metrics["seeds"].append({"seed": seed, "gain": round(gain, 6)})
        gains.append(gain)

    mean = sum(gains) / len(gains)
    var = sum((g - mean) ** 2 for g in gains) / len(gains)
    cv = (var ** 0.5 / abs(mean)) if mean != 0 else float("inf")
    metrics["gain_mean"] = round(mean, 6)
    metrics["gain_cv"] = round(cv, 4) if cv != float("inf") else None

    record.fold_stability = round(1.0 / (1.0 + cv), 4) if cv != float("inf") else 0.0
    record.scores["f6_gain_mean"] = round(mean, 6)

    if any(g <= policy.f6_min_gain for g in gains):
        v = StageVerdict("F6", False, "gain not positive across all seeds", metrics)
    elif cv > policy.f6_max_gain_cv:
        v = StageVerdict("F6", False, f"unstable gain (cv={cv:.3f})", metrics)
    else:
        v = StageVerdict("F6", True, f"stable gain (cv={cv:.3f})", metrics)
    record.add_stage(v)
    if not v.passed:
        record.final_reason = v.reason or "F6 rejected"
    return v
