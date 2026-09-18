"""Ensemble builder (doc 07).

Constructs and validates ensembles from multiple trial results:
- Ensemble eligibility: which trials can be combined
- Stacking guard: prevents overfitting in stacked ensembles
- Weight optimization: finds optimal ensemble weights
- Pareto front: multi-objective quality/cost tradeoff

Normative source: doc 07 section "Ensembles".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..contracts import (
    Direction, EnsembleSpec, TrialResult, TrialStatus,
)


@dataclass
class EnsemblePolicy:
    """Policy for ensemble construction (doc 07)."""

    min_trials: int = 2
    max_members: int = 10
    min_diversity: float = 0.1  # minimum pairwise metric diversity
    stack_overfit_threshold: float = 0.05  # max train-cv gap for stacking
    weight_min: float = 0.05  # minimum weight per member
    weight_method: str = "inverse_error"  # "equal" | "inverse_error" | "optimal"


@dataclass
class EnsembleCandidate:
    """A candidate ensemble with its evaluation."""

    member_ids: list[str]
    weights: list[float]
    cv_score: float = 0.0
    diversity_score: float = 0.0
    stack_gap: float = 0.0  # train_score - cv_score
    total_cost_s: float = 0.0
    passed_guard: bool = True
    guard_reason: str = ""


def check_eligibility(
    results: list[TrialResult],
    policy: EnsemblePolicy,
) -> list[TrialResult]:
    """Filter trial results to only those eligible for ensembling.

    Eligibility criteria (doc 07):
    - Must be COMPLETED
    - Must have at least one fold metric
    """
    eligible = []
    for r in results:
        if r.status != TrialStatus.COMPLETED:
            continue
        if not r.fold_metrics:
            continue
        eligible.append(r)
    return eligible


def compute_diversity(results: list[TrialResult]) -> float:
    """Compute pairwise diversity of trial fold metrics.

    Higher diversity = better ensemble potential.
    Uses coefficient of variation of fold metrics across trials.
    """
    if len(results) < 2:
        return 0.0

    means = []
    for r in results:
        if r.fold_metrics:
            m = sum(m.value for m in r.fold_metrics) / len(r.fold_metrics)
            means.append(m)

    if not means or len(means) < 2:
        return 0.0

    overall_mean = sum(means) / len(means)
    if overall_mean == 0:
        return 0.0

    var = sum((m - overall_mean) ** 2 for m in means) / len(means)
    cv = math.sqrt(var) / abs(overall_mean)
    return min(1.0, cv)  # cap at 1.0


def optimize_weights(
    results: list[TrialResult],
    policy: EnsemblePolicy,
) -> list[float]:
    """Compute ensemble weights for trial results.

    Methods:
    - equal: uniform weights
    - inverse_error: weight = 1/error (lower error = higher weight)
    - optimal: grid search for best weight combination
    """
    n = len(results)
    if n == 0:
        return []

    if policy.weight_method == "equal":
        return [1.0 / n] * n

    # Get mean scores for each trial
    scores = []
    for r in results:
        if r.fold_metrics:
            scores.append(sum(m.value for m in r.fold_metrics) / len(r.fold_metrics))
        else:
            scores.append(float("inf"))

    if policy.weight_method == "inverse_error":
        inv_errors = [1.0 / max(s, 1e-10) for s in scores]
        total = sum(inv_errors)
        weights = [ie / total for ie in inv_errors]
        # Enforce minimum weight
        weights = [max(w, policy.weight_min) for w in weights]
        total = sum(weights)
        return [w / total for w in weights]

    # "optimal": simple grid search over 2-member combinations
    if n == 2:
        best_score = float("inf")
        best_w = 0.5
        for w_pct in range(10, 100, 5):
            w = w_pct / 100.0
            # Simple weighted average score
            combined = w * scores[0] + (1 - w) * scores[1]
            if combined < best_score:
                best_score = combined
                best_w = w
        return [best_w, 1.0 - best_w]

    return [1.0 / n] * n  # fallback to equal


def check_stacking_guard(
    results: list[TrialResult],
    policy: EnsemblePolicy,
) -> tuple[bool, str]:
    """Check for overfitting risk in stacked ensembles (doc 07).

    The guard compares training metrics vs CV metrics for each member.
    If the gap is too large, stacking is rejected.
    """
    for r in results:
        if not r.fold_metrics or not r.train_metrics:
            continue
        cv_mean = sum(m.value for m in r.fold_metrics) / len(r.fold_metrics)
        train_mean = sum(m.value for m in r.train_metrics) / len(r.train_metrics)
        gap = abs(train_mean - cv_mean)
        if gap > policy.stack_overfit_threshold:
            return False, f"member {r.trial_id}: train-cv gap {gap:.4f} > {policy.stack_overfit_threshold}"

    return True, "stacking guard passed"


def build_ensemble(
    results: list[TrialResult],
    policy: Optional[EnsemblePolicy] = None,
) -> Optional[EnsembleSpec]:
    """Build the best ensemble from available trial results (doc 07).

    Steps:
    1. Filter to eligible trials
    2. Check diversity
    3. Optimize weights
    4. Check stacking guard
    5. Return EnsembleSpec if valid
    """
    policy = policy or EnsemblePolicy()

    eligible = check_eligibility(results, policy)
    if len(eligible) < policy.min_trials:
        return None

    # Limit to max_members
    eligible = eligible[:policy.max_members]

    # Check diversity
    diversity = compute_diversity(eligible)
    if diversity < policy.min_diversity:
        return None  # too homogeneous

    # Check stacking guard
    passed, _reason = check_stacking_guard(eligible, policy)
    if not passed:
        return None

    # Optimize weights
    weights = optimize_weights(eligible, policy)

    # Estimate ensemble cost
    total_cost = sum(
        sum(r.resource_measurements[0].wall_time_s or 0
            for r in [m] if r.resource_measurements)
        for m in eligible
    )

    member_ids = [r.trial_id for r in eligible]
    return EnsembleSpec(
        member_trial_ids=member_ids,
        weights=weights,
        stack_overfit_guard_result="passed",
        latency_estimate=total_cost,
    )


def pareto_front(
    results: list[TrialResult],
    quality_direction: Direction = Direction.MAXIMIZE,
) -> list[TrialResult]:
    """Find the Pareto front of quality vs cost (doc 07).

    A trial is on the Pareto front if no other trial is both better
    in quality and lower in cost.
    """
    def _quality(r: TrialResult) -> float:
        if not r.fold_metrics:
            return 0.0
        return sum(m.value for m in r.fold_metrics) / len(r.fold_metrics)

    def _cost(r: TrialResult) -> float:
        if not r.resource_measurements:
            return 0.0
        return sum(rm.wall_time_s or 0 for rm in r.resource_measurements)

    front = []
    for r in results:
        if r.status != TrialStatus.COMPLETED:
            continue
        dominated = False
        for other in results:
            if other.status != TrialStatus.COMPLETED or other is r:
                continue
            q_other = _quality(other)
            c_other = _cost(other)
            q_r = _quality(r)
            c_r = _cost(r)
            # other dominates r if better quality AND lower cost
            if quality_direction == Direction.MAXIMIZE:
                if q_other >= q_r and c_other <= c_r and (q_other > q_r or c_other < c_r):
                    dominated = True
                    break
            else:
                if q_other <= q_r and c_other <= c_r and (q_other < q_r or c_other < c_r):
                    dominated = True
                    break
        if not dominated:
            front.append(r)

    return front
