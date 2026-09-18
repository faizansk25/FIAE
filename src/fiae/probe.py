"""F3 incremental probe + greedy portfolio selection (doc 04).

F3 asks the only question that matters (doc 04 hypothesis principle): does
adding this candidate to the current portfolio improve a *valid* metric on a
small held-out split? A tiny ridge regression (pure stdlib, small k normal
equations solved exactly by Gauss-Jordan) serves as the cheap probe model for
numeric targets; binary targets are probed as clipped-probability Brier.

The probe never accepts a feature into the final portfolio by itself -- it
produces `incremental_gain` evidence on the candidate's
FeatureAcceptanceRecord. The greedy forward selector then composes a
portfolio, skipping candidates redundant with already-selected outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from .contracts import Task
from .problem.splits import kfold_indexes
from .search.records import FeatureAcceptanceRecord, StageVerdict


@dataclass
class ProbePolicy:
    """F3 policy thresholds (policy, not universal constants)."""

    n_folds: int = 3
    seed: int = 0
    ridge_lambda: float = 1e-6
    min_gain: float = 1e-9          # require strictly positive improvement
    corr_max: float = 0.98          # redundancy threshold in the selector


def _solve_normal_equations(a: list[list[float]], b: list[float],
                            lam: float = 1e-6) -> list[float]:
    """Solve (a^T a + lam I) w = a^T b exactly via Gauss-Jordan (small k)."""
    k = len(a[0]) if a else 0
    m = [[sum(a[r][i] * a[r][j] for r in range(len(a))) for j in range(k)]
         + [sum(a[r][i] * b[r] for r in range(len(a)))] for i in range(k)]
    for i in range(k):
        m[i][i] += lam  # policy ridge for numerical stability
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-18:
            return [0.0] * k
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        m[col] = [v / pv for v in m[col]]
        for r in range(k):
            if r != col and m[r][col] != 0.0:
                f = m[r][col]
                m[r] = [v - f * w for v, w in zip(m[r], m[col])]
    return [m[i][k] for i in range(k)]


def _design(columns: list[Sequence[float]], rows: Sequence[int]) -> list[list[float]]:
    return [[1.0] + [float(cols[r]) for cols in columns] for r in rows]


def _rmse(pred: Sequence[float], y: Sequence[float], rows: Sequence[int]) -> float:
    return math.sqrt(
        sum((p - y[r]) ** 2 for p, r in zip(pred, rows)) / max(len(rows), 1)
    )


def _brier(pred: Sequence[float], y: Sequence[float], rows: Sequence[int]) -> float:
    return sum(
        (min(max(p, 0.0), 1.0) - y[r]) ** 2 for p, r in zip(pred, rows)
    ) / max(len(rows), 1)


def _cv_metric(
    columns: list[Sequence[float]],
    y: list[float],
    folds: list[tuple[list[int], list[int]]],
    task: Task,
    lam: float,
) -> float:
    errs: list[float] = []
    for train, val in folds:
        if not train or not val:
            continue
        w = _solve_normal_equations(_design(columns, train),
                                    [y[r] for r in train], lam)
        if all(v == 0.0 for v in w):
            continue
        pred = [sum(wi * xi for wi, xi in zip(w, row))
                for row in _design(columns, val)]
        f = _rmse if task is Task.REGRESSION else _brier
        errs.append(f(pred, y, val))
    return sum(errs) / len(errs) if errs else float("inf")


def f3_incremental_probe(
    record: FeatureAcceptanceRecord,
    base_columns: list[Sequence[float]],
    candidate: Sequence[float],
    y: list[float],
    task: Task,
    policy: Optional[ProbePolicy] = None,
) -> StageVerdict:
    """F3: K-fold CV metric of [base] vs [base + candidate].

    Records the verdict on ``record`` and sets ``record.incremental_gain`` =
    metric(base) - metric(base + candidate) (positive means the candidate
    helps). The candidate itself is not retained.
    """
    policy = policy or ProbePolicy()
    n = len(y)
    metrics: dict = {"n_rows": n, "n_base": len(base_columns), "task": task.value}
    if n == 0 or any(len(c) != n for c in [*base_columns, candidate]):
        v = StageVerdict("F3", False, "length mismatch or empty target", metrics)
        record.add_stage(v)
        record.final_reason = v.reason or "F3 rejected"
        return v

    folds = kfold_indexes(n, policy.n_folds, policy.seed)
    base_m = _cv_metric(base_columns, y, folds, task, policy.ridge_lambda)
    with_m = _cv_metric([*base_columns, candidate], y, folds, task,
                        policy.ridge_lambda)
    gain = base_m - with_m  # both metrics: lower is better
    metrics["metric_base"] = round(base_m, 6)
    metrics["metric_with_candidate"] = round(with_m, 6)

    record.incremental_gain = round(gain, 6)
    record.scores["f3_gain"] = record.incremental_gain
    if gain <= policy.min_gain:
        v = StageVerdict("F3", False, f"no incremental gain ({gain:.3e})", metrics)
    else:
        v = StageVerdict("F3", True, f"incremental gain {gain:.3e}", metrics)
    record.add_stage(v)
    if not v.passed:
        record.final_reason = v.reason or "F3 rejected"
    return v


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    if n < 2 or len(b) != n:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (yv - mb) for x, yv in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((yv - mb) ** 2 for yv in b))
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / (da * db)


def select_portfolio(
    candidates: dict[str, Sequence[float]],
    y: list[float],
    task: Task,
    policy: Optional[ProbePolicy] = None,
    max_features: int = 8,
) -> tuple[list[str], dict[str, float]]:
    """Greedy forward selection over F2-surviving candidate outputs.

    Repeatedly probes every remaining candidate against the already-selected
    base, takes the best strictly-positive gain, and skips candidates whose
    output is near-duplicate (|pearson| > corr_max) with a selected one.
    Returns (selected names in acceptance order, gains per selected name).
    """
    policy = policy or ProbePolicy()
    selected: list[str] = []
    gains: dict[str, float] = {}
    remaining = list(candidates.keys())

    while remaining and len(selected) < max_features:
        # redundancy filter against selected outputs
        filtered = [
            name for name in remaining
            if all(abs(pearson(candidates[name], candidates[s])) <= policy.corr_max
                   for s in selected)
        ]
        remaining = filtered
        if not remaining:
            break

        best_name: Optional[str] = None
        best_gain = policy.min_gain
        for name in remaining:
            rec = FeatureAcceptanceRecord(feature_id=f"cand_{name}", operator=name)
            f3_incremental_probe(
                rec, [candidates[s] for s in selected], candidates[name],
                y, task, policy,
            )
            if (rec.incremental_gain or 0.0) > best_gain:
                best_name, best_gain = name, rec.incremental_gain

        if best_name is None:
            break
        selected.append(best_name)
        gains[best_name] = best_gain
        remaining.remove(best_name)
    return selected, gains
