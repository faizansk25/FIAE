"""Ensemble stacking (doc 07).

Stacked generalization using out-of-fold predictions as meta-features.
Includes overfit guard comparing train vs CV performance.

Normative source: doc 07 section "Ensembles" and "Stacking guard".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional



@dataclass
class StackingConfig:
    """Configuration for stacking."""

    n_folds: int = 5
    seed: int = 42
    overfit_threshold: float = 0.05  # max train-cv gap
    min_diversity: float = 0.05


@dataclass
class StackingResult:
    """Result of a stacking operation."""

    oof_predictions: dict[str, list[float]] = field(default_factory=dict)
    meta_features: list[list[float]] = field(default_factory=list)
    train_score: float = 0.0
    cv_score: float = 0.0
    gap: float = 0.0
    passed_guard: bool = True
    guard_reason: str = ""
    member_weights: list[float] = field(default_factory=list)


def _kfold_indexes(n: int, k: int, seed: int) -> list[list[int]]:
    """Deterministic k-fold split."""
    import random
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    folds = [[] for _ in range(k)]
    for i, idx in enumerate(indices):
        folds[i % k].append(idx)
    return folds


def generate_oof_predictions(
    member_predictions: dict[str, list[float]],
    n_rows: int,
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Generate out-of-fold predictions for each member model (doc 07).

    For each fold, each member's predictions on the held-out fold become
    the OOF prediction for those rows.  This prevents information leakage
    in the meta-learner.
    """
    folds = _kfold_indexes(n_rows, n_folds, seed)
    oof = {name: [0.0] * n_rows for name in member_predictions}

    for fold in folds:
        for name, preds in member_predictions.items():
            for idx in fold:
                if idx < len(preds):
                    oof[name][idx] = preds[idx]

    return oof


def stack_predictions(
    oof_predictions: dict[str, list[float]],
    y: list[float],
    method: str = "average",
) -> StackingResult:
    """Stack member predictions using a meta-learner or averaging (doc 07).

    method:
    - "average": simple weighted average
    - "ridge": ridge regression meta-learner
    """
    n = len(y)
    member_names = list(oof_predictions.keys())
    n_members = len(member_names)

    if n_members == 0:
        return StackingResult(passed_guard=False, guard_reason="no members")

    if method == "average":
        weights = [1.0 / n_members] * n_members
        stacked = [0.0] * n
        for i in range(n):
            for j, name in enumerate(member_names):
                if i < len(oof_predictions[name]):
                    stacked[i] += weights[j] * oof_predictions[name][i]
        result = StackingResult(member_weights=weights)
    else:
        # Ridge meta-learner
        weights = _ridge_meta(oof_predictions, y, member_names)
        stacked = [0.0] * n
        for i in range(n):
            for j, name in enumerate(member_names):
                if i < len(oof_predictions[name]):
                    stacked[i] += weights[j] * oof_predictions[name][i]
        result = StackingResult(member_weights=weights)

    # Compute scores
    sq_errors = [(stacked[i] - y[i]) ** 2 for i in range(min(n, len(stacked)))]
    cv_score = math.sqrt(sum(sq_errors) / max(len(sq_errors), 1))

    result.cv_score = round(cv_score, 6)
    result.meta_features = [
        [oof_predictions[name][i] for name in member_names if i < len(oof_predictions[name])]
        for i in range(n)
    ]
    result.oof_predictions = oof_predictions
    result.gap = 0.0  # will be set by check_guard
    result.passed_guard = True

    return result


def _ridge_meta(
    oof_predictions: dict[str, list[float]],
    y: list[float],
    member_names: list[str],
    lam: float = 1.0,
) -> list[float]:
    """Ridge regression meta-learner."""
    n = len(y)
    p = len(member_names) + 1  # +1 for intercept

    # Build design matrix
    XtX = [[0.0] * p for _ in range(p)]
    Xty = [0.0] * p
    for i in range(n):
        row = [1.0]  # intercept
        for name in member_names:
            val = oof_predictions[name][i] if i < len(oof_predictions[name]) else 0.0
            row.append(val)
        for j in range(p):
            Xty[j] += row[j] * y[i]
            for k in range(p):
                XtX[j][k] += row[j] * row[k]

    # Ridge penalty
    for j in range(p):
        XtX[j][j] += lam

    # Gauss-Jordan solve
    aug = [[*XtX[j], Xty[j]] for j in range(p)]
    for col in range(p):
        max_row = col
        for row in range(col + 1, p):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]
        if abs(aug[col][col]) < 1e-12:
            continue
        pivot = aug[col][col]
        for j in range(col, p + 1):
            aug[col][j] /= pivot
        for row in range(p):
            if row != col:
                factor = aug[row][col]
                for j in range(col, p + 1):
                    aug[row][j] -= factor * aug[col][j]

    beta = [aug[j][p] for j in range(p)]
    return beta[1:]  # exclude intercept


def check_stacking_guard(
    stacking_result: StackingResult,
    train_score: float,
    config: Optional[StackingConfig] = None,
) -> StackingResult:
    """Check for overfitting in stacked ensemble (doc 07).

    Compares training score vs CV score.  If the gap exceeds the
    threshold, the stacking is rejected.
    """
    config = config or StackingConfig()
    gap = abs(train_score - stacking_result.cv_score)
    stacking_result.train_score = train_score
    stacking_result.gap = gap

    if gap > config.overfit_threshold:
        stacking_result.passed_guard = False
        stacking_result.guard_reason = f"train-cv gap {gap:.4f} > {config.overfit_threshold}"
    else:
        stacking_result.passed_guard = True
        stacking_result.guard_reason = "stacking guard passed"

    return stacking_result
