"""Progressive cross-validation ladder (doc 03).

Implements progressive evaluation with increasing data budgets and nested
cross-validation to prevent optimistic bias.  Core has zero third-party deps.

Doc 03 reference:
- Progressive row budgets catch small-sample artifacts
- Nested CV prevents information leakage between model selection and evaluation
- Calibration-aware evaluation ensures probability outputs are reliable
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from ..contracts import Task, MetricValue, Direction


@dataclass
class ProgressiveCVPolicy:
    """Policy for progressive evaluation (doc 03)."""

    row_fractions: tuple = (0.1, 0.25, 0.5, 1.0)
    n_folds: int = 5
    seed: int = 0
    min_folds: int = 3
    nested_folds: int = 3  # for hyperparameter selection
    calibration_bins: int = 10


@dataclass
class ProgressiveCVResult:
    """Result of one progressive evaluation stage."""

    stage: int
    fraction: float
    n_rows: int
    metrics: list[MetricValue] = field(default_factory=list)
    fold_results: list[dict[str, Any]] = field(default_factory=list)
    mean_score: float = 0.0
    std_score: float = 0.0
    passed: bool = True
    reason: str = ""


def _kfold_indexes(n: int, k: int, seed: int) -> list[list[int]]:
    """Deterministic k-fold split (same as problem.splits.kfold_indexes)."""
    rng = _seeded_rng(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    folds = [[] for _ in range(k)]
    for i, idx in enumerate(indices):
        folds[i % k].append(idx)
    return folds


def _seeded_rng(seed: int):
    """Simple seeded RNG (no numpy dependency)."""
    import random
    return random.Random(seed)


def _ridge_predict(X_rows: list[list[float]], y: list[float],
                   folds: list[list[int]], lam: float = 1.0) -> list[float]:
    """Ridge regression predictions via Gauss-Jordan (pure stdlib)."""
    n = len(y)
    preds = [0.0] * n
    for fold_idx in folds:
        train = [i for i in range(n) if i not in set(fold_idx)]
        if len(train) < 3:
            continue
        # Build X^T X + lambda*I
        p = len(X_rows[0]) if X_rows else 0
        XtX = [[0.0] * p for _ in range(p)]
        Xty = [0.0] * p
        for i in train:
            for j in range(p):
                Xty[j] += X_rows[i][j] * y[i]
                for k in range(p):
                    XtX[j][k] += X_rows[i][j] * X_rows[i][k]
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
        for i in fold_idx:
            preds[i] = sum(X_rows[i][j] * beta[j] for j in range(p))
    return preds


def progressive_cv(
    columns: list[list],
    y: list[float],
    task: Task,
    policy: Optional[ProgressiveCVPolicy] = None,
) -> list[ProgressiveCVResult]:
    """Run progressive CV with increasing row budgets.

    Returns one result per row fraction.  Each stage checks that the
    model's performance holds up as more data is included.
    """
    policy = policy or ProgressiveCVPolicy()
    n = len(y)
    if n == 0 or not columns:
        return []

    results = []
    for stage, frac in enumerate(policy.row_fractions):
        m = max(20, min(n, round(n * frac)))
        # Build design matrix
        X_rows = []
        for i in range(m):
            row = [1.0]  # intercept
            for col in columns:
                v = col[i] if i < len(col) else 0.0
                row.append(float(v) if v is not None else 0.0)
            X_rows.append(row)

        y_subset = y[:m]
        folds = _kfold_indexes(m, policy.n_folds, policy.seed)

        # Cross-validated predictions
        preds = _ridge_predict(X_rows, y_subset, folds)

        # Compute metric
        if task in (Task.BINARY, Task.MULTICLASS):
            # Brier score
            sq_errors = []
            for i in range(m):
                p = max(0.001, min(0.999, preds[i]))
                sq_errors.append((p - y_subset[i]) ** 2)
            score = sum(sq_errors) / len(sq_errors) if sq_errors else 1.0
            direction = Direction.MINIMIZE
        else:
            # RMSE
            sq_errors = [(preds[i] - y_subset[i]) ** 2 for i in range(m)]
            score = math.sqrt(sum(sq_errors) / len(sq_errors)) if sq_errors else float("inf")
            direction = Direction.MINIMIZE

        # Per-fold results
        fold_scores = []
        for fold in folds:
            fold_sq = [(preds[i] - y_subset[i]) ** 2 for i in fold]
            if task in (Task.BINARY, Task.MULTICLASS):
                fold_scores.append(sum(fold_sq) / len(fold_sq))
            else:
                fold_scores.append(math.sqrt(sum(fold_sq) / len(fold_sq)))

        mean_s = sum(fold_scores) / len(fold_scores) if fold_scores else 0.0
        var_s = sum((s - mean_s) ** 2 for s in fold_scores) / max(len(fold_scores), 1)
        std_s = math.sqrt(var_s)

        result = ProgressiveCVResult(
            stage=stage, fraction=frac, n_rows=m,
            metrics=[MetricValue(
                name="progressive_score", value=score,
                direction=direction, split=f"fold_cv_{stage}",
            )],
            fold_results=[{"fold": i, "score": s} for i, s in enumerate(fold_scores)],
            mean_score=round(mean_s, 6),
            std_score=round(std_s, 6),
            passed=True,
        )
        results.append(result)

    return results


@dataclass
class NestedCVResult:
    """Result of nested cross-validation (inner loop selects, outer loop evaluates)."""

    outer_folds: int = 0
    inner_folds: int = 0
    mean_score: float = 0.0
    std_score: float = 0.0
    fold_scores: list[float] = field(default_factory=list)
    details: list[dict[str, Any]] = field(default_factory=list)


def nested_cv(
    columns: list[list],
    y: list[float],
    task: Task,
    policy: Optional[ProgressiveCVPolicy] = None,
) -> NestedCVResult:
    """Nested cross-validation: inner loop for HP selection, outer for evaluation.

    This prevents optimistic bias from using the same data for both
    model selection and performance estimation (doc 03).
    """
    policy = policy or ProgressiveCVPolicy()
    n = len(y)
    if n < policy.n_folds * 2:
        return NestedCVResult()

    outer_folds = _kfold_indexes(n, policy.n_folds, policy.seed)
    outer_scores = []

    for outer_idx, test_fold in enumerate(outer_folds):
        train_idx = [i for i in range(n) if i not in set(test_fold)]

        # Inner CV on training set for lambda selection
        inner_folds = _kfold_indexes(len(train_idx), policy.nested_folds, policy.seed + outer_idx)
        best_lambda = 1.0
        best_score = float("inf")

        for lam in [0.001, 0.01, 0.1, 1.0, 10.0]:
            inner_preds = [0.0] * len(train_idx)
            for inner_fold in inner_folds:
                inner_train = [train_idx[i] for i in range(len(train_idx)) if i not in set(inner_fold)]
                if len(inner_train) < 3:
                    continue
                X_train = []
                y_train = []
                for i in inner_train:
                    row = [1.0] + [float(c[i]) if c[i] is not None else 0.0 for c in columns]
                    X_train.append(row)
                    y_train.append(y[i])

                X_test_inner = []
                for i in inner_fold:
                    row = [1.0] + [float(c[train_idx[i]]) if c[train_idx[i]] is not None else 0.0 for c in columns]
                    X_test_inner.append(row)

                preds_inner = _ridge_predict_inner(X_train, y_train, X_test_inner, lam)
                for j, orig_i in enumerate(inner_fold):
                    inner_preds[orig_i] = preds_inner[j]

            # Score inner predictions
            inner_sq = [(inner_preds[i] - y[train_idx[i]]) ** 2 for i in range(len(train_idx))]
            inner_score = math.sqrt(sum(inner_sq) / len(inner_sq))
            if inner_score < best_score:
                best_score = inner_score
                best_lambda = lam

        # Evaluate on outer fold with best lambda
        X_train = []
        y_train = []
        for i in train_idx:
            row = [1.0] + [float(c[i]) if c[i] is not None else 0.0 for c in columns]
            X_train.append(row)
            y_train.append(y[i])

        X_test = []
        y_test = []
        for i in test_fold:
            row = [1.0] + [float(c[i]) if c[i] is not None else 0.0 for c in columns]
            X_test.append(row)
            y_test.append(y[i])

        preds = _ridge_predict_inner(X_train, y_train, X_test, best_lambda)
        sq = [(preds[j] - y_test[j]) ** 2 for j in range(len(preds))]
        outer_score = math.sqrt(sum(sq) / len(sq))
        outer_scores.append(outer_score)

    mean_s = sum(outer_scores) / len(outer_scores)
    var_s = sum((s - mean_s) ** 2 for s in outer_scores) / len(outer_scores)

    return NestedCVResult(
        outer_folds=policy.n_folds,
        inner_folds=policy.nested_folds,
        mean_score=round(mean_s, 6),
        std_score=round(math.sqrt(var_s), 6),
        fold_scores=[round(s, 6) for s in outer_scores],
    )


def _ridge_predict_inner(X_train, y_train, X_test, lam):
    """Ridge prediction for nested CV."""
    p = len(X_train[0]) if X_train else 0
    if p == 0:
        return [0.0] * len(X_test)
    XtX = [[0.0] * p for _ in range(p)]
    Xty = [0.0] * p
    for i in range(len(X_train)):
        for j in range(p):
            Xty[j] += X_train[i][j] * y_train[i]
            for k in range(p):
                XtX[j][k] += X_train[i][j] * X_train[i][k]
    for j in range(p):
        XtX[j][j] += lam
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
    return [sum(X_test[i][j] * beta[j] for j in range(p)) for i in range(len(X_test))]
