"""Benchmark baselines, dataset difficulty scoring (doc 12).

Provides baseline model comparisons and dataset difficulty metrics
to evaluate whether FIAE-generated features provide real value.

Normative source: doc 12 "Benchmark baselines", "Dataset difficulty scoring".
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Baseline models (doc 12)
# ---------------------------------------------------------------------------

@dataclass
class BaselineResult:
    """Result from a baseline model."""
    name: str
    score: float
    time_s: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


def run_baselines(
    X: list[list[float]],
    y: list[float],
    task: str = "classification",
    baselines: list[str] | None = None,
) -> list[BaselineResult]:
    """Run standard baseline models for comparison (doc 12).

    Returns a list of BaselineResult for each baseline model.
    """
    if not baselines:
        baselines = ["majority", "random", "linear"]

    results = []
    for name in baselines:
        t0 = time.monotonic()
        try:
            score = _run_single_baseline(name, X, y, task)
            dt = time.monotonic() - t0
            results.append(BaselineResult(name=name, score=score, time_s=dt))
        except Exception as e:
            dt = time.monotonic() - t0
            results.append(BaselineResult(
                name=name, score=float("nan"), time_s=dt,
                details={"error": str(e)},
            ))

    return results


def _run_single_baseline(
    name: str, X: list[list[float]], y: list[float], task: str
) -> float:
    """Run a single baseline and return its score."""
    try:
        import numpy as np
        from sklearn.model_selection import cross_val_score, KFold
    except ImportError:
        # Fallback: simple heuristic
        if name == "majority":
            from collections import Counter
            mode_count = Counter(y).most_common(1)[0][1]
            return mode_count / len(y) if y else 0.0
        return 0.0

    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=float)
    n_splits = min(5, max(2, len(y) // 2))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    if name == "majority":
        from collections import Counter
        mode_val = Counter(y).most_common(1)[0][0]
        preds = [mode_val] * len(y)
        if task == "classification":
            return sum(p == t for p, t in zip(preds, y)) / len(y) if y else 0.0
        return -np.mean((np.array(preds) - y_arr) ** 2)

    if name == "random":
        rng = np.random.RandomState(42)
        if task == "classification":
            unique = list(set(y))
            preds = rng.choice(unique, size=len(y))
            return sum(p == t for p, t in zip(preds, y)) / len(y) if y else 0.0
        preds = rng.uniform(y_arr.min(), y_arr.max(), size=len(y))
        return -np.mean((preds - y_arr) ** 2)

    if name == "linear":
        from sklearn.linear_model import Ridge, LogisticRegression
        if task == "classification":
            model = LogisticRegression(max_iter=1000, random_state=42)
            scoring = "roc_auc" if len(set(y)) == 2 else "accuracy"
        else:
            model = Ridge(alpha=1.0)
            scoring = "neg_mean_squared_error"
        scores = cross_val_score(model, X_arr, y_arr, cv=kf, scoring=scoring)
        return float(np.mean(scores))

    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        if task == "classification":
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            scoring = "roc_auc" if len(set(y)) == 2 else "accuracy"
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            scoring = "neg_mean_squared_error"
        scores = cross_val_score(model, X_arr, y_arr, cv=kf, scoring=scoring)
        return float(np.mean(scores))

    return 0.0


# ---------------------------------------------------------------------------
# Dataset difficulty scoring (doc 12)
# ---------------------------------------------------------------------------

@dataclass
class DifficultyProfile:
    """Dataset difficulty profile (doc 12)."""
    overall_score: float = 0.0  # 0 (easy) to 1 (hard)
    n_rows: int = 0
    n_features: int = 0
    class_balance: float = 1.0  # 1.0 = perfectly balanced
    feature_relevance: float = 0.0  # 0 = noise, 1 = strong signal
    noise_level: float = 0.0
    nonlinearity: float = 0.0
    missingness: float = 0.0
    cardinality_ratio: float = 0.0  # unique_values / n_rows
    difficulty_class: str = "medium"  # "easy" | "medium" | "hard" | "very_hard"

    def to_dict(self) -> dict:
        return {
            "overall_score": self.overall_score,
            "difficulty_class": self.difficulty_class,
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "class_balance": self.class_balance,
            "feature_relevance": self.feature_relevance,
            "noise_level": self.noise_level,
            "nonlinearity": self.nonlinearity,
            "missingness": self.missingness,
        }


def score_dataset_difficulty(
    X: list[list[float]],
    y: list[float],
    task: str = "classification",
) -> DifficultyProfile:
    """Score dataset difficulty on a 0-1 scale (doc 12).

    Factors:
    - sample size (fewer = harder)
    - class balance / target dispersion
    - feature relevance (linear correlation)
    - noise level
    - missingness
    - nonlinearity
    """
    n_rows = len(y)
    n_features = len(X[0]) if X and X[0] else 0

    # 1. Sample size difficulty
    size_score = max(0.0, 1.0 - math.log10(max(n_rows, 1)) / 5.0)

    # 2. Class balance / target dispersion
    if task == "classification":
        from collections import Counter
        counts = Counter(y)
        total = len(y)
        if total > 0:
            max_class = max(counts.values())
            balance = 1.0 - (max_class / total)
        else:
            balance = 1.0
        balance_score = 1.0 - balance  # imbalanced = harder
    else:
        # Regression: dispersion relative to range
        y_arr = [v for v in y if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if y_arr:
            y_range = max(y_arr) - min(y_arr)
            y_std = (sum((v - sum(y_arr)/len(y_arr))**2 for v in y_arr) / len(y_arr)) ** 0.5
            balance_score = min(1.0, y_std / max(abs(y_range), 1e-9))
        else:
            balance_score = 0.5

    # 3. Feature relevance (avg absolute correlation)
    relevance_scores = []
    for j in range(min(n_features, 20)):
        col = [X[i][j] for i in range(min(n_rows, len(X))) if X[i][j] is not None]
        if len(col) < 10:
            continue
        y_sub = y[:len(col)]
        mean_x = sum(col) / len(col)
        mean_y = sum(y_sub) / len(y_sub)
        cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(col, y_sub)) / len(col)
        std_x = (sum((a - mean_x)**2 for a in col) / len(col)) ** 0.5
        std_y = (sum((b - mean_y)**2 for b in y_sub) / len(y_sub)) ** 0.5
        if std_x > 0 and std_y > 0:
            relevance_scores.append(abs(cov / (std_x * std_y)))
    feature_relevance = sum(relevance_scores) / max(len(relevance_scores), 1)
    relevance_difficulty = 1.0 - feature_relevance  # low relevance = harder

    # 4. Missingness
    total_cells = n_rows * max(n_features, 1)
    missing_cells = sum(
        1 for i in range(min(n_rows, len(X)))
        for j in range(n_features)
        if X[i][j] is None or (isinstance(X[i][j], float) and math.isnan(X[i][j]))
    )
    missingness = missing_cells / max(total_cells, 1)

    # 5. Noise (from baseline comparison)
    noise_score = min(1.0, max(0.0, 1.0 - feature_relevance * 2))

    # 6. Nonlinearity (from linear vs nonlinear model gap - simplified)
    nonlinearity = max(0.0, min(1.0, 0.5 - feature_relevance))

    # Overall difficulty
    overall = (
        0.20 * size_score
        + 0.15 * balance_score
        + 0.25 * relevance_difficulty
        + 0.10 * missingness
        + 0.15 * noise_score
        + 0.15 * nonlinearity
    )
    overall = max(0.0, min(1.0, overall))

    if overall < 0.25:
        difficulty_class = "easy"
    elif overall < 0.50:
        difficulty_class = "medium"
    elif overall < 0.75:
        difficulty_class = "hard"
    else:
        difficulty_class = "very_hard"

    return DifficultyProfile(
        overall_score=overall,
        n_rows=n_rows,
        n_features=n_features,
        class_balance=balance if task == "classification" else (1.0 - balance_score),
        feature_relevance=feature_relevance,
        noise_level=noise_score,
        nonlinearity=nonlinearity,
        missingness=missingness,
        difficulty_class=difficulty_class,
    )


# ---------------------------------------------------------------------------
# Feature value assessment (doc 12)
# ---------------------------------------------------------------------------

@dataclass
class FeatureValueAssessment:
    """Assessment of whether FIAE features add value over baselines."""
    fiae_score: float = 0.0
    best_baseline_score: float = 0.0
    improvement_pct: float = 0.0
    provides_value: bool = False
    recommendation: str = ""


def assess_feature_value(
    fiae_score: float,
    baseline_results: list[BaselineResult],
    min_improvement_pct: float = 2.0,
) -> FeatureValueAssessment:
    """Assess whether FIAE features provide value over baselines (doc 12)."""
    valid = [b for b in baseline_results if not math.isnan(b.score)]
    if not valid:
        return FeatureValueAssessment(
            fiae_score=fiae_score,
            recommendation="No valid baselines for comparison",
        )

    best_baseline = max(valid, key=lambda b: b.score)
    improvement = ((fiae_score - best_baseline.score) / max(abs(best_baseline.score), 1e-9)) * 100

    if improvement >= min_improvement_pct:
        rec = f"FIAE provides {improvement:.1f}% improvement over {best_baseline.name}"
    elif improvement > 0:
        rec = f"Marginal improvement ({improvement:.1f}%) over {best_baseline.name}"
    else:
        rec = f"No improvement over {best_baseline.name} ({improvement:.1f}%)"

    return FeatureValueAssessment(
        fiae_score=fiae_score,
        best_baseline_score=best_baseline.score,
        improvement_pct=improvement,
        provides_value=improvement >= min_improvement_pct,
        recommendation=rec,
    )
