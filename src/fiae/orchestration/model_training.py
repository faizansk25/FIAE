"""Actual model training with sklearn (doc 07, doc 15).

Provides TrialRunner with real model fitting using scikit-learn.
Supports the full trial lifecycle: create → fit → evaluate → report.

Normative source: doc 07 "Unified TrialSpec", doc 15 steps 0068-0090.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..contracts import (
    Direction, MetricValue, ResourceMeasurement, TrialResult,
    TrialStatus,
)
from ..ids import new_id


# ---------------------------------------------------------------------------
# TrialSpec (doc 07)
# ---------------------------------------------------------------------------

@dataclass
class TrialSpec:
    """Unified trial specification (doc 07)."""
    trial_id: str = ""
    feature_portfolio_id: str = ""
    model_family: str = "random_forest"
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    row_fraction: float = 1.0
    fold_count: int = 5
    seed: int = 42
    timeout_s: float = 300.0
    feature_fraction: float = 1.0

    def __post_init__(self):
        if not self.trial_id:
            self.trial_id = new_id("trial")


# ---------------------------------------------------------------------------
# Model training (doc 07)
# ---------------------------------------------------------------------------

def _get_sklearn_model(family: str, params: dict[str, Any]):
    """Instantiate an sklearn model from family name and hyperparameters."""
    try:
        from sklearn.ensemble import (
            RandomForestClassifier, RandomForestRegressor,
            GradientBoostingClassifier, GradientBoostingRegressor,
            AdaBoostClassifier, AdaBoostRegressor,
            BaggingClassifier, BaggingRegressor,
            ExtraTreesClassifier, ExtraTreesRegressor,
        )
        from sklearn.linear_model import (
            Ridge, Lasso, ElasticNet, SGDClassifier, SGDRegressor,
            LogisticRegression,
        )
        from sklearn.svm import SVC, SVR
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    except ImportError as err:
        raise ImportError("scikit-learn is required for model training") from err

    f = family.lower().replace("-", "_").replace(" ", "_")

    if f in ("random_forest", "rf"):
        if params.get("task") == "classification":
            return RandomForestClassifier(**{k: v for k, v in params.items() if k != "task"})
        return RandomForestRegressor(**{k: v for k, v in params.items() if k != "task"})
    if f in ("gradient_boosting", "gbm", "xgboost"):
        if params.get("task") == "classification":
            return GradientBoostingClassifier(**{k: v for k, v in params.items() if k != "task"})
        return GradientBoostingRegressor(**{k: v for k, v in params.items() if k != "task"})
    if f in ("extra_trees", "et"):
        if params.get("task") == "classification":
            return ExtraTreesClassifier(**{k: v for k, v in params.items() if k != "task"})
        return ExtraTreesRegressor(**{k: v for k, v in params.items() if k != "task"})
    if f in ("ada_boost", "adaboost"):
        if params.get("task") == "classification":
            return AdaBoostClassifier(**{k: v for k, v in params.items() if k != "task"})
        return AdaBoostRegressor(**{k: v for k, v in params.items() if k != "task"})
    if f in ("bagging",):
        if params.get("task") == "classification":
            return BaggingClassifier(**{k: v for k, v in params.items() if k != "task"})
        return BaggingRegressor(**{k: v for k, v in params.items() if k != "task"})
    if f in ("ridge",):
        return Ridge(**{k: v for k, v in params.items() if k != "task"})
    if f in ("lasso",):
        return Lasso(**{k: v for k, v in params.items() if k != "task"})
    if f in ("elastic_net", "elasticnet"):
        return ElasticNet(**{k: v for k, v in params.items() if k != "task"})
    if f in ("logistic_regression", "logistic"):
        return LogisticRegression(**{k: v for k, v in params.items() if k != "task"})
    if f in ("sgd",):
        if params.get("task") == "classification":
            return SGDClassifier(**{k: v for k, v in params.items() if k != "task"})
        return SGDRegressor(**{k: v for k, v in params.items() if k != "task"})
    if f in ("svm", "svc"):
        if params.get("task") == "classification":
            return SVC(**{k: v for k, v in params.items() if k != "task"})
        return SVR(**{k: v for k, v in params.items() if k != "task"})
    if f in ("knn", "k_neighbors", "kneighbors"):
        if params.get("task") == "classification":
            return KNeighborsClassifier(**{k: v for k, v in params.items() if k != "task"})
        return KNeighborsRegressor(**{k: v for k, v in params.items() if k != "task"})
    if f in ("decision_tree", "dt"):
        if params.get("task") == "classification":
            return DecisionTreeClassifier(**{k: v for k, v in params.items() if k != "task"})
        return DecisionTreeRegressor(**{k: v for k, v in params.items() if k != "task"})

    # Default fallback: random forest
    if params.get("task") == "classification":
        return RandomForestClassifier(n_estimators=100, random_state=params.get("random_state", 42))
    return RandomForestRegressor(n_estimators=100, random_state=params.get("random_state", 42))


@dataclass(frozen=True)
class MetricSpec:
    """What a task's primary metric *is*, and how sklearn must be asked.

    M40 (external audit): sklearn's ``neg_*`` scorers are negated losses
    that sklearn *maximizes* -- an optimization convention, not a domain
    fact. FIAE's own contract says "MSE, minimize" and converts at the
    sklearn boundary, so no downstream comparison has to know about the
    negation trick.
    """

    name: str
    sklearn_scorer: str
    direction: Direction
    # True when sklearn returns the negated loss and we un-negate it.
    negated: bool = False


BINARY_PRIMARY = MetricSpec(
    name="roc_auc", sklearn_scorer="roc_auc", direction=Direction.MAXIMIZE)
REGRESSION_PRIMARY = MetricSpec(
    name="mse", sklearn_scorer="neg_mean_squared_error",
    direction=Direction.MINIMIZE, negated=True)


def _resolve_spec(task: str, y: list) -> Optional[MetricSpec]:
    """The one metric spec this task may be judged by, or None.

    M40: the old code silently degraded ROC-AUC to accuracy whenever the
    labels were not binary, and reported every regression score under one
    name regardless of what it measured. A task with no defined primary
    metric returns None so the trial fails honestly.
    """
    if task == "classification":
        return BINARY_PRIMARY if len(set(y)) == 2 else None
    return REGRESSION_PRIMARY


def is_better(candidate: MetricValue, incumbent: MetricValue) -> bool:
    """Does ``candidate`` beat ``incumbent``? (M40)

    Comparison follows each metric's declared direction, so a MINIMIZE
    metric (MSE) is ranked the right way round automatically. Different
    metric identities are never compared; the same name with contradictory
    directions is an invariant violation and is refused rather than
    guessed at.
    """
    if candidate.name != incumbent.name:
        return False
    if candidate.direction != incumbent.direction:
        raise ValueError(
            f"metric {candidate.name!r} compared with conflicting "
            f"directions: {incumbent.direction} vs {candidate.direction}"
        )
    return (candidate.value > incumbent.value
            if candidate.direction is Direction.MAXIMIZE
            else candidate.value < incumbent.value)


def _evaluate_cv(model, X, y, task: str, n_folds: int = 5, seed: int = 42) -> dict[str, float]:
    """Cross-validate a model and return metrics."""
    from collections import Counter
    from sklearn.model_selection import (
        StratifiedKFold,
        cross_val_score,
        KFold,
    )
    import numpy as np

    spec = _resolve_spec(task, y)
    if spec is None:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                "folds": [],
                "scoring": "failed:no_defined_primary_metric:" + task}

    if task == "classification":
        # M40: fold count is bounded by the limiting unit -- how many
        # members the *minority* class has. Capping by len(set(y)) meant
        # every binary run silently used 2 folds instead of the 5 asked
        # for. An impossible fold count fails instead of being coerced:
        # max(2, 1) would hand sklearn a stratified split it cannot make.
        n_splits = min(n_folds, min(Counter(y).values()))
        if n_splits < 2:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                    "folds": [],
                    "scoring": "failed:insufficient_class_support_for_cv:" + spec.name}
        kf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
    else:
        # M40: never manufacture a fold count the data cannot support.
        n_splits = min(n_folds, len(y))
        if n_splits < 2:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                    "folds": [],
                    "scoring": "failed:insufficient_rows_for_cv:" + spec.name}
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    try:
        raw = cross_val_score(model, X, y, cv=kf, scoring=spec.sklearn_scorer)
        # Convert at the boundary: sklearn hands back the negated loss, and
        # FIAE publishes the loss itself (MSE, minimize).
        scores = -raw if spec.negated else raw
        mean = float(np.mean(scores))
        if mean != mean:  # NaN — e.g. scorer/estimator task mismatch
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                    "folds": [], "scoring": "failed:" + spec.name}
        return {
            "mean": mean,
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            # Real per-fold evidence, not copies of the aggregate. Item 5:
            # eligibility requires fold metrics, so this is what makes an
            # ensemble eligible at all.
            "folds": [float(s) for s in scores],
            "scoring": spec.name,
            "direction": spec.direction.value,
        }
    except Exception as exc:
        # M40: no silent train/test fallback. It swapped ROC-AUC for
        # accuracy on classification and reported -R^2 as if it were
        # negative MSE for regression, then labelled both "fallback" --
        # a score wearing the wrong name, compared against real CV scores.
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                "folds": [],
                "scoring": f"failed:{type(exc).__name__}:{spec.name}"}


# ---------------------------------------------------------------------------
# TrialRunner (doc 07, doc 15)
# ---------------------------------------------------------------------------

def primary_metric(trial: Any) -> Optional[MetricValue]:
    """The trial's promoted CV metric, identified by its own name.

    M40: callers used to hard-code ``name == "quality"``, which matched a
    ROC-AUC, an accuracy and a negated MSE equally well. Read the metric
    that carries the scorer's identity instead.
    """
    for m in getattr(trial, "metrics", []) or []:
        if m.name != "cv_std":
            return m
    return None


@dataclass
class TrialRunner:
    """Runs a single trial with actual model training (doc 07)."""
    default_timeout_s: float = 300.0
    default_n_folds: int = 5

    def run_trial(
        self,
        spec: TrialSpec,
        X: list[list[float]],
        y: list[float],
        task: str = "classification",
    ) -> TrialResult:
        """Execute a trial: instantiate model, train, evaluate, return result."""
        t0 = time.monotonic()

        try:
            import numpy as np
            X_arr = np.array(X, dtype=float)
            y_arr = np.array(y, dtype=float)

            # Apply row fraction
            n = int(len(y_arr) * spec.row_fraction)
            n = max(20, min(n, len(y_arr)))
            X_sub = X_arr[:n]
            y_sub = y_arr[:n]

            # Apply feature fraction
            n_features = X_sub.shape[1] if X_sub.ndim > 1 else 1
            n_feat = max(1, int(n_features * spec.feature_fraction))
            X_sub = X_sub[:, :n_feat]

            # Build params
            params = dict(spec.hyperparameters)
            params["task"] = task
            params["random_state"] = spec.seed
            # Only add n_estimators for ensemble models
            if (spec.model_family.lower() in ("random_forest", "rf", "gradient_boosting", "gbm",
                                              "extra_trees", "et", "ada_boost", "adaboost", "bagging")
                    and "n_estimators" not in params):
                params["n_estimators"] = 100

            # Train and evaluate
            model = _get_sklearn_model(spec.model_family, params)
            cv_results = _evaluate_cv(
                model, X_sub, y_sub, task,
                n_folds=spec.fold_count, seed=spec.seed,
            )
            if str(cv_results.get("scoring", "")).startswith("failed:"):
                # Scoring failed on every fold (task/estimator mismatch etc.)
                # — fail the trial honestly instead of reporting a fake 0.0.
                return TrialResult(
                    trial_id=spec.trial_id,
                    status=TrialStatus.FAILED,
                    resource_measurements=[
                        ResourceMeasurement(wall_time_s=time.monotonic() - t0)
                    ],
                    diagnosis={
                        "error": "cross-validation scoring failed ("
                                 + str(cv_results["scoring"]) + ")",
                        "model_family": spec.model_family,
                    },
                )

            # Fit final model on all data
            model.fit(X_sub, y_sub)
            model_bytes = 0
            try:
                import pickle
                model_bytes = len(pickle.dumps(model))
            except Exception:
                pass

            wall_time = time.monotonic() - t0
            scoring = str(cv_results["scoring"])
            metric_dir = Direction(cv_results["direction"])
            metrics = [
                # M40: the metric carries its own identity and direction.
                # "quality" was a bare float that could be a ROC-AUC, an
                # accuracy or a negated MSE, and every score was declared
                # MAXIMIZE -- which inverted ranking for any real loss.
                MetricValue(
                    name=scoring, value=cv_results["mean"],
                    direction=metric_dir, split="cv_mean",
                    aggregation="mean",
                ),
                MetricValue(
                    name="cv_std", value=cv_results["std"],
                    direction=Direction.MINIMIZE, split="cv_std",
                ),
            ]
            # M40 item 5: real per-fold evidence. Without these, ensemble
            # eligibility rejects every real trial and the canonical
            # fallback fabricates weights from rejected trials.
            fold_metrics = [
                MetricValue(
                    name=scoring, value=v, direction=metric_dir,
                    split="cv", fold=i,
                )
                for i, v in enumerate(cv_results.get("folds", []))
            ]

            return TrialResult(
                trial_id=spec.trial_id,
                status=TrialStatus.COMPLETED,
                metrics=metrics,
                fold_metrics=fold_metrics,
                resource_measurements=[
                    ResourceMeasurement(
                        wall_time_s=wall_time,
                        model_bytes=model_bytes,
                    )
                ],
                diagnosis={
                    "model_family": spec.model_family,
                    "cv_results": cv_results,
                    "n_rows_used": n,
                    "n_features_used": n_feat,
                },
            )

        except Exception as e:
            wall_time = time.monotonic() - t0
            return TrialResult(
                trial_id=spec.trial_id,
                status=TrialStatus.FAILED,
                resource_measurements=[
                    ResourceMeasurement(wall_time_s=wall_time)
                ],
                diagnosis={"error": str(e), "model_family": spec.model_family},
            )


# ---------------------------------------------------------------------------
# Fidelity ladder (doc 07)
# ---------------------------------------------------------------------------

LARGE_DATA_LADDER = [0.05, 0.16, 0.40, 1.0]
MEDIUM_DATA_LADDER = [0.16, 0.32, 0.64, 1.0]
SMALL_DATA_LADDER = [1.0]


def get_fidelity_ladder(n_rows: int) -> list[float]:
    """Select fidelity ladder based on dataset size (doc 07)."""
    if n_rows > 100_000:
        return LARGE_DATA_LADDER
    if n_rows > 10_000:
        return MEDIUM_DATA_LADDER
    return SMALL_DATA_LADDER


# ---------------------------------------------------------------------------
# Utility computation (doc 07)
# ---------------------------------------------------------------------------

def compute_utility(
    quality: float,
    fold_variance: float = 0.0,
    generalization_gap: float = 0.0,
    time_s: float = 0.0,
    memory_mb: float = 0.0,
    lambda_variance: float = 0.5,
    lambda_gap: float = 1.0,
    lambda_time: float = 0.1,
    lambda_ram: float = 0.05,
) -> float:
    """Compute promotion utility (doc 07).

    utility = normalized_quality
            - λ_variance * fold_variance
            - λ_gap * generalization_gap
            - λ_time * time
            - λ_ram * memory
    """
    return (
        quality
        - lambda_variance * fold_variance
        - lambda_gap * generalization_gap
        - lambda_time * time_s
        - lambda_ram * memory_mb
    )
