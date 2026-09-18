"""Actual model training with sklearn (doc 07, doc 15).

Provides TrialRunner with real model fitting using scikit-learn.
Supports the full trial lifecycle: create → fit → evaluate → report.

Normative source: doc 07 "Unified TrialSpec", doc 15 steps 0068-0090.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

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


def _evaluate_cv(model, X, y, task: str, n_folds: int = 5, seed: int = 42) -> dict[str, float]:
    """Cross-validate a model and return metrics."""
    from sklearn.model_selection import cross_val_score, KFold
    import numpy as np

    kf = KFold(n_splits=min(n_folds, max(2, len(y) // 2)), shuffle=True, random_state=seed)

    if task == "classification":
        scoring = "roc_auc" if len(set(y)) == 2 else "accuracy"
    else:
        scoring = "neg_mean_squared_error"

    try:
        scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)
        mean = float(np.mean(scores))
        if mean != mean:  # NaN — e.g. scorer/estimator task mismatch
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                    "scoring": "failed:" + scoring}
        return {
            "mean": mean,
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "scoring": scoring,
        }
    except Exception:
        # Fallback to simple train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        model.fit(X_train, y_train)
        if task == "classification":
            score = model.score(X_test, y_test)
        else:
            score = -model.score(X_test, y_test)  # negative MSE
        return {"mean": score, "std": 0.0, "min": score, "max": score, "scoring": "fallback"}


# ---------------------------------------------------------------------------
# TrialRunner (doc 07, doc 15)
# ---------------------------------------------------------------------------

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
            metrics = [
                MetricValue(
                    name="quality", value=cv_results["mean"],
                    direction=Direction.MAXIMIZE, split="cv_mean",
                ),
                MetricValue(
                    name="cv_std", value=cv_results["std"],
                    direction=Direction.MINIMIZE, split="cv_std",
                ),
            ]

            return TrialResult(
                trial_id=spec.trial_id,
                status=TrialStatus.COMPLETED,
                metrics=metrics,
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
