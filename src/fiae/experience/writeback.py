"""Experience store write-back and cost predictors (doc 06).

- ``write_case``: persists completed feature engineering cases to the store
- ``predict_cost``: estimates compute cost from feature characteristics
- ``predict_quality``: estimates likely quality gain from meta-features
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..contracts import (
    DatasetProfile, FeatureNode, MetricValue,
    ResourceMeasurement,
)
from ..ids import new_id
from .case import CaseRecord, CaseContext, CaseAction, CaseResult, CaseCost, CaseDiagnosis
from .metafeatures import extract_meta_features
from .store import ExperienceStore


@dataclass
class CostPredictor:
    """Predict compute cost from feature characteristics (doc 06).

    Uses a simple linear model calibrated from observed run data:
    cost_seconds = a * n_rows + b * n_features + c * operator_complexity + d
    """

    n_rows_coeff: float = 0.0001
    n_features_coeff: float = 0.01
    complexity_coeff: float = 0.5
    intercept: float = 0.1

    _observations: list[dict] = field(default_factory=list)

    def observe(
        self, n_rows: int, n_features: int, complexity: float, actual_seconds: float
    ) -> None:
        """Record an observation for calibration."""
        self._observations.append({
            "n_rows": n_rows, "n_features": n_features,
            "complexity": complexity, "seconds": actual_seconds,
        })

    def predict(
        self, n_rows: int, n_features: int, operator_cost_shape: str = "O(n)"
    ) -> float:
        """Predict cost in seconds."""
        complexity = _cost_complexity(operator_cost_shape)
        return (
            self.n_rows_coeff * n_rows
            + self.n_features_coeff * n_features
            + self.complexity_coeff * complexity
            + self.intercept
        )

    def calibrate(self) -> None:
        """Recalibrate coefficients from observations using least-squares."""
        if len(self._observations) < 5:
            return  # too few observations
        # Simple gradient descent
        lr = 0.0001
        for _ in range(100):
            grad = [0.0, 0.0, 0.0, 0.0]
            for obs in self._observations:
                x1 = obs["n_rows"]
                x2 = obs["n_features"]
                x3 = _cost_complexity("O(n)")
                pred = (self.n_rows_coeff * x1 + self.n_features_coeff * x2
                        + self.complexity_coeff * x3 + self.intercept)
                err = pred - obs["seconds"]
                grad[0] += err * x1
                grad[1] += err * x2
                grad[2] += err * x3
                grad[3] += err
            n = len(self._observations)
            self.n_rows_coeff -= lr * grad[0] / n
            self.n_features_coeff -= lr * grad[1] / n
            self.complexity_coeff -= lr * grad[2] / n
            self.intercept -= lr * grad[3] / n


def _cost_complexity(cost_shape: str) -> float:
    """Map cost shape string to a numeric complexity score."""
    shape = cost_shape.upper()
    if "NLOGN" in shape or "N LOG N" in shape:
        return 4.0
    if "N * WINDOW" in shape or "N*WINDOW" in shape:
        return 8.0
    if "NNZ*K" in shape:
        return 3.0
    if "NKP" in shape:
        return 5.0
    if "N*K" in shape:
        return 2.0
    return 1.0  # O(n)


@dataclass
class QualityPredictor:
    """Estimate likely quality gain from feature meta-characteristics (doc 06).

    Uses meta-features of the candidate feature to predict its utility:
    - operator family track record
    - input column characteristics (variance, null rate, cardinality)
    - interaction vs unary
    """

    family_weights: dict[str, float] = field(default_factory=lambda: {
        "numeric": 0.3, "numeric interaction": 0.25,
        "categorical": 0.2, "datetime": 0.15,
        "temporal": 0.2, "group aggregate": 0.15,
        "text": 0.1, "text representation": 0.1,
        "dimensionality": 0.05, "cluster": 0.05,
        "model-informed": 0.1,
    })

    def predict_gain(
        self, operator_family: str, input_null_fraction: float,
        input_variance: float, n_unique_inputs: int,
    ) -> float:
        """Predict expected quality gain (0-1 scale)."""
        family_prior = self.family_weights.get(operator_family, 0.1)
        # Low null rate is good
        null_factor = 1.0 - input_null_fraction
        # Moderate variance is best (too low = constant, too high = noisy)
        var_score = min(1.0, input_variance / 10.0) * (1.0 - min(1.0, input_variance / 1000.0))
        # Fewer unique inputs for interaction operators is better (more generalizable)
        cardinality_factor = 1.0 / (1.0 + math.log1p(n_unique_inputs) / 10.0)

        return family_prior * null_factor * (0.5 + 0.5 * var_score) * (0.7 + 0.3 * cardinality_factor)


def write_case(
    store: ExperienceStore,
    profile: DatasetProfile,
    features: list[FeatureNode],
    metrics: list[MetricValue],
    resources: list[ResourceMeasurement],
    task: str,
) -> str:
    """Write a completed feature engineering case to the experience store.

    Returns the case_id.
    """
    meta = extract_meta_features(profile)

    context = CaseContext(
        dataset_fingerprint=profile.dataset_fingerprint,
        rows=profile.rows_observed,
        columns=len(profile.columns),
        meta_features=meta,
        task=task,
    )

    action = CaseAction(
        features=[f.feature_id for f in features],
        operators=[f.operator for f in features],
        params=[f.params for f in features],
    )

    total_time = sum(r.wall_time_s or 0.0 for r in resources)
    total_memory = max((r.peak_rss_bytes or 0 for r in resources), default=0)

    result = CaseResult(
        metrics={m.name: m.value for m in metrics},
        feature_count=len(features),
    )

    cost = CaseCost(
        wall_time_s=total_time,
        memory_bytes=total_memory,
        cpu_seconds=sum(r.cpu_time_s or 0.0 for r in resources),
    )

    diagnosis = CaseDiagnosis(
        success=all(m.value > 0 for m in metrics) if metrics else False,
        failure_tags=[],
    )

    case = CaseRecord(
        case_id=new_id("case"),
        context=context,
        action=action,
        result=result,
        cost=cost,
        diagnosis=diagnosis,
    )

    store.write(case)
    return case.case_id
