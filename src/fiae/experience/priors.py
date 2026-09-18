"""Retrieval priors and Bayesian aggregation (doc 06).

RetrievalPrior holds the aggregated knowledge from similar past cases.
Bayesian smoothing prevents over-confidence from small sample sizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# -----------------------------------------------------------------------------
# Prior components
# -----------------------------------------------------------------------------
@dataclass
class FeatureFamilyPrior:
    """Prior belief about a feature family's usefulness."""

    family: str
    successes: int = 0
    attempts: int = 0
    posterior_mean: float = 0.5  # default: uninformative

    def update(self, success: bool, alpha: float = 1.0, beta: float = 1.0) -> None:
        """Update with a new observation using Bayesian smoothing."""
        self.attempts += 1
        if success:
            self.successes += 1
        self.posterior_mean = (self.successes + alpha) / (self.attempts + alpha + beta)


@dataclass
class ModelFamilyPrior:
    """Prior belief about a model family's usefulness."""

    family: str
    successes: int = 0
    attempts: int = 0
    posterior_mean: float = 0.5
    best_hyperparameters: dict[str, Any] = field(default_factory=dict)

    def update(self, success: bool, alpha: float = 1.0, beta: float = 1.0) -> None:
        """Update with a new observation."""
        self.attempts += 1
        if success:
            self.successes += 1
        self.posterior_mean = (self.successes + alpha) / (self.attempts + alpha + beta)


# -----------------------------------------------------------------------------
# Full retrieval prior
# -----------------------------------------------------------------------------
@dataclass
class RetrievalPrior:
    """Aggregated priors from experience retrieval.

    This is the output of the retrieval engine, providing search-order
    guidance without overriding current-dataset evidence.
    """

    feature_family_priors: dict[str, FeatureFamilyPrior] = field(default_factory=dict)
    transformation_priors: dict[str, float] = field(default_factory=dict)
    model_family_priors: dict[str, ModelFamilyPrior] = field(default_factory=dict)
    hyperparameter_priors: dict[str, dict[str, Any]] = field(default_factory=dict)
    split_priors: dict[str, float] = field(default_factory=dict)
    ensemble_priors: dict[str, float] = field(default_factory=dict)
    cost_priors: dict[str, float] = field(default_factory=dict)
    failure_priors: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    ood_score: float = 0.0
    source_count: int = 0

    def top_feature_families(self, n: int = 5) -> list[str]:
        """Return the top N feature families by posterior mean."""
        sorted_families = sorted(
            self.feature_family_priors.items(),
            key=lambda kv: kv[1].posterior_mean,
            reverse=True,
        )
        return [name for name, _ in sorted_families[:n]]

    def top_model_families(self, n: int = 3) -> list[str]:
        """Return the top N model families by posterior mean."""
        sorted_models = sorted(
            self.model_family_priors.items(),
            key=lambda kv: kv[1].posterior_mean,
            reverse=True,
        )
        return [name for name, _ in sorted_models[:n]]

    def generic(self) -> "RetrievalPrior":
        """Return a generic, uninformative prior."""
        return RetrievalPrior(confidence=0.0, ood_score=1.0)


# -----------------------------------------------------------------------------
# Bayesian aggregation
# -----------------------------------------------------------------------------
def bayesian_posterior(
    successes: int,
    attempts: int,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """Compute a Bayesian posterior mean with Beta prior.

    Uses a Beta(alpha, beta) prior, which defaults to uniform (alpha=beta=1).

    Parameters
    ----------
    successes:
        Number of successful outcomes.
    attempts:
        Total number of attempts.
    alpha:
        Prior pseudo-count for successes (default 1.0 for uniform prior).
    beta:
        Prior pseudo-count for failures (default 1.0 for uniform prior).

    Returns
    -------
    float
        Posterior mean estimate in [0, 1].
    """
    return (successes + alpha) / (attempts + alpha + beta)
