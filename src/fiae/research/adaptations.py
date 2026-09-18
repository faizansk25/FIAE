"""Research adaptation decisions (doc 14).

Documents how FIAE adapts ideas from 12 reference systems into its
architecture.  Each adaptation records the source system, the adopted
idea, and the specific FIAE design decision.

Normative source: doc 14 "Reference System Research".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AdaptationDecision:
    """One adaptation from a reference system to FIAE."""

    source_system: str
    source_url: str
    adopted_idea: str
    fiae_design: str
    rationale: str
    status: str = "implemented"  # "implemented" | "planned" | "rejected"


# Complete catalog of adaptation decisions (doc 14)
ADAPTATIONS: list[AdaptationDecision] = [
    AdaptationDecision(
        source_system="H2O Driverless AI",
        source_url="https://docs.h2o.ai/driverless-ai/",
        adopted_idea="Automatic feature engineering with genetic algorithm search",
        fiae_design="Proposal-based feature generation with greedy/beam portfolio search",
        rationale="Genetic algorithms are expensive; greedy forward selection with F3 probe is sufficient for initial pipeline. Beam search reserved for later refinement.",
        status="implemented",
    ),
    AdaptationDecision(
        source_system="H2O Driverless AI",
        source_url="https://docs.h2o.ai/driverless-ai/",
        adopted_idea="Transform reliability scoring via cross-validation",
        fiae_design="F3 incremental probe with K-fold CV; F4 progressive evaluation; F6 multi-seed stability",
        rationale="Three-stage reliability filtering (F3/F4/F6) catches unstable features that single-pass CV misses.",
        status="implemented",
    ),
    AdaptationDecision(
        source_system="H2O Driverless AI",
        source_url="https://docs.h2o.ai/driverless-ai/",
        adopted_idea="Rolling/lag/shift operators with time-aware validation",
        fiae_design="Temporal operators (lag, lead, diff, rolling_*, expanding_mean, ewma) with no-future invariant",
        rationale="No-future property (insert-after-emit) prevents temporal leakage. Time-aware splitting via doc 03.",
        status="implemented",
    ),
    AdaptationDecision(
        source_system="DataRobot",
        source_url="https://www.datarobot.com/",
        adopted_idea="Model blueprint search with automatic algorithm selection",
        fiae_design="Model registry with 12 families, routing by data characteristics, trial orchestration",
        rationale="DataRobot's blueprint search is proprietary; FIAE uses open model routing with resource-aware selection.",
        status="implemented",
    ),
    AdaptationDecision(
        source_system="DataRobot",
        source_url="https://www.datarobot.com/",
        adopted_idea="Feature importance and interaction detection",
        fiae_design="F5 complementarity gate (Pearson correlation), greedy portfolio with redundancy filtering",
        rationale="Pearson correlation is a cheap proxy for redundancy. More sophisticated interaction detection reserved for later.",
        status="implemented",
    ),
    AdaptationDecision(
        source_system="Auto-sklearn",
        source_url="https://automl.github.io/auto-sklearn/",
        adopted_idea="Meta-learning from experience store",
        fiae_design="ExperienceStore with SQLite, R0-R3 retrieval hierarchy, Bayesian prior updating",
        rationale="Auto-sklearn's meta-features are adapted with FIAE's own metafeature extraction (doc 06). Retrieval uses cosine distance on meta-feature vectors.",
        status="implemented",
    ),
    AdaptationDecision(
        source_system="Auto-sklearn",
        source_url="https://automl.github.io/auto-sklearn/",
        adopted_idea="Successive halving for HPO resource allocation",
        fiae_design="Successive halving + Hyperband algorithms in HPO module",
        rationale="Resource-efficient pruning: start many cheap trials, promote top fraction to higher budgets.",
        status="implemented",
    ),
    AdaptationDecision(
        source_system="TPOT",
        source_url="https://github.com/EpistasisLab/tpot",
        adopted_idea="Genetic programming for pipeline synthesis",
        fiae_design="DAG-based feature graph with canonical signatures and duplicate collapse",
        rationale="GP is powerful but expensive. FIAE uses structured proposal generation with DAG deduplication for faster iteration.",
        status="planned",
    ),
    AdaptationDecision(
        source_system="Featuretools",
        source_url="https://www.featuretools.com/",
        adopted_idea="Deep feature synthesis via entityset relationships",
        fiae_design="Group aggregate operators (group_count, group_mean, etc.) with entity key detection",
        rationale="Featuretools' full DFS is expensive; FIAE focuses on the most common aggregation patterns.",
        status="implemented",
    ),
    AdaptationDecision(
        source_system="scikit-learn",
        source_url="https://scikit-learn.org/",
        adopted_idea="Transformer API (fit/transform/predict) with pipeline composition",
        fiae_design="FittedPipeline class with fit/transform lifecycle for L1 operators",
        rationale="Adopted sklearn's fit/transform pattern. FittedPipeline adds state serialization and split_id-keyed isolation.",
        status="implemented",
    ),
    AdaptationDecision(
        source_system="scikit-learn",
        source_url="https://scikit-learn.org/",
        adopted_idea="Cross-validation utilities (KFold, StratifiedKFold)",
        fiae_design="problem.splits module with stratified, kfold, group, time-ordered strategies",
        rationale="Extended sklearn's CV with time-ordered backtest and group-aware splitting for panel data.",
        status="implemented",
    ),
    AdaptationDecision(
        source_system="scikit-learn",
        source_url="https://scikit-learn.org/",
        adopted_idea="Pipeline and ColumnTransformer for preprocessing",
        fiae_design="PipelineIR for structured feature pipeline representation and code generation",
        rationale="FIAE's PipelineIR is richer than sklearn's: includes fit scope, leakage class, and can generate standalone Python code.",
        status="implemented",
    ),
    AdaptationDecision(
        source_system="LightGBM",
        source_url="https://lightgbm.readthedocs.io/",
        adopted_idea="Categorical feature handling native to tree learners",
        fiae_design="CatBoost/LightGBM in model registry with supports_categorical flag",
        rationale="Native categorical support avoids one-hot explosion for high-cardinality features.",
        status="implemented",
    ),
    AdaptationDecision(
        source_system="XGBoost",
        source_url="https://xgboost.readthedocs.io/",
        adopted_idea="Regularized gradient boosting with early stopping",
        fiae_design="FidelitySpec with early stopping support in TrialRunner",
        rationale="Early stopping prevents overfitting in GBM trials. Integrated into the trial orchestration layer.",
        status="implemented",
    ),
    AdaptationDecision(
        source_system="Optuna",
        source_url="https://optuna.org/",
        adopted_idea="TPE (Tree-structured Parzen Estimator) for HPO",
        fiae_design="TPE-like selection in HPO module (simplified version)",
        rationale="Full TPE requires kernel density estimation. Simplified version splits good/bad trials and samples near good points.",
        status="implemented",
    ),
    AdaptationDecision(
        source_system="Optuna",
        source_url="https://optuna.org/",
        adopted_idea="Pruning with median pruner and hyperband pruner",
        fiae_design="Successive halving pruning in TrialRunner",
        rationale="Median pruner requires full trial history; successive halving is simpler and nearly as effective.",
        status="implemented",
    ),
    AdaptationDecision(
        source_system="MLbox",
        source_url="https://github.com/mlbox-ml/mlbox",
        adopted_idea="Automatic feature selection via LASSO and tree-based importance",
        fiae_design="Greedy portfolio selection with F3 probe + F4/F6 stability gates",
        rationale="FIAE's multi-stage filtering (F0→F2→F3→F5→F4→F6) is more thorough than single-pass importance ranking.",
        status="implemented",
    ),
]


def get_adaptations_by_status(status: str) -> list[AdaptationDecision]:
    """Get all adaptations with a given status."""
    return [a for a in ADAPTATIONS if a.status == status]


def get_adaptations_by_source(system: str) -> list[AdaptationDecision]:
    """Get all adaptations from a specific source system."""
    return [a for a in ADAPTATIONS if system.lower() in a.source_system.lower()]


def adaptation_summary() -> dict[str, Any]:
    """Summarize all adaptation decisions."""
    by_status = {}
    for a in ADAPTATIONS:
        by_status.setdefault(a.status, []).append(a.source_system)

    by_source = {}
    for a in ADAPTATIONS:
        by_source.setdefault(a.source_system, []).append(a.adopted_idea)

    return {
        "total": len(ADAPTATIONS),
        "implemented": sum(1 for a in ADAPTATIONS if a.status == "implemented"),
        "planned": sum(1 for a in ADAPTATIONS if a.status == "planned"),
        "rejected": sum(1 for a in ADAPTATIONS if a.status == "rejected"),
        "source_systems": len(by_source),
        "by_source": {k: len(v) for k, v in by_source.items()},
    }
