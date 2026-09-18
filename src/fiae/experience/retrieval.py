"""Experience retrieval with family-weighted meta-feature distance (doc 06).

Implements R0 (hard compatibility), R1 (cheap buckets), R2 (weighted
meta-feature distance), and R3 (learned ranking) from the experience store
retrieval hierarchy.

Normative source: doc 06 "Retrieval hierarchy", "Retrieval pseudocode",
"R3 — learned ranking".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..contracts import Task
from .case import CaseRecord
from .store import ExperienceStore
from .priors import RetrievalPrior, FeatureFamilyPrior, ModelFamilyPrior


# ---------------------------------------------------------------------------
# Retrieval prior output (doc 06)
# ---------------------------------------------------------------------------

# RetrievalPrior imported from priors.py — no duplicate definition


# ---------------------------------------------------------------------------
# RetrievalQuery and RetrievalEngine (backward compatibility)
# ---------------------------------------------------------------------------
@dataclass
class RetrievalQuery:
    """Query for experience retrieval."""
    task: Task = Task.AUTO
    rows: int = 0
    columns: int = 0
    meta: dict = field(default_factory=dict)
    meta_features: Any = None  # DatasetMetaFeatures or dict
    k: int = 10


class RetrievalEngine:
    """High-level retrieval engine wrapping the store."""

    def __init__(self, store: ExperienceStore):
        self.store = store

    def retrieve(self, query: RetrievalQuery) -> RetrievalPrior:
        task_str = query.task.value if hasattr(query.task, 'value') else str(query.task)
        return retrieve_priors(
            self.store, task_str, query.rows, query.columns,
            query_meta=query.meta, k=query.k,
        )


# ---------------------------------------------------------------------------
# Family weights for R2 distance (doc 06)
# ---------------------------------------------------------------------------

DEFAULT_FAMILY_WEIGHTS: dict[str, float] = {
    "shape": 0.10,
    "type_composition": 0.15,
    "missingness": 0.10,
    "numeric_distribution": 0.12,
    "categorical_structure": 0.12,
    "target": 0.15,
    "relationship": 0.13,
    "runtime_source": 0.08,
}

# Hard compatibility filters (R0)
HARD_FILTERS = {
    "task", "target_semantics", "time_structure", "group_structure",
}


def _robust_distance(a: float, b: float, scale: float = 1.0) -> float:
    """Robust distance normalized by scale, clipped to [0, 1]."""
    if scale <= 0:
        scale = 1.0
    return min(1.0, abs(a - b) / scale)


def _meta_distance(
    query_meta: dict[str, float],
    case_meta: dict[str, float],
    family_weights: dict[str, float] | None = None,
) -> float:
    """Family-weighted meta-feature distance (R2).

    No noisy high-dimensional family may dominate the distance simply
    because it has more dimensions (doc 06).
    """
    weights = family_weights or DEFAULT_FAMILY_WEIGHTS

    # Group features by family prefix
    families: dict[str, list[tuple[float, float]]] = {}
    for key in query_meta:
        family = key.split("_")[0] if "_" in key else key
        q_val = query_meta.get(key, 0.0)
        c_val = case_meta.get(key, 0.0)
        families.setdefault(family, []).append((q_val, c_val))

    total_distance = 0.0
    total_weight = 0.0
    for family, pairs in families.items():
        w = weights.get(family, 0.05)
        # Average distance within family, capped at 1
        family_dist = sum(_robust_distance(a, b) for a, b in pairs) / max(len(pairs), 1)
        family_dist = min(1.0, family_dist)
        total_distance += w * family_dist
        total_weight += w

    return total_distance / max(total_weight, 1e-9)


# ---------------------------------------------------------------------------
# Bayesian smoothing (doc 06)
# ---------------------------------------------------------------------------

def bayesian_smooth(
    successes: int, attempts: int, alpha: float = 1.0, beta: float = 1.0
) -> float:
    """Bayesian-smoothed success rate.

    posterior = (successes + alpha) / (attempts + alpha + beta)
    """
    return (successes + alpha) / (attempts + alpha + beta)


# ---------------------------------------------------------------------------
# R0 — hard compatibility filter
# ---------------------------------------------------------------------------

def _hard_filter(
    case: CaseRecord, query_task: str
) -> bool:
    """R0: Reject cases with incompatible task type."""
    task_val = case.context.task
    # Handle both Task enum and string
    if hasattr(task_val, 'value'):
        task_str = task_val.value
    else:
        task_str = str(task_val)
    return query_task.lower() in task_str.lower() or task_str.lower() in query_task.lower() or task_str == 'auto'


# ---------------------------------------------------------------------------
# R1 — cheap bucket rank
# ---------------------------------------------------------------------------

def _cheap_bucket_rank(
    cases: list[CaseRecord], query_rows: int, query_cols: int
) -> list[CaseRecord]:
    """R1: Bucket by task, row scale, width. Cheap pre-filter."""
    filtered = []
    for c in cases:
        # Extract rows/cols from metadata or user_constraints
        meta = c.context.dataset_meta_features
        constraints = c.context.user_constraints
        # Handle both dict and DatasetMetaFeatures
        if hasattr(meta, 'raw_stats'):
            raw = meta.raw_stats
        elif isinstance(meta, dict):
            raw = meta
        else:
            raw = {}
        if isinstance(constraints, dict):
            rows = raw.get("rows", constraints.get("rows", 0))
            cols = raw.get("columns", constraints.get("columns", 0))
        else:
            rows = raw.get("rows", 0)
            cols = raw.get("columns", 0)
        # Row scale bucket: within 10x
        if rows > 0 and query_rows > 0:
            ratio = max(rows / query_rows, query_rows / rows)
            if ratio > 10.0:
                continue
        # Width bucket: within 5x
        if cols > 0 and query_cols > 0:
            ratio = max(cols / query_cols, query_cols / cols)
            if ratio > 5.0:
                continue
        filtered.append(c)
    return filtered


# ---------------------------------------------------------------------------
# R2 — weighted meta-feature distance retrieval
# ---------------------------------------------------------------------------

def retrieve_priors(
    store: ExperienceStore,
    query_task,
    query_rows: int = 0,
    query_cols: int = 0,
    query_meta: dict[str, float] | None = None,
    k: int = 10,
    family_weights: dict[str, float] | None = None,
) -> RetrievalPrior:
    """Full R0→R1→R2 retrieval pipeline (doc 06).

    1. R0: hard filter by task compatibility
    2. R1: cheap bucket rank by row/col scale
    3. R2: weighted meta-feature distance, top-k
    4. Bayesian aggregate into RetrievalPrior
    """
    # Handle RetrievalQuery as second argument for backward compatibility
    if isinstance(query_task, RetrievalQuery):
        rq = query_task
        query_task = rq.task.value if hasattr(rq.task, 'value') else str(rq.task)
        query_rows = rq.rows
        query_cols = rq.columns
        # Support both .meta and .meta_features
        if rq.meta_features is not None:
            if hasattr(rq.meta_features, 'families'):
                # DatasetMetaFeatures: flatten to dict
                flat = {}
                for fam, features in rq.meta_features.families.items():
                    fam_name = fam.value if hasattr(fam, 'value') else str(fam)
                    for fname, fval in features.items():
                        if isinstance(fval, (int, float)):
                            flat[f"{fam_name}_{fname}"] = fval
                query_meta = flat if flat else None
            elif isinstance(rq.meta_features, dict):
                query_meta = rq.meta_features
            else:
                query_meta = rq.meta or None
        else:
            query_meta = rq.meta or None
        k = rq.k

    # R0: Get all cases from store
    try:
        all_cases = store.all_cases()
    except Exception:
        return RetrievalPrior(confidence=0.0, ood_score=1.0)

    if not all_cases:
        return RetrievalPrior(confidence=0.0, ood_score=1.0)

    # R0: hard compatibility filter
    compatible = [_hard_filter(c, query_task) for c in all_cases]
    compatible = [c for c, ok in zip(all_cases, compatible) if ok]

    if not compatible:
        return RetrievalPrior(confidence=0.0, ood_score=1.0)

    # R1: cheap bucket rank
    bucketed = _cheap_bucket_rank(compatible, query_rows, query_cols)

    if not bucketed:
        return RetrievalPrior(confidence=0.1, ood_score=0.9)

    # R2: compute distances
    if query_meta is None:
        query_meta = {}

    scored = []
    for case in bucketed:
        # Get meta_features from dataset_meta_features or metadata
        case_meta = {}
        if isinstance(case.context.dataset_meta_features, dict):
            case_meta = dict(case.context.dataset_meta_features)
        # Flatten nested dicts
        flat_meta = {}
        for k, v in case_meta.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, (int, float)):
                        flat_meta[f"{k}_{k2}"] = v2
            elif isinstance(v, (int, float)):
                flat_meta[k] = v
        dist = _meta_distance(query_meta, flat_meta, family_weights)
        scored.append((dist, case))

    # Sort by distance, take top-k
    scored.sort(key=lambda x: x[0])
    nearest = [case for _, case in scored[:k]]

    # Bayesian aggregate priors using priors.py structure
    prior = RetrievalPrior(
        source_count=len(nearest),
    )

    # Aggregate transformation priors from nearest neighbors
    trans_counts: dict[str, int] = {}
    trans_attempts: dict[str, int] = {}
    total_success = 0
    total_attempts = 0

    for case in nearest:
        success = case.is_success()
        total_attempts += 1
        if success:
            total_success += 1

        # Aggregate feature portfolio
        if case.action.feature_portfolio:
            for fid in case.action.feature_portfolio:
                if isinstance(fid, str):
                    trans_attempts[fid] = trans_attempts.get(fid, 0) + 1
                    if success:
                        trans_counts[fid] = trans_counts.get(fid, 0) + 1

        # Aggregate model family priors
        if case.action.model_family:
            mf = case.action.model_family
            if mf not in prior.model_family_priors:
                prior.model_family_priors[mf] = ModelFamilyPrior(family=mf)
            prior.model_family_priors[mf].update(success)

        # Aggregate metrics
        if case.result.primary_metric_name and case.result.primary_metric_value is not None:
            prior.cost_priors[case.result.primary_metric_name] = (
                prior.cost_priors.get(case.result.primary_metric_name, 0.0) + case.result.primary_metric_value
            )
        for mname, mval in case.result.secondary_metrics.items():
            if isinstance(mval, (int, float)):
                prior.cost_priors[mname] = prior.cost_priors.get(mname, 0.0) + mval

    # Smooth transformation priors and populate feature_family_priors
    for op, attempts in trans_attempts.items():
        successes = trans_counts.get(op, 0)
        prior.transformation_priors[op] = bayesian_smooth(successes, attempts)
        # Also populate feature_family_priors as FeatureFamilyPrior objects
        ffp = FeatureFamilyPrior(family=op, successes=successes, attempts=attempts)
        ffp.posterior_mean = bayesian_smooth(successes, attempts)
        prior.feature_family_priors[op] = ffp

    # Normalize cost priors
    n = max(len(nearest), 1)
    for key in prior.cost_priors:
        prior.cost_priors[key] /= n

    # Confidence from retrieval distance and count
    if scored:
        min_dist = scored[0][0]
        count_factor = min(1.0, len(nearest) / k)
        prior.confidence = count_factor * (1.0 - min_dist)
    else:
        prior.confidence = 0.0

    # OOD score
    prior.ood_score = 1.0 - prior.confidence

    # Bayesian overall success rate
    prior.failure_priors["overall_success_rate"] = bayesian_smooth(
        total_success, total_attempts
    )

    return prior


# ---------------------------------------------------------------------------
# R3 — Learned ranking (optional, doc 06)
# ---------------------------------------------------------------------------

@dataclass
class R3Ranker:
    """Simple learned ranking model for R3 retrieval.

    Uses a linear scoring function trained on past (query, case, utility)
    triples. Only activates after enough distinct datasets exist.
    """
    weights: dict[str, float] = field(default_factory=dict)
    bias: float = 0.0
    n_training_samples: int = 0
    min_samples_for_activation: int = 20

    def is_active(self) -> bool:
        return self.n_training_samples >= self.min_samples_for_activation

    def score(
        self, query_meta: dict[str, float], case_meta: dict[str, float]
    ) -> float:
        """Score a (query, case) pair. Higher = more useful."""
        if not self.weights:
            return 0.0
        dot = 0.0
        for key in set(query_meta.keys()) | set(case_meta.keys()):
            q = query_meta.get(key, 0.0)
            c = case_meta.get(key, 0.0)
            feature = q * c  # interaction feature
            w = self.weights.get(key, 0.0)
            dot += w * feature
        return dot + self.bias

    def update(
        self,
        query_meta: dict[str, float],
        case_meta: dict[str, float],
        utility: float,
        lr: float = 0.01,
    ) -> None:
        """Update weights with online gradient step."""
        pred = self.score(query_meta, case_meta)
        error = pred - utility
        for key in set(query_meta.keys()) | set(case_meta.keys()):
            q = query_meta.get(key, 0.0)
            c = case_meta.get(key, 0.0)
            feature = q * c
            self.weights[key] = self.weights.get(key, 0.0) - lr * error * feature
        self.bias -= lr * error
        self.n_training_samples += 1
