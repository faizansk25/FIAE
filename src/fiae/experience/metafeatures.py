"""Dataset meta-feature extraction for experience retrieval (doc 06).

Meta-features characterize a dataset for similarity-based retrieval.
They are extracted from DatasetProfile and normalized for distance computation.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any

from ..contracts import DatasetProfile, SemanticType


class MetaFeatureFamily(str, enum.Enum):
    """Families of meta-features for weighted distance computation."""

    SHAPE = "shape"
    TYPE_COMPOSITION = "type_composition"
    MISSINGNESS = "missingness"
    NUMERIC_DISTRIBUTION = "numeric_distribution"
    CATEGORICAL_STRUCTURE = "categorical_structure"
    TARGET = "target"
    RELATIONSHIP = "relationship"
    RUNTIME = "runtime"


@dataclass
class DatasetMetaFeatures:
    """Normalized meta-feature vector for a dataset.

    Each family maps to a dict of feature_name -> float value.
    All values are normalized to roughly [0, 1] for distance computation.
    """

    dataset_fingerprint: str
    families: dict[MetaFeatureFamily, dict[str, float]] = field(default_factory=dict)
    raw_stats: dict[str, Any] = field(default_factory=dict)

    def family_vector(self, family: MetaFeatureFamily) -> dict[str, float]:
        """Return the feature dict for a specific family."""
        return self.families.get(family, {})

    def to_flat(self) -> dict[str, float]:
        """Flatten all families into a single dict with family-prefixed keys."""
        flat: dict[str, float] = {}
        for fam, feats in self.families.items():
            for k, v in feats.items():
                flat[f"{fam.value}.{k}"] = v
        return flat


def _safe_log(x: float) -> float:
    """Safe logarithm that handles zero and negative inputs."""
    if x <= 0:
        return 0.0
    return math.log1p(x)


def _normalize_ratio(numerator: float, denominator: float) -> float:
    """Normalize a ratio to [0, 1], handling division by zero."""
    if denominator <= 0:
        return 0.0
    return min(numerator / denominator, 1.0)


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clip value to [lo, hi]."""
    return max(lo, min(hi, value))


# -----------------------------------------------------------------------------
# Extraction
# -----------------------------------------------------------------------------
def extract_meta_features(profile: DatasetProfile) -> DatasetMetaFeatures:
    """Extract normalized meta-features from a DatasetProfile.

    Parameters
    ----------
    profile:
        The dataset profile produced by the intake profiler.

    Returns
    -------
    DatasetMetaFeatures
        Normalized meta-feature vector ready for similarity computation.
    """
    families: dict[MetaFeatureFamily, dict[str, float]] = {}
    raw_stats: dict[str, Any] = {}

    # --- Shape family ---
    n_rows = max(profile.rows_observed or 0, 1)
    n_cols = max(len(profile.columns), 1)
    families[MetaFeatureFamily.SHAPE] = {
        "log_rows": _safe_log(n_rows) / 25.0,
        "log_columns": _safe_log(n_cols) / 10.0,
        "row_col_ratio": _clip(n_rows / (n_cols * 1000)),
        "observed_cells": _safe_log(n_rows * n_cols) / 30.0,
    }
    raw_stats["rows_observed"] = profile.rows_observed
    raw_stats["n_columns"] = len(profile.columns)

    # --- Type composition family ---
    type_counts: dict[str, int] = {}
    for col in profile.columns:
        st = col.semantic_type.value
        type_counts[st] = type_counts.get(st, 0) + 1
    families[MetaFeatureFamily.TYPE_COMPOSITION] = {
        "numeric_fraction": _normalize_ratio(
            type_counts.get(SemanticType.CONTINUOUS_NUMERIC.value, 0), n_cols
        ),
        "integer_fraction": _normalize_ratio(
            type_counts.get(SemanticType.COUNT.value, 0), n_cols
        ),
        "categorical_fraction": _normalize_ratio(
            type_counts.get(SemanticType.LOW_CARDINALITY_CATEGORICAL.value, 0)
            + type_counts.get(SemanticType.HIGH_CARDINALITY_CATEGORICAL.value, 0),
            n_cols,
        ),
        "boolean_fraction": _normalize_ratio(
            type_counts.get(SemanticType.BOOLEAN.value, 0), n_cols
        ),
        "datetime_fraction": _normalize_ratio(
            type_counts.get(SemanticType.DATETIME.value, 0), n_cols
        ),
        "text_fraction": _normalize_ratio(
            type_counts.get(SemanticType.FREE_TEXT.value, 0), n_cols
        ),
        "identifier_fraction": _normalize_ratio(
            type_counts.get(SemanticType.IDENTIFIER.value, 0), n_cols
        ),
        "high_cardinality_fraction": _normalize_ratio(
            type_counts.get(SemanticType.HIGH_CARDINALITY_CATEGORICAL.value, 0), n_cols
        ),
    }

    # --- Missingness family ---
    null_fractions = [col.null_fraction for col in profile.columns if col.null_fraction is not None]
    if null_fractions:
        mean_null = sum(null_fractions) / len(null_fractions)
        max_null = max(null_fractions)
    else:
        mean_null = 0.0
        max_null = 0.0
    n_null = max(len(null_fractions), 1)
    families[MetaFeatureFamily.MISSINGNESS] = {
        "global_mean": _clip(mean_null),
        "global_max": _clip(max_null),
        "high_missing_fraction": _clip(
            sum(1 for f in null_fractions if f > 0.5) / n_null
        ),
        "above_10pct": _clip(sum(1 for f in null_fractions if f > 0.1) / n_null),
        "above_30pct": _clip(sum(1 for f in null_fractions if f > 0.3) / n_null),
        "above_50pct": _clip(sum(1 for f in null_fractions if f > 0.5) / n_null),
        "above_90pct": _clip(sum(1 for f in null_fractions if f > 0.9) / n_null),
    }

    # --- Numeric distribution family ---
    numeric_cols = [
        col for col in profile.columns
        if col.semantic_type in (SemanticType.CONTINUOUS_NUMERIC, SemanticType.COUNT)
        and col.statistics
        and "numeric" in col.statistics
    ]
    if numeric_cols:
        skew_values = []
        kurt_values = []
        zero_fractions = []
        neg_fractions = []
        for col in numeric_cols:
            num_stats = col.statistics.get("numeric", {})
            skew_values.append(abs(num_stats.get("skew", 0.0)))
            kurt_values.append(num_stats.get("kurtosis", 0.0))
            zero_fractions.append(num_stats.get("zero_fraction", 0.0))
            neg_fractions.append(num_stats.get("negative_fraction", 0.0))
        n_num = len(numeric_cols)
        families[MetaFeatureFamily.NUMERIC_DISTRIBUTION] = {
            "mean_abs_skew": _clip(sum(skew_values) / n_num / 5.0),
            "mean_kurtosis": _clip(sum(kurt_values) / n_num / 10.0),
            "mean_zero_fraction": _clip(sum(zero_fractions) / n_num),
            "mean_negative_fraction": _clip(sum(neg_fractions) / n_num),
            "numeric_coverage": _normalize_ratio(n_num, n_cols),
        }
    else:
        families[MetaFeatureFamily.NUMERIC_DISTRIBUTION] = {
            "mean_abs_skew": 0.0,
            "mean_kurtosis": 0.0,
            "mean_zero_fraction": 0.0,
            "mean_negative_fraction": 0.0,
            "numeric_coverage": 0.0,
        }

    # --- Categorical structure family ---
    cat_cols = [
        col for col in profile.columns
        if col.semantic_type in (
            SemanticType.LOW_CARDINALITY_CATEGORICAL,
            SemanticType.HIGH_CARDINALITY_CATEGORICAL,
        )
    ]
    if cat_cols:
        cardinalities = [col.distinct_estimate for col in cat_cols if col.distinct_estimate]
        if cardinalities:
            mean_card = sum(cardinalities) / len(cardinalities)
            max_card = max(cardinalities)
        else:
            mean_card = 0.0
            max_card = 0.0
        rare_fraction = sum(
            1 for col in cat_cols
            if col.statistics and col.statistics.get("semantic_evidence", {}).get("rare_fraction", 0) > 0.1
        ) / max(len(cat_cols), 1)
        families[MetaFeatureFamily.CATEGORICAL_STRUCTURE] = {
            "mean_log_cardinality": _safe_log(mean_card) / 10.0,
            "max_log_cardinality": _safe_log(max_card) / 15.0,
            "rare_level_fraction": _clip(rare_fraction),
            "categorical_coverage": _normalize_ratio(len(cat_cols), n_cols),
        }
    else:
        families[MetaFeatureFamily.CATEGORICAL_STRUCTURE] = {
            "mean_log_cardinality": 0.0,
            "max_log_cardinality": 0.0,
            "rare_level_fraction": 0.0,
            "categorical_coverage": 0.0,
        }

    # --- Target family ---
    target_meta = getattr(profile, "meta_features", {}) or {}
    families[MetaFeatureFamily.TARGET] = {
        "imbalance_ratio": _clip(target_meta.get("imbalance_ratio", 0.5)),
        "target_entropy": _clip(target_meta.get("target_entropy", 0.0)),
        "is_imbalanced": 1.0 if target_meta.get("is_imbalanced", False) else 0.0,
    }

    # --- Relationship family ---
    families[MetaFeatureFamily.RELATIONSHIP] = {
        "duplicate_column_ratio": _normalize_ratio(
            sum(1 for col in profile.columns if any("duplicate_of" in w for w in col.warnings)),
            n_cols,
        ),
    }

    # --- Runtime family ---
    source_cost = profile.source_cost or {}
    families[MetaFeatureFamily.RUNTIME] = {
        "log_elapsed": _safe_log(source_cost.get("elapsed_s", 1.0)) / 10.0,
        "estimated_bytes": _safe_log(source_cost.get("estimated_bytes", 1)) / 30.0,
    }

    return DatasetMetaFeatures(
        dataset_fingerprint=profile.dataset_fingerprint,
        families=families,
        raw_stats=raw_stats,
    )


# -----------------------------------------------------------------------------
# Distance computation
# -----------------------------------------------------------------------------
def meta_feature_distance(
    a: DatasetMetaFeatures,
    b: DatasetMetaFeatures,
    family_weights: dict[MetaFeatureFamily, float] | None = None,
) -> float:
    """Compute weighted distance between two meta-feature vectors.

    Returns a value in [0, 1] where 0 means identical and 1 means maximally different.
    """
    if family_weights is None:
        family_weights = dict.fromkeys(MetaFeatureFamily, 1.0)

    total_weight = 0.0
    weighted_distance = 0.0

    all_families = set(a.families.keys()) | set(b.families.keys())

    for family in all_families:
        weight = family_weights.get(family, 1.0)
        vec_a = a.family_vector(family)
        vec_b = b.family_vector(family)

        # Compute Euclidean distance between the two feature dicts
        all_keys = set(vec_a.keys()) | set(vec_b.keys())
        if not all_keys:
            continue

        sum_sq = 0.0
        for key in all_keys:
            val_a = vec_a.get(key, 0.0)
            val_b = vec_b.get(key, 0.0)
            sum_sq += (val_a - val_b) ** 2

        # Normalize by number of dimensions to keep distance in [0, 1]
        dim = len(all_keys)
        family_dist = (sum_sq / dim) ** 0.5 if dim > 0 else 0.0
        weighted_distance += weight * family_dist
        total_weight += weight

    if total_weight == 0:
        return 0.0
    return min(weighted_distance / total_weight, 1.0)
