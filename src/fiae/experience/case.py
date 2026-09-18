"""Case schema and failure taxonomy for the Experience Store (doc 06).

A CaseRecord captures the full context, action, result, cost, and diagnosis
of a single trial or run for future retrieval and meta-learning.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional

from ..contracts import Task, TrialStatus


# -----------------------------------------------------------------------------
# Failure taxonomy (doc 06)
# -----------------------------------------------------------------------------
class FailureTag(str, enum.Enum):
    """Canonical failure/rejection reason codes."""

    LEAKAGE_CONFIRMED = "LEAKAGE_CONFIRMED"
    LEAKAGE_SUSPECTED = "LEAKAGE_SUSPECTED"
    VALIDATION_INVALID = "VALIDATION_INVALID"
    TARGET_ENCODER_NOT_CROSSFIT = "TARGET_ENCODER_NOT_CROSSFIT"
    TEMPORAL_LOOKAHEAD = "TEMPORAL_LOOKAHEAD"
    GROUP_CONTAMINATION = "GROUP_CONTAMINATION"
    OVERFIT_HIGH_VARIANCE = "OVERFIT_HIGH_VARIANCE"
    UNDERFIT_HIGH_BIAS = "UNDERFIT_HIGH_BIAS"
    NO_INCREMENTAL_GAIN = "NO_INCREMENTAL_GAIN"
    REDUNDANT = "REDUNDANT"
    UNSTABLE_ACROSS_FOLDS = "UNSTABLE_ACROSS_FOLDS"
    RESOURCE_RAM = "RESOURCE_RAM"
    RESOURCE_TIME = "RESOURCE_TIME"
    RESOURCE_LATENCY = "RESOURCE_LATENCY"
    RESOURCE_MODEL_SIZE = "RESOURCE_MODEL_SIZE"
    INVALID_DOMAIN = "INVALID_DOMAIN"
    NUMERIC_OVERFLOW = "NUMERIC_OVERFLOW"
    TOO_MANY_LEVELS = "TOO_MANY_LEVELS"
    ENSEMBLE_REDUNDANT = "ENSEMBLE_REDUNDANT"
    ENSEMBLE_NEGATIVE_GAIN = "ENSEMBLE_NEGATIVE_GAIN"
    HPO_NONCONVERGENT = "HPO_NONCONVERGENT"
    GRADIENT_EXPLODING = "GRADIENT_EXPLODING"
    GRADIENT_VANISHING = "GRADIENT_VANISHING"
    DATA_SHIFT = "DATA_SHIFT"
    CODEGEN_MISMATCH = "CODEGEN_MISMATCH"
    SERIALIZATION_FAILURE = "SERIALIZATION_FAILURE"


# -----------------------------------------------------------------------------
# Case components
# -----------------------------------------------------------------------------
@dataclass
class CaseContext:
    """Context in which the trial/run was executed."""

    task: Task = Task.AUTO
    target_semantics: Optional[str] = None
    prediction_time_semantics: Optional[str] = None
    dataset_meta_features: dict[str, Any] = field(default_factory=dict)
    group_structure: Optional[str] = None
    time_structure: Optional[str] = None
    data_quality_profile: dict[str, Any] = field(default_factory=dict)
    user_constraints: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseAction:
    """What was attempted: validation plan, feature graph, model, etc."""

    validation_plan_hash: Optional[str] = None
    feature_graph_hash: Optional[str] = None
    feature_portfolio: list[str] = field(default_factory=list)
    model_family: Optional[str] = None
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    ensemble_plan: Optional[str] = None
    fidelity_stage: str = "probe"
    resource_budget: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseResult:
    """Outcome of the trial/run."""

    primary_metric_name: Optional[str] = None
    primary_metric_value: Optional[float] = None
    secondary_metrics: dict[str, float] = field(default_factory=dict)
    fold_mean: Optional[float] = None
    fold_std: Optional[float] = None
    generalization_gap: Optional[float] = None
    calibration_result: Optional[str] = None
    robustness_score: Optional[float] = None


@dataclass
class CaseCost:
    """Resource consumption."""

    wall_time_s: Optional[float] = None
    cpu_time_s: Optional[float] = None
    peak_ram_mb: Optional[float] = None
    model_bytes: Optional[int] = None
    materialization_bytes: Optional[int] = None
    inference_latency_ms: Optional[float] = None
    inference_batch_size: Optional[int] = None


@dataclass
class CaseDiagnosis:
    """Success/failure diagnosis."""

    status: TrialStatus = TrialStatus.COMPLETED
    success_tags: list[str] = field(default_factory=list)
    failure_tags: list[FailureTag] = field(default_factory=list)
    leakage_findings: list[str] = field(default_factory=list)
    instability_notes: Optional[str] = None
    evidence_refs: list[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Full case record
# -----------------------------------------------------------------------------
@dataclass
class CaseRecord:
    """Immutable record of a trial/run for experience retrieval.

    Identity is derived from dataset + split + feature + model + hyperparameters + seed.
    """

    case_id: str
    dataset_fingerprint: str
    schema_fingerprint: str
    engine_commit: str = "dev"
    context: CaseContext = field(default_factory=CaseContext)
    action: CaseAction = field(default_factory=CaseAction)
    result: CaseResult = field(default_factory=CaseResult)
    cost: CaseCost = field(default_factory=CaseCost)
    diagnosis: CaseDiagnosis = field(default_factory=CaseDiagnosis)
    split_fingerprint: Optional[str] = None
    seed: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_success(self) -> bool:
        """Return True if the case represents a successful outcome."""
        return (
            self.diagnosis.status == TrialStatus.COMPLETED
            and not self.diagnosis.failure_tags
        )

    def identity_hash(self) -> str:
        """Deterministic identity for deduplication (doc 06)."""
        from ..ids import content_hash

        material = {
            "dataset_fingerprint": self.dataset_fingerprint,
            "split_fingerprint": self.split_fingerprint,
            "feature_graph_hash": self.action.feature_graph_hash,
            "model_family": self.action.model_family,
            "hyperparameters": self.action.hyperparameters,
            "seed": self.seed,
        }
        return "case_" + content_hash(material)
