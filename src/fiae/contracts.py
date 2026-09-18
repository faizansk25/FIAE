"""Typed contracts and schemas for all FIAE modules.

Normative source: doc 13 (Core Contracts, Typed Schemas, IDs, Serialization).
Modules communicate through these typed contracts; no undocumented mutable
dictionaries as architecture. Durable records carry schema_version.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional

SCHEMA_VERSION = 1


# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------
class Task(str, enum.Enum):
    AUTO = "auto"
    BINARY = "binary_classification"
    MULTICLASS = "multiclass_classification"
    REGRESSION = "regression"
    FORECASTING = "forecasting"
    RANKING = "ranking"
    UNSUPERVISED = "unsupervised"


class SemanticType(str, enum.Enum):
    CONTINUOUS_NUMERIC = "continuous_numeric"
    COUNT = "count"
    ORDINAL = "ordinal"
    LOW_CARDINALITY_CATEGORICAL = "low_cardinality_categorical"
    HIGH_CARDINALITY_CATEGORICAL = "high_cardinality_categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    FREE_TEXT = "free_text"
    IDENTIFIER = "identifier"
    ENTITY_KEY = "entity_key"
    GEOGRAPHY_CODE = "geography_like_code"
    PERCENTAGE = "percentage_like"
    CURRENCY = "currency_like"
    SENSITIVE_STRING = "url_email_like_sensitive"
    MIXED_UNKNOWN = "mixed_unknown"


class Direction(str, enum.Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class SplitStrategy(str, enum.Enum):
    STRATIFIED_KFOLD = "stratified_kfold"
    KFOLD = "kfold"
    GROUP_KFOLD = "group_kfold"
    TIME_ORDERED = "time_ordered_backtest"
    CUSTOM = "custom"


class LeakageSeverity(str, enum.Enum):
    INFORMATIONAL = "informational"
    REVIEW_REQUIRED = "review_required"
    HARD_REJECT = "hard_reject"


class LeakageClass(str, enum.Enum):
    """Leakage taxonomy (doc 03 / doc 05)."""

    L0 = "L0"  # deterministic row-wise
    L1 = "L1"  # learned unsupervised; fit on training fold
    L2 = "L2"  # target-aware; strict cross-fit
    L3 = "L3"  # temporal/history; availability-aware
    L4 = "L4"  # prediction-time semantic risk
    L5 = "L5"  # confirmed/future leakage; forbidden


class FitScope(str, enum.Enum):
    NONE = "none"
    TRAINING_FOLD = "training_fold"
    DEVELOPMENT = "development"


class TargetPermission(str, enum.Enum):
    """Plugin/operation target-access classes (doc 11)."""

    P0_NONE = "P0_no_target"
    P1_CROSSFIT = "P1_target_through_crossfit_api_only"
    P2_HISTORICAL_CUTOFF = "P2_historical_target_up_to_temporal_cutoff"
    P3_EVALUATOR_ONLY = "P3_evaluator_only"


class TrialStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    PRUNED = "PRUNED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    OOM = "OOM"
    REJECTED = "REJECTED"


# --------------------------------------------------------------------------
# Data contracts
# --------------------------------------------------------------------------
@dataclass
class DataSourceSpec:
    kind: str
    uri_or_path: str
    format: str
    seekable: bool = False
    compression: Optional[str] = None
    credentials_ref: Optional[str] = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ColumnProfile:
    name: str
    physical_dtype: str
    semantic_type: SemanticType
    null_fraction: float
    distinct_estimate: int
    distinct_ratio: float
    statistics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class DatasetProfile:
    dataset_fingerprint: str
    rows_observed: int
    columns: list[ColumnProfile]
    rows_estimated: Optional[int] = None
    meta_features: dict[str, Any] = field(default_factory=dict)
    source_cost: dict[str, Any] = field(default_factory=dict)
    quality_findings: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ProblemDefinition:
    task: Task
    target: Optional[str]
    positive_class: Optional[Any] = None
    prediction_time_semantics: Optional[str] = None
    group_keys: list[str] = field(default_factory=list)
    time_key: Optional[str] = None
    horizon: Optional[str] = None
    metrics: list["MetricValue"] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)


@dataclass
class ConstraintSpec:
    wall_time_s: Optional[float] = None
    memory_bytes: Optional[int] = None
    latency_s: Optional[float] = None
    latency_batch_size: Optional[int] = None
    model_bytes: Optional[int] = None
    interpretability: Optional[str] = None
    quality_target: Optional[float] = None
    worker_limit: Optional[int] = None


@dataclass
class MetricValue:
    name: str
    value: float
    direction: Direction
    split: str
    fold: Optional[int] = None
    aggregation: Optional[str] = None
    unit: Optional[str] = None


@dataclass
class ValidationPlan:
    strategy: SplitStrategy
    folds: int
    seed: int
    split_fingerprint: str
    group_key: Optional[str] = None
    time_key: Optional[str] = None
    cutoffs: list[Any] = field(default_factory=list)
    final_holdout: bool = True
    nested_policy: Optional[str] = None


@dataclass
class LeakageFinding:
    finding_id: str
    subject: str
    severity: LeakageSeverity
    type: LeakageClass
    evidence: dict[str, Any] = field(default_factory=dict)
    action: str = "warn"
    override_allowed: bool = False


@dataclass
class FeatureNode:
    feature_id: str
    operator: str
    inputs: list[str]
    params: dict[str, Any] = field(default_factory=dict)
    fit_scope: FitScope = FitScope.NONE
    target_permission: TargetPermission = TargetPermission.P0_NONE
    time_semantics: Optional[str] = None
    null_policy: str = "preserve"
    cost_hint: dict[str, Any] = field(default_factory=dict)
    lineage_hash: Optional[str] = None


@dataclass
class FittedStateRef:
    state_id: str
    feature_id: str
    split_id: str  # split identity prevents cross-fold state reuse (doc 13)
    artifact_ref: str
    state_hash: str


@dataclass
class FeaturePortfolio:
    portfolio_id: str
    feature_ids: list[str]
    graph_hash: str
    estimated_cost: dict[str, Any] = field(default_factory=dict)
    verified_metrics: list[MetricValue] = field(default_factory=list)


@dataclass
class ModelSpec:
    family: str
    backend: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    capabilities: dict[str, Any] = field(default_factory=dict)
    resource_hints: dict[str, Any] = field(default_factory=dict)


@dataclass
class FidelitySpec:
    row_fraction: float = 1.0
    fold_count: int = 3
    iterations: Optional[int] = None
    feature_fraction: float = 1.0
    stage: str = "probe"


@dataclass
class TrialSpec:
    trial_id: str
    portfolio_id: str
    model_spec: ModelSpec
    fidelity: FidelitySpec
    constraints: Optional[ConstraintSpec] = None
    timeout_s: Optional[float] = None
    resource_reservation: dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None


@dataclass
class ResourceMeasurement:
    wall_time_s: Optional[float] = None
    cpu_time_s: Optional[float] = None
    peak_rss_bytes: Optional[int] = None
    disk_bytes: Optional[int] = None
    model_bytes: Optional[int] = None
    # Latency without batch size / hardware context is incomplete (doc 13).
    inference_seconds: Optional[float] = None
    inference_batch_size: Optional[int] = None
    hardware_context: Optional[str] = None


@dataclass
class TrialResult:
    trial_id: str
    status: TrialStatus
    metrics: list[MetricValue] = field(default_factory=list)
    fold_metrics: list[MetricValue] = field(default_factory=list)
    train_metrics: list[MetricValue] = field(default_factory=list)
    resource_measurements: list[ResourceMeasurement] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    diagnosis: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleSpec:
    member_trial_ids: list[str]
    weights: list[float]
    stacker: Optional[ModelSpec] = None
    oof_refs: list[str] = field(default_factory=list)
    latency_estimate: Optional[float] = None
    size_estimate: Optional[int] = None
    stack_overfit_guard_result: Optional[str] = None


@dataclass
class DecisionRecord:
    decision_id: str
    type: str
    subject: str
    action: str
    reason: str
    evidence_refs: list[str] = field(default_factory=list)
    rule_refs: list[str] = field(default_factory=list)
    metric_constraint_snapshot: dict[str, Any] = field(default_factory=dict)
    override_policy: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class ArtifactRef:
    artifact_id: str
    kind: str
    path_or_uri: str
    hash: str
    bytes: Optional[int] = None
    producer_run: Optional[str] = None
    trust_level: str = "unverified"


@dataclass
class RunManifest:
    run_id: str
    completion_state: str
    source_fingerprint: Optional[str] = None
    config_hash: Optional[str] = None
    engine_build: Optional[str] = None
    environment: dict[str, Any] = field(default_factory=dict)
    selected_pipeline_hash: Optional[str] = None
    artifacts: list[ArtifactRef] = field(default_factory=list)


@dataclass
class PredictionContext:
    """Temporal operators receive this explicitly (doc 13)."""

    decision_timestamp: Optional[str] = None
    observation_cutoff: Optional[str] = None
    horizon: Optional[str] = None
    entity_keys: list[Any] = field(default_factory=list)
    allowed_source_lag: Optional[str] = None


# EventEnvelope is part of the normative contract surface (doc 13); the runtime
# implementation lives in fiae.events and is re-exported here so all modules
# can `from fiae.contracts import EventEnvelope`.
from .events import EventEnvelope  # noqa: E402, F401


