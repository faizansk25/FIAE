"""Error taxonomy and typed error container.

Normative source: doc 01 (error taxonomy) and doc 13 (FIAEError contract).
Every error carries: code, severity, component, run/stage/trial/feature IDs,
recoverable flag, retry policy, human reason, machine-readable evidence.
Domain errors are explicit; they never silently become misleading numeric values.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional


class ErrorCode(str, enum.Enum):
    """The 24 normative error codes from doc 01."""

    DATA_FORMAT_ERROR = "DATA_FORMAT_ERROR"
    SCHEMA_AMBIGUITY = "SCHEMA_AMBIGUITY"
    SOURCE_CHANGED_DURING_RUN = "SOURCE_CHANGED_DURING_RUN"
    TARGET_MISSING = "TARGET_MISSING"
    TARGET_INVALID = "TARGET_INVALID"
    TASK_AMBIGUOUS = "TASK_AMBIGUOUS"
    SPLIT_INVALID = "SPLIT_INVALID"
    LEAKAGE_CONFIRMED = "LEAKAGE_CONFIRMED"
    PREDICTION_TIME_UNAVAILABLE = "PREDICTION_TIME_UNAVAILABLE"
    FEATURE_PRECONDITION_FAILED = "FEATURE_PRECONDITION_FAILED"
    FEATURE_DOMAIN_ERROR = "FEATURE_DOMAIN_ERROR"
    FEATURE_CARDINALITY_EXPLOSION = "FEATURE_CARDINALITY_EXPLOSION"
    RESOURCE_PRECHECK_FAILED = "RESOURCE_PRECHECK_FAILED"
    TRIAL_TIMEOUT = "TRIAL_TIMEOUT"
    TRIAL_OOM = "TRIAL_OOM"
    MODEL_FIT_FAILED = "MODEL_FIT_FAILED"
    METRIC_UNDEFINED = "METRIC_UNDEFINED"
    HPO_INVALID = "HPO_INVALID"
    ENSEMBLE_INVALID = "ENSEMBLE_INVALID"
    CALIBRATION_INVALID = "CALIBRATION_INVALID"
    CODEGEN_MISMATCH = "CODEGEN_MISMATCH"
    REPRODUCIBILITY_MISMATCH = "REPRODUCIBILITY_MISMATCH"
    ARTIFACT_CORRUPTION = "ARTIFACT_CORRUPTION"
    CANCELLED = "CANCELLED"


class Severity(str, enum.Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class RetryPolicy(str, enum.Enum):
    NONE = "NONE"
    IMMEDIATE = "IMMEDIATE"
    BACKOFF = "BACKOFF"


# Default retry policies. Retry only transient failures; never repeat unchanged
# deterministic invalid-domain, confirmed-leakage, or preflight-OOM trials (doc 08).
_DEFAULT_RETRY: dict[ErrorCode, RetryPolicy] = {
    ErrorCode.TRIAL_TIMEOUT: RetryPolicy.BACKOFF,
    ErrorCode.MODEL_FIT_FAILED: RetryPolicy.BACKOFF,
    ErrorCode.SOURCE_CHANGED_DURING_RUN: RetryPolicy.NONE,
    ErrorCode.LEAKAGE_CONFIRMED: RetryPolicy.NONE,
    ErrorCode.FEATURE_DOMAIN_ERROR: RetryPolicy.NONE,
    ErrorCode.RESOURCE_PRECHECK_FAILED: RetryPolicy.NONE,
    ErrorCode.TRIAL_OOM: RetryPolicy.NONE,
}

# Codes that are recoverable by design (partial results / cancellation paths).
_DEFAULT_RECOVERABLE: set[ErrorCode] = {
    ErrorCode.TRIAL_TIMEOUT,
    ErrorCode.TRIAL_OOM,
    ErrorCode.MODEL_FIT_FAILED,
    ErrorCode.CANCELLED,
    ErrorCode.RESOURCE_PRECHECK_FAILED,
    ErrorCode.FEATURE_PRECONDITION_FAILED,
}


@dataclass
class FIAEError(Exception):
    """Typed, durable error record (doc 13 FIAEError + doc 01 requirements)."""

    code: ErrorCode
    safe_message: str
    severity: Severity = Severity.ERROR
    component: str = ""
    run_id: Optional[str] = None
    stage_id: Optional[str] = None
    trial_id: Optional[str] = None
    feature_id: Optional[str] = None
    subject_id: Optional[str] = None
    recoverable: Optional[bool] = None
    retry_policy: Optional[RetryPolicy] = None
    internal_cause_ref: Optional[str] = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(f"[{self.code.value}] {self.safe_message}")
        if self.recoverable is None:
            self.recoverable = self.code in _DEFAULT_RECOVERABLE
        if self.retry_policy is None:
            self.retry_policy = _DEFAULT_RETRY.get(self.code, RetryPolicy.NONE)

    @property
    def category(self) -> str:
        return self.code.value

    @property
    def retryable(self) -> bool:
        return self.retry_policy is not RetryPolicy.NONE

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "component": self.component,
            "run_id": self.run_id,
            "stage_id": self.stage_id,
            "trial_id": self.trial_id,
            "feature_id": self.feature_id,
            "subject_id": self.subject_id,
            "recoverable": self.recoverable,
            "retry_policy": self.retry_policy.value if self.retry_policy else None,
            "retryable": self.retryable,
            "safe_message": self.safe_message,
            "internal_cause_ref": self.internal_cause_ref,
            "evidence": self.evidence,
        }


def require(condition: bool, error: FIAEError) -> None:
    """Fail closed: raise the typed error unless the invariant holds."""
    if not condition:
        raise error
