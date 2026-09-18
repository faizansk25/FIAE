import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fiae.errors import (
    ErrorCode,
    FIAEError,
    RetryPolicy,
    Severity,
    require,
)


def test_error_taxonomy_complete():
    expected = {
        "DATA_FORMAT_ERROR",
        "SCHEMA_AMBIGUITY",
        "SOURCE_CHANGED_DURING_RUN",
        "TARGET_MISSING",
        "TARGET_INVALID",
        "TASK_AMBIGUOUS",
        "SPLIT_INVALID",
        "LEAKAGE_CONFIRMED",
        "PREDICTION_TIME_UNAVAILABLE",
        "FEATURE_PRECONDITION_FAILED",
        "FEATURE_DOMAIN_ERROR",
        "FEATURE_CARDINALITY_EXPLOSION",
        "RESOURCE_PRECHECK_FAILED",
        "TRIAL_TIMEOUT",
        "TRIAL_OOM",
        "MODEL_FIT_FAILED",
        "METRIC_UNDEFINED",
        "HPO_INVALID",
        "ENSEMBLE_INVALID",
        "CALIBRATION_INVALID",
        "CODEGEN_MISMATCH",
        "REPRODUCIBILITY_MISMATCH",
        "ARTIFACT_CORRUPTION",
        "CANCELLED",
    }
    assert {c.value for c in ErrorCode} == expected
    assert len(expected) == 24


def test_fiae_error_carries_all_required_fields():
    err = FIAEError(
        code=ErrorCode.LEAKAGE_CONFIRMED,
        safe_message="Column copies target exactly.",
        component="leakage_detector",
        run_id="run_1",
        stage_id="stage_2",
        evidence={"column": "status_final"},
    )
    d = err.to_dict()
    for key in (
        "code",
        "severity",
        "component",
        "run_id",
        "stage_id",
        "trial_id",
        "feature_id",
        "recoverable",
        "retry_policy",
        "retryable",
        "safe_message",
        "internal_cause_ref",
        "evidence",
    ):
        assert key in d
    # Confirmed leakage is never retryable and not recoverable.
    assert err.retry_policy is RetryPolicy.NONE
    assert not err.retryable
    assert err.severity is Severity.ERROR


def test_transient_failure_is_retryable():
    err = FIAEError(code=ErrorCode.TRIAL_TIMEOUT, safe_message="Trial exceeded timeout.")
    assert err.retryable
    assert err.retry_policy is RetryPolicy.BACKOFF


def test_error_is_an_exception_and_require_fails_closed():
    err = FIAEError(code=ErrorCode.TARGET_MISSING, safe_message="No target column.")
    try:
        require(False, err)
        raise AssertionError("require() must raise")
    except FIAEError as caught:
        assert caught.code is ErrorCode.TARGET_MISSING
    require(True, err)  # no raise
