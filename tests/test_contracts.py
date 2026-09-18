import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fiae import contracts as C


def test_core_contracts_exist():
    for name in (
        "DataSourceSpec",
        "ColumnProfile",
        "DatasetProfile",
        "ProblemDefinition",
        "ConstraintSpec",
        "ValidationPlan",
        "LeakageFinding",
        "FeatureNode",
        "FittedStateRef",
        "FeaturePortfolio",
        "ModelSpec",
        "FidelitySpec",
        "TrialSpec",
        "TrialResult",
        "EnsembleSpec",
        "MetricValue",
        "ResourceMeasurement",
        "DecisionRecord",
        "EventEnvelope",
        "ArtifactRef",
        "RunManifest",
        "PredictionContext",
    ):
        assert hasattr(C, name), f"missing contract: {name}"


def test_leakage_classes_l0_to_l5():
    values = {c.value for c in C.LeakageClass}
    assert values == {"L0", "L1", "L2", "L3", "L4", "L5"}


def test_severity_levels_match_doc_01():
    assert {s.value for s in C.LeakageSeverity} == {
        "informational",
        "review_required",
        "hard_reject",
    }


def test_target_permission_levels_p0_to_p3():
    values = {p.value for p in C.TargetPermission}
    assert len(values) == 4
    assert any(v.startswith("P0") for v in values)
    assert any(v.startswith("P3") for v in values)


def test_trial_spec_roundtrip_via_fiae_ids():
    from fiae.ids import content_hash

    spec = C.TrialSpec(
        trial_id="t1",
        portfolio_id="p1",
        model_spec=C.ModelSpec(family="gradient_boosting", backend="sklearn", seed=42),
        fidelity=C.FidelitySpec(row_fraction=0.16, fold_count=3, stage="screen"),
    )
    h1 = content_hash(spec)
    # Same semantic content, constructed differently, same identity.
    spec2 = C.TrialSpec(
        trial_id="t1",
        portfolio_id="p1",
        fidelity=C.FidelitySpec(stage="screen", fold_count=3, row_fraction=0.16),
        model_spec=C.ModelSpec(backend="sklearn", seed=42, family="gradient_boosting"),
    )
    assert h1 == content_hash(spec2)


def test_resource_measurement_fields_present():
    rm = C.ResourceMeasurement(wall_time_s=1.0, peak_rss_bytes=100, inference_batch_size=64)
    assert rm.inference_batch_size == 64  # latency requires batch size context
    assert rm.hardware_context is None  # incomplete until provided
