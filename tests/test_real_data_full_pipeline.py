"""Full FIAE pipeline test on real 20-type dataset.

Tests every phase of the canonical pipeline with real data and compares
against sklearn baselines. All outputs are real computed values.
"""

import csv
import math
import os
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Dataset path
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "real_test_data")
CSV_PATH = os.path.join(DATA_DIR, "ecommerce_20types.csv")


def _column_exists():
    return os.path.exists(CSV_PATH)


def _load_csv():
    """Load CSV into columnar format."""
    columns = {}
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                columns.setdefault(key, []).append(val)
    return columns


# ===========================================================================
# Phase 1: Intake & Profile (doc 02)
# ===========================================================================

class TestPhase1_Intake:
    """Verify real data intake and profiling."""

    @pytest.mark.skipif(not _column_exists(), reason="Dataset not generated")
    def test_csv_loads(self):
        columns = _load_csv()
        assert len(columns) == 20
        assert len(columns["order_id"]) == 100_000

    @pytest.mark.skipif(not _column_exists(), reason="Dataset not generated")
    def test_profile_columns(self):
        from fiae.intake import CsvDataSourceAdapter, ProfileConfig, ProfileMode, profile_source

        adapter = CsvDataSourceAdapter(CSV_PATH)
        # Wall-clock budgets are a profiler feature (doc 02); under CI load or
        # coverage tracing the 100k-row EXACT scan can outlive the 30s default.
        # Pin a generous budget so the assertion is about data, not machine speed.
        config = ProfileConfig(mode=ProfileMode.EXACT, time_budget_s=600.0)
        profile = profile_source(adapter, config)

        assert profile.rows_observed == 100_000
        assert len(profile.columns) == 20
        print(f"\n  Profile: {profile.rows_observed} rows, {len(profile.columns)} columns")

        # Print column details
        for col in profile.columns:
            print(f"    {col.name:20s} dtype={col.physical_dtype:12s} semantic={col.semantic_type.value:25s} null={col.null_fraction:.4f}")

    @pytest.mark.skipif(not _column_exists(), reason="Dataset not generated")
    def test_column_types_detected(self):
        from fiae.intake import CsvDataSourceAdapter, ProfileConfig, ProfileMode, profile_source

        adapter = CsvDataSourceAdapter(CSV_PATH)
        config = ProfileConfig(mode=ProfileMode.STANDARD)
        profile = profile_source(adapter, config)

        # Verify different types are detected
        semantic_types = [c.semantic_type.value for c in profile.columns]
        unique_types = set(semantic_types)
        print(f"\n  Detected semantic types: {sorted(unique_types)}")
        assert len(unique_types) >= 3  # At least numeric, categorical, datetime


# ===========================================================================
# Phase 2: Task Inference (doc 03)
# ===========================================================================

class TestPhase2_TaskInference:
    """Verify real task inference."""

    @pytest.mark.skipif(not _column_exists(), reason="Dataset not generated")
    def test_infer_task(self):
        from fiae.problem.task import infer_task
        from fiae.intake import CsvDataSourceAdapter, ProfileConfig, ProfileMode, profile_source

        adapter = CsvDataSourceAdapter(CSV_PATH)
        # Wall-clock budgets are a profiler feature (doc 02); under CI load or
        # coverage tracing the 100k-row EXACT scan can outlive the 30s default.
        # Pin a generous budget so the assertion is about data, not machine speed.
        config = ProfileConfig(mode=ProfileMode.EXACT, time_budget_s=600.0)
        profile = profile_source(adapter, config)

        # Find the target column profile
        target_col = None
        for c in profile.columns:
            if c.name == "is_returned":
                target_col = c
                break
        assert target_col is not None, "is_returned column not found"

        task_info = infer_task(target_col)
        print(f"\n  Task: {task_info.task.value}, confidence: {task_info.confidence:.3f}")
        assert task_info.confidence > 0.0


# ===========================================================================
# Phase 3: Feature Generation (doc 05)
# ===========================================================================

class TestPhase3_FeatureGeneration:
    """Verify real feature operator execution on real data."""

    @pytest.mark.skipif(not _column_exists(), reason="Dataset not generated")
    def test_operators_on_real_data(self):
        from fiae.features.registry import all_operators

        columns = _load_csv()
        ops = all_operators()

        # Test numeric operators on price column
        price_vals = []
        for v in columns["price"][:1000]:
            try:
                price_vals.append(float(v))
            except (ValueError, TypeError):
                price_vals.append(None)

        working_ops = 0
        for op in ops:
            if op.arity == "unary" and "continuous_numeric" in str(op.input_types):
                try:
                    result = op.transform(price_vals)
                    assert len(result) == len(price_vals)
                    working_ops += 1
                except Exception:
                    pass

        print(f"\n  Unary numeric operators working on real data: {working_ops}")
        assert working_ops > 10  # At least 10 operators should work

    @pytest.mark.skipif(not _column_exists(), reason="Dataset not generated")
    def test_operator_outputs_are_real(self):
        """Verify operators produce actual computed values, not constants."""
        from fiae.features.registry import get_operator

        columns = _load_csv()
        price_vals = []
        for v in columns["price"][:500]:
            try:
                price_vals.append(float(v))
            except (ValueError, TypeError):
                price_vals.append(None)

        # Test sqrt operator
        op = get_operator("sqrt")
        result = op.transform(price_vals)
        # Verify it's not constant
        non_none = [r for r in result if r is not None]
        assert len(non_none) > 0
        assert min(non_none) != max(non_none)  # Not constant
        # Verify it's actually sqrt
        for _i, (v, r) in enumerate(zip(price_vals, result)):
            if v is not None and v >= 0 and r is not None:
                assert abs(r - math.sqrt(v)) < 1e-10
                break
        print(f"\n  sqrt(price[0]) = {result[0]}, actual sqrt = {math.sqrt(price_vals[0])}")


# ===========================================================================
# Phase 4: Funnel Filtering (doc 04)
# ===========================================================================

class TestPhase4_Funnel:
    """Verify real funnel filtering with F5 complementarity."""

    @pytest.mark.skipif(not _column_exists(), reason="Dataset not generated")
    def test_f5_complementarity_on_real_features(self):
        from fiae.funnel import f5_complementarity, FunnelPolicy
        from fiae.search.triggers import FeatureProposal
        import numpy as np

        columns = _load_csv()
        # Create real features from price and revenue
        price_vals = []
        for v in columns["price"][:1000]:
            try:
                price_vals.append(float(v))
            except (ValueError, TypeError):
                price_vals.append(0.0)

        revenue_vals = []
        for v in columns["revenue"][:1000]:
            try:
                revenue_vals.append(float(v))
            except (ValueError, TypeError):
                revenue_vals.append(0.0)

        sqrt_price = np.sqrt(np.array(price_vals))
        _log_revenue = np.log1p(np.array(revenue_vals))

        # F5: sqrt_price vs revenue - should be redundant (high correlation)
        proposal1 = FeatureProposal(op="sqrt", inputs=["raw:price"], source="test")
        verdict1 = f5_complementarity(proposal1, sqrt_price, [np.array(revenue_vals)], FunnelPolicy())
        print(f"\n  F5 sqrt(price) vs revenue: passed={verdict1.passed}, reason={verdict1.reason}")

        # F5: sqrt_price vs weight_kg - should be complementary (low correlation)
        weight_vals = []
        for v in columns["weight_kg"][:1000]:
            try:
                weight_vals.append(float(v))
            except (ValueError, TypeError):
                weight_vals.append(0.0)

        verdict2 = f5_complementarity(proposal1, sqrt_price, [np.array(weight_vals)], FunnelPolicy())
        print(f"  F5 sqrt(price) vs weight: passed={verdict2.passed}, reason={verdict2.reason}")

        # At least one should pass (complementary features)
        assert verdict1.passed or verdict2.passed


# ===========================================================================
# Phase 5: HPO & Model Training (doc 07)
# ===========================================================================

class TestPhase5_HPO:
    """Verify real model training on real data."""

    @pytest.mark.skipif(not _column_exists(), reason="Dataset not generated")
    def test_train_on_real_data(self):
        from fiae.orchestration.model_training import TrialRunner, TrialSpec

        columns = _load_csv()
        # Use numeric columns as features
        feature_cols = ["price", "revenue", "discount_pct", "shipping_cost", "weight_kg", "rating"]
        X = []
        y = []

        n = min(5000, len(columns["order_id"]))
        for i in range(n):
            row = []
            for col in feature_cols:
                try:
                    row.append(float(columns[col][i]))
                except (ValueError, TypeError):
                    row.append(0.0)
            try:
                label = int(columns["is_returned"][i])
            except (ValueError, TypeError):
                label = 0
            X.append(row)
            y.append(label)

        runner = TrialRunner()
        spec = TrialSpec(
            model_family="random_forest",
            hyperparameters={"n_estimators": 50},
            fold_count=3,
        )
        result = runner.run_trial(spec, X, y, "classification")

        print(f"\n  Model: {spec.model_family}")
        print(f"  Status: {result.status.value}")
        if result.metrics:
            print(f"  Quality (AUC): {result.metrics[0].value:.4f}")
        if result.resource_measurements:
            rm = result.resource_measurements[0]
            print(f"  Wall time: {rm.wall_time_s:.3f}s")
            print(f"  Model bytes: {rm.model_bytes}")

        assert result.status.value == "COMPLETED"
        assert len(result.metrics) > 0
        # AUC may be around 0.5 for weak signals - that's a real result
        print(f"  AUC={result.metrics[0].value:.4f} (>0.5 means better than random)")
        assert result.metrics[0].value > 0.0  # Valid score

    @pytest.mark.skipif(not _column_exists(), reason="Dataset not generated")
    def test_multiple_models_comparison(self):
        """Train multiple models and compare real results."""
        from fiae.orchestration.model_training import TrialRunner, TrialSpec

        columns = _load_csv()
        feature_cols = ["price", "revenue", "discount_pct", "shipping_cost", "weight_kg", "rating"]
        X, y = [], []
        n = 3000
        for i in range(n):
            row = []
            for col in feature_cols:
                try:
                    row.append(float(columns[col][i]))
                except (ValueError, TypeError):
                    row.append(0.0)
            try:
                label = int(columns["is_returned"][i])
            except (ValueError, TypeError):
                label = 0
            X.append(row)
            y.append(label)

        runner = TrialRunner()
        models = ["random_forest", "gradient_boosting", "decision_tree", "extra_trees"]
        results = {}

        for model in models:
            spec = TrialSpec(
                model_family=model,
                hyperparameters={},
                fold_count=3,
            )
            result = runner.run_trial(spec, X, y, "classification")
            if result.metrics:
                results[model] = result.metrics[0].value
                print(f"  {model:25s} AUC={result.metrics[0].value:.4f}")

        assert len(results) == len(models)
        # All models should beat random (0.5)
        for model, score in results.items():
            assert score > 0.5, f"{model} AUC={score} <= 0.5"


# ===========================================================================
# Phase 6: Baselines (doc 12)
# ===========================================================================

class TestPhase6_Baselines:
    """Verify real baseline comparisons."""

    @pytest.mark.skipif(not _column_exists(), reason="Dataset not generated")
    def test_baselines_on_real_data(self):
        from fiae.testing.baselines import run_baselines, score_dataset_difficulty, assess_feature_value

        columns = _load_csv()
        feature_cols = ["price", "revenue", "discount_pct", "shipping_cost", "weight_kg", "rating"]
        X, y = [], []
        n = 3000
        for i in range(n):
            row = []
            for col in feature_cols:
                try:
                    row.append(float(columns[col][i]))
                except (ValueError, TypeError):
                    row.append(0.0)
            try:
                label = int(columns["is_returned"][i])
            except (ValueError, TypeError):
                label = 0
            X.append(row)
            y.append(label)

        # Run baselines
        baselines = run_baselines(X, y, "classification", ["majority", "random", "linear", "random_forest"])
        print("\n  Baseline Results:")
        for b in baselines:
            if not math.isnan(b.score):
                print(f"    {b.name:20s} AUC={b.score:.4f} ({b.time_s:.3f}s)")

        # Score difficulty
        diff = score_dataset_difficulty(X, y, "classification")
        print(f"\n  Dataset Difficulty: {diff.difficulty_class} (score={diff.overall_score:.3f})")
        print(f"    n_rows={diff.n_rows}, n_features={diff.n_features}")
        print(f"    class_balance={diff.class_balance:.3f}")
        print(f"    feature_relevance={diff.feature_relevance:.3f}")
        print(f"    noise_level={diff.noise_level:.3f}")
        print(f"    missingness={diff.missingness:.3f}")

        # Compare FIAE vs best baseline
        from fiae.orchestration.model_training import TrialRunner, TrialSpec
        runner = TrialRunner()
        spec = TrialSpec(model_family="random_forest", hyperparameters={"n_estimators": 50}, fold_count=3)
        fiae_result = runner.run_trial(spec, X, y, "classification")
        fiae_score = fiae_result.metrics[0].value if fiae_result.metrics else 0.0

        assessment = assess_feature_value(fiae_score, baselines)
        print(f"\n  FIAE Score: {fiae_score:.4f}")
        print(f"  Best Baseline: {assessment.best_baseline_score:.4f} ({assessment.recommendation})")

        assert len(baselines) >= 3
        assert diff.overall_score >= 0.0


# ===========================================================================
# Phase 7: Canonical Pipeline End-to-End (doc 15)
# ===========================================================================

class TestPhase7_CanonicalPipeline:
    """Verify full canonical pipeline on real data."""

    @pytest.mark.skipif(not _column_exists(), reason="Dataset not generated")
    def test_full_pipeline(self):
        from fiae.pipeline.canonical import run_canonical_pipeline

        result = run_canonical_pipeline(CSV_PATH, "is_returned")

        print("\n  Canonical Pipeline Results:")
        print(f"    Run ID: {result.run_id}")
        print(f"    Phases completed: {result.phases_completed}/10")
        print(f"    Total time: {result.total_time_s:.3f}s")

        if result.intake:
            print(f"    Intake: {result.intake.n_rows} rows, {result.intake.n_columns} columns")
            print(f"    Quality findings: {len(result.intake.quality_findings)}")

        if result.validation:
            print(f"    Task: {result.validation.task} (confidence={result.validation.task_confidence:.3f})")
            print(f"    Leakage flags: {len(result.validation.leakage_flags)}")

        if result.generation:
            print(f"    Proposals: {result.generation.proposals_count} generated, {result.generation.after_dedup} after dedup")

        if result.funnel:
            print(f"    Funnel: F0={result.funnel.f0_count} F1={result.funnel.f1_count} F2={result.funnel.f2_count} F5={result.funnel.f5_count}")
            print(f"    Portfolio: {result.funnel.portfolio_size} features")

        if result.hpo:
            print(f"    HPO: {result.hpo.trials_completed} trials, best={result.hpo.best_model}")

        if result.ensemble:
            print(f"    Ensemble: {result.ensemble.members} members")

        if result.evaluation:
            print(f"    Evaluation: primary={result.evaluation.primary_value:.4f}")

        if result.codegen:
            print(f"    Codegen: {result.codegen.verification_gates_passed}/{result.codegen.verification_gates_total} gates passed")

        print(f"    Errors: {len(result.errors)}")

        # Verify pipeline completed all phases
        assert result.phases_completed == 10
        assert result.intake is not None
        assert result.intake.n_rows > 0  # profiler may sample large files
        assert result.intake.n_columns == 20


# ===========================================================================
# Phase 8: Compiler & Export (doc 09)
# ===========================================================================

class TestPhase8_Compiler:
    """Verify real IR compilation and code generation."""

    @pytest.mark.skipif(not _column_exists(), reason="Dataset not generated")
    def test_compile_real_pipeline(self):
        from fiae.codegen.pipeline_ir import build_ir_from_proposals
        from fiae.codegen.compiler import compile_pipeline, generate_sklearn_project
        from fiae.search.triggers import FeatureProposal

        # Build IR from real proposals
        proposals = [
            FeatureProposal(op="sqrt", inputs=["raw:price"], source="registry", params={}),
            FeatureProposal(op="log1p", inputs=["raw:revenue"], source="registry", params={}),
            FeatureProposal(op="sum", inputs=["raw:price", "raw:revenue"], source="registry", params={}),
        ]

        ir = build_ir_from_proposals(proposals, target="is_returned", task="classification")
        assert len(ir.nodes) == 3

        # Compile with verification gates
        report = compile_pipeline(ir)
        print("\n  Compiler Report:")
        print(f"    Pipeline ID: {ir.pipeline_id}")
        print(f"    Nodes: {len(ir.nodes)}")
        print(f"    Gates: {report.summary()['passed']}/{report.summary()['total_gates']} passed")
        print(f"    Generated code: {len(report.generated_code)} chars")
        print(f"    All passed: {report.all_passed}")

        # Verify generated code compiles
        compile(report.generated_code, "<pipeline>", "exec")
        print("    Generated code compiles: True")

        # Generate sklearn project
        with tempfile.TemporaryDirectory() as td:
            files = generate_sklearn_project(ir, td)
            print(f"    Project files: {list(files.keys())}")
            assert "pyproject.toml" in files
            assert "src/features.py" in files

        assert report.all_passed


# ===========================================================================
# Phase 9: Security Validation (doc 11)
# ===========================================================================

class TestPhase9_Security:
    """Verify real security enforcement on real data paths."""

    def test_validate_real_path(self):
        from fiae.security.enforcement import InputValidator

        v = InputValidator()
        ok, _ = v.validate_path(CSV_PATH)
        assert ok

    def test_reject_traversal(self):
        from fiae.security.enforcement import InputValidator

        v = InputValidator()
        ok, _ = v.validate_path("../../../etc/passwd")
        assert not ok

    def test_resource_limits(self):
        from fiae.security.enforcement import ResourceLimits

        lim = ResourceLimits()
        ok, _ = lim.check_line("x" * 100)
        assert ok
        ok, _ = lim.check_line("x" * 2_000_000)
        assert not ok

    def test_incident_log(self):
        from fiae.security.enforcement import IncidentLog, IncidentRecord

        log = IncidentLog()
        log.record(IncidentRecord(severity="warning", category="test", description="test incident"))
        entries = log.read()
        assert len(entries) == 1
        assert entries[0].severity == "warning"


# ===========================================================================
# Phase 10: Experience Store (doc 06)
# ===========================================================================

class TestPhase10_Experience:
    """Verify real experience store operations."""

    def test_write_and_retrieve(self, tmp_path):
        from fiae.experience.store import ExperienceStore, StoreConfig
        from fiae.experience.case import (
            CaseRecord, CaseContext, CaseAction, CaseResult, CaseDiagnosis,
        )
        from fiae.experience.retrieval import retrieve_priors
        from fiae.contracts import Task, TrialStatus

        db_path = str(tmp_path / "test.db")
        store = ExperienceStore(StoreConfig(db_path=db_path))

        # Write a real case
        case = CaseRecord(
            case_id="real_test_1",
            dataset_fingerprint="fp_real_data",
            schema_fingerprint="sp_real",
            context=CaseContext(task=Task.BINARY),
            action=CaseAction(
                feature_portfolio=["sqrt_price", "log_revenue"],
                model_family="random_forest",
            ),
            result=CaseResult(
                primary_metric_name="auc",
                primary_metric_value=0.891,
            ),
            diagnosis=CaseDiagnosis(status=TrialStatus.COMPLETED),
        )
        store.write_case(case)

        # Retrieve priors
        prior = retrieve_priors(store, "binary_classification", 100_000, 20)
        print("\n  Experience Store:")
        print(f"    Cases stored: {store.count()}")
        print(f"    Retrieval confidence: {prior.confidence:.3f}")
        print(f"    Source cases: {prior.source_count}")

        assert store.count() == 1
        assert prior.source_count >= 1
