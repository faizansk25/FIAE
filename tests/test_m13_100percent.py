"""Comprehensive tests for M13 — doc 07/08 model training + parallel execution,
doc 09 verification, doc 10 CLI, doc 11 enforcement, doc 12 baselines,
doc 14 research, doc 15 canonical pipeline."""

import math
import os
import time


# ---------------------------------------------------------------------------
# Doc 07: Actual model training
# ---------------------------------------------------------------------------

from fiae.orchestration.model_training import (
    TrialSpec, TrialRunner, _get_sklearn_model, _evaluate_cv,
    compute_utility, get_fidelity_ladder,
    LARGE_DATA_LADDER, MEDIUM_DATA_LADDER, SMALL_DATA_LADDER,
)


class TestModelTraining:
    def _make_data(self, n=200):
        import random
        rng = random.Random(42)
        X = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(n)]
        y = [1 if sum(row) > 0 else 0 for row in X]
        return X, y

    def test_get_sklearn_model_rf(self):
        model = _get_sklearn_model("random_forest", {"task": "classification", "n_estimators": 10})
        assert model is not None

    def test_get_sklearn_model_ridge(self):
        model = _get_sklearn_model("ridge", {"task": "regression"})
        assert model is not None

    def test_evaluate_cv(self):
        from sklearn.ensemble import RandomForestClassifier
        X, y = self._make_data()
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        results = _evaluate_cv(model, X, y, "classification", n_folds=3)
        assert "mean" in results
        assert "std" in results

    def test_trial_runner_classification(self):
        X, y = self._make_data()
        runner = TrialRunner()
        spec = TrialSpec(model_family="random_forest", hyperparameters={"n_estimators": 10})
        result = runner.run_trial(spec, X, y, task="classification")
        assert result.status.value == "COMPLETED"
        assert len(result.metrics) >= 1

    def test_trial_runner_regression(self):
        import random
        rng = random.Random(42)
        X = [[rng.gauss(0, 1)] for _ in range(100)]
        y = [row[0] * 2 + rng.gauss(0, 0.1) for row in X]
        runner = TrialRunner()
        spec = TrialSpec(model_family="ridge", hyperparameters={})
        result = runner.run_trial(spec, X, y, task="regression")
        assert result.status.value == "COMPLETED"

    def test_fidelity_ladder(self):
        assert get_fidelity_ladder(200_000) == LARGE_DATA_LADDER
        assert get_fidelity_ladder(50_000) == MEDIUM_DATA_LADDER
        assert get_fidelity_ladder(500) == SMALL_DATA_LADDER

    def test_compute_utility(self):
        u = compute_utility(
            quality=0.9, fold_variance=0.05, generalization_gap=0.02,
            time_s=0.5, memory_mb=10.0,
        )
        assert u < 0.9  # penalties reduce quality
        assert u > 0.0  # still positive


# ---------------------------------------------------------------------------
# Doc 08: Real parallel executor
# ---------------------------------------------------------------------------

from fiae.orchestration.parallel_executor import (
    ParallelExecutor, InlineExecutor, ResourcePool, EventEnvelope, EventStream,
)


class TestResourcePool:
    def test_available_workers(self):
        pool = ResourcePool(max_workers=4)
        assert pool.available_workers == 4

    def test_reserve_and_release(self):
        pool = ResourcePool(max_workers=2)
        token = pool.reserve({"memory_mb": 100})
        assert pool.available_workers == 1
        pool.release(token, 100)
        assert pool.available_workers == 2

    def test_tokens_available(self):
        pool = ResourcePool(max_workers=1, max_memory_mb=1000)
        assert pool.tokens_available({"memory_mb": 500})
        pool.reserve({"memory_mb": 800})
        assert not pool.tokens_available({"memory_mb": 500})  # memory exceeded


class TestParallelExecutor:
    def test_submit_and_poll(self):
        executor = ParallelExecutor(pool=ResourcePool(max_workers=2))
        executor.start()
        token = executor.submit("job1", lambda: 42)
        assert token is not None
        time.sleep(0.2)
        results = executor.poll()
        assert len(results) >= 1
        assert results[0][1] == "completed"
        executor.shutdown()

    def test_backpressure(self):
        pool = ResourcePool(max_workers=1, max_memory_mb=100)
        executor = ParallelExecutor(pool=pool)
        executor.start()
        token1 = executor.submit("j1", lambda: time.sleep(0.5), estimate={"memory_mb": 80})
        assert token1 is not None
        # Second should be rejected due to backpressure
        token2 = executor.submit("j2", lambda: 1, estimate={"memory_mb": 80})
        assert token2 is None
        executor.shutdown()

    def test_cancel(self):
        executor = ParallelExecutor(pool=ResourcePool(max_workers=2))
        executor.start()
        executor.submit("job1", lambda: time.sleep(10))
        time.sleep(0.05)
        executor.cancel("job1")
        executor.shutdown()


class TestInlineExecutor:
    def test_execute_immediately(self):
        executor = InlineExecutor()
        executor.submit("j1", lambda: 42)
        results = executor.poll()
        assert len(results) == 1
        assert results[0][2] == 42


class TestEventStream:
    def test_emit_and_drain(self):
        stream = EventStream(max_buffer=10)
        stream.emit(EventEnvelope(event_type="test", job_id="j1"))
        stream.emit(EventEnvelope(event_type="test", job_id="j2"))
        events = stream.drain()
        assert len(events) == 2

    def test_auto_flush(self):
        stream = EventStream(max_buffer=2)
        stream.emit(EventEnvelope(event_type="a"))
        stream.emit(EventEnvelope(event_type="b"))
        # Buffer is full, next emit triggers flush
        stream.emit(EventEnvelope(event_type="c"))


# ---------------------------------------------------------------------------
# Doc 11: Resource enforcement
# ---------------------------------------------------------------------------

from fiae.security.enforcement import (
    ResourceLimits, InputValidator, IncidentRecord, IncidentLog,
    PluginPermission, validate_plugin_permission,
)


class TestResourceLimits:
    def test_check_line_ok(self):
        lim = ResourceLimits(max_line_bytes=1000)
        ok, _msg = lim.check_line("hello world")
        assert ok

    def test_check_line_too_large(self):
        lim = ResourceLimits(max_line_bytes=10)
        ok, msg = lim.check_line("x" * 100)
        assert not ok
        assert "too large" in msg.lower()

    def test_check_field(self):
        lim = ResourceLimits(max_field_bytes=5)
        ok, _ = lim.check_field("hi")
        assert ok
        ok, _ = lim.check_field("x" * 100)
        assert not ok


class TestInputValidator:
    def test_validate_path_ok(self):
        v = InputValidator()
        ok, _ = v.validate_path("data/train.csv")
        assert ok

    def test_validate_path_traversal(self):
        v = InputValidator()
        ok, _ = v.validate_path("../../etc/passwd")
        assert not ok

    def test_validate_column_name(self):
        v = InputValidator()
        ok, _ = v.validate_column_name("col1")
        assert ok
        ok, _ = v.validate_column_name("")
        assert not ok

    def test_sanitize_string(self):
        v = InputValidator()
        result = v.sanitize_string("hello\x00world")
        assert "\x00" not in result

    def test_verify_dependency(self):
        v = InputValidator()
        ok, _ = v.verify_dependency("scikit-learn")
        assert ok
        ok, _ = v.verify_dependency("unknown-package")
        assert not ok


class TestIncidentLog:
    def test_record_and_read(self, tmp_path):
        log_path = str(tmp_path / "incidents.jsonl")
        log = IncidentLog(log_path)
        inc = IncidentRecord(severity="warning", category="resource", description="test")
        inc_id = log.record(inc)
        assert inc_id
        entries = log.read()
        assert len(entries) == 1
        assert entries[0].severity == "warning"

    def test_append_only(self, tmp_path):
        log = IncidentLog()
        log.record(IncidentRecord(description="a"))
        log.record(IncidentRecord(description="b"))
        log.record(IncidentRecord(description="c"))
        entries = log.read()
        assert len(entries) == 3


class TestPluginPermission:
    def test_valid_plugin(self):
        perm = PluginPermission(name="test_op", target_access="P0")
        violations = validate_plugin_permission(perm)
        assert len(violations) == 0

    def test_invalid_target_access(self):
        perm = PluginPermission(name="test_op", target_access="P5")
        violations = validate_plugin_permission(perm)
        assert len(violations) > 0

    def test_memory_exceeds_limit(self):
        perm = PluginPermission(name="test_op", estimated_memory_mb=9999)
        lim = ResourceLimits(max_ram_mb=1000)
        violations = validate_plugin_permission(perm, lim)
        assert len(violations) > 0


# ---------------------------------------------------------------------------
# Doc 12: Benchmark baselines + difficulty scoring
# ---------------------------------------------------------------------------

from fiae.testing.baselines import (
    run_baselines, score_dataset_difficulty, assess_feature_value,
    BaselineResult, DifficultyProfile,
)


class TestBaselines:
    def _make_data(self, n=100):
        import random
        rng = random.Random(42)
        X = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(n)]
        y = [1 if sum(row) > 0 else 0 for row in X]
        return X, y

    def test_run_baselines(self):
        X, y = self._make_data()
        results = run_baselines(X, y, "classification", baselines=["majority", "random"])
        assert len(results) == 2
        for r in results:
            assert not math.isnan(r.score)

    def test_baselines_include_linear(self):
        X, y = self._make_data()
        results = run_baselines(X, y, "classification")
        names = {r.name for r in results}
        assert "linear" in names


class TestDifficultyScoring:
    def _make_data(self, n=200):
        import random
        rng = random.Random(42)
        X = [[rng.gauss(0, 1) for _ in range(5)] for _ in range(n)]
        y = [1 if sum(row) > 1 else 0 for row in X]
        return X, y

    def test_score_difficulty(self):
        X, y = self._make_data()
        profile = score_dataset_difficulty(X, y, "classification")
        assert 0 <= profile.overall_score <= 1
        assert profile.n_rows == 200
        assert profile.n_features == 5
        assert profile.difficulty_class in ("easy", "medium", "hard", "very_hard")

    def test_regression_difficulty(self):
        import random
        rng = random.Random(42)
        X = [[rng.gauss(0, 1)] for _ in range(100)]
        y = [row[0] * 2 for row in X]
        profile = score_dataset_difficulty(X, y, "regression")
        assert 0 <= profile.overall_score <= 1

    def test_difficulty_profile_to_dict(self):
        p = DifficultyProfile(overall_score=0.5, difficulty_class="medium")
        d = p.to_dict()
        assert d["overall_score"] == 0.5


class TestFeatureValueAssessment:
    def test_provides_value(self):
        baselines = [BaselineResult(name="linear", score=0.7)]
        assessment = assess_feature_value(0.85, baselines)
        assert assessment.provides_value
        assert assessment.improvement_pct > 0

    def test_no_value(self):
        baselines = [BaselineResult(name="rf", score=0.95)]
        assessment = assess_feature_value(0.85, baselines)
        assert not assessment.provides_value


# ---------------------------------------------------------------------------
# Doc 14: Research adaptations
# ---------------------------------------------------------------------------

from fiae.research.adaptations import ADAPTATIONS, get_adaptations_by_status, adaptation_summary


class TestResearchAdaptations:
    def test_adaptations_exist(self):
        assert len(ADAPTATIONS) > 0

    def test_filter_by_status(self):
        implemented = get_adaptations_by_status("implemented")
        assert isinstance(implemented, list)

    def test_summary(self):
        s = adaptation_summary()
        assert "total" in s


# ---------------------------------------------------------------------------
# Doc 15: Canonical pipeline
# ---------------------------------------------------------------------------

from fiae.pipeline.canonical import (
    run_canonical_pipeline, RunContext, CanonicalResult,
    phase_bootstrap, phase_intake,
)


class TestCanonicalPipeline:
    def test_bootstrap(self, tmp_path):
        # Create a dummy source file
        csv_path = str(tmp_path / "test.csv")
        with open(csv_path, "w") as f:
            f.write("x1,x2,y\n1,2,3\n4,5,6\n7,8,9\n")

        ctx = phase_bootstrap(csv_path, "y")
        assert ctx.run_id
        assert ctx.source_fingerprint
        assert os.path.exists(ctx.metadata_store_path)

    def test_intake(self, tmp_path):
        csv_path = str(tmp_path / "test.csv")
        with open(csv_path, "w") as f:
            f.write("x1,x2,y\n1,2,3\n4,5,6\n7,8,9\n")

        ctx = RunContext()
        ctx.run_dir = str(tmp_path)
        intake = phase_intake(ctx, csv_path, "y")
        assert intake.n_rows > 0
        assert intake.n_columns > 0

    def test_canonical_pipeline(self, tmp_path):
        csv_path = str(tmp_path / "test.csv")
        with open(csv_path, "w") as f:
            f.write("x1,x2,y\n")
            for i in range(50):
                f.write(f"{i},{i*2},{i*3}\n")

        result = run_canonical_pipeline(csv_path, "y")
        assert result.phases_completed >= 5  # at least through generation
        assert result.run_id
        summary = result.summary()
        assert "run_id" in summary

    def test_canonical_result_summary(self):
        result = CanonicalResult(run_id="test", phases_completed=10)
        s = result.summary()
        assert s["phases_completed"] == 10
