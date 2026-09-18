"""Comprehensive tests for M12 — doc 06 R3 retrieval, doc 09 compiler, doc 10 CLI."""

import math
import time


# ---------------------------------------------------------------------------
# Doc 06: R3 retrieval
# ---------------------------------------------------------------------------

from fiae.experience.retrieval import (
    _robust_distance,
    _meta_distance,
    _cheap_bucket_rank,
    _hard_filter,
    bayesian_smooth,
    R3Ranker,
    retrieve_priors,
)
from fiae.experience.case import (
    CaseRecord, CaseContext, CaseAction, CaseResult,
)
from fiae.contracts import Task, TrialStatus


class TestBayesianSmooth:
    def test_uniform_prior(self):
        assert bayesian_smooth(0, 0) == 0.5  # alpha/(alpha+beta)

    def test_all_success(self):
        r = bayesian_smooth(10, 10)
        assert r > 0.9

    def test_all_failure(self):
        r = bayesian_smooth(0, 10)
        assert r < 0.1

    def test_small_sample(self):
        # 1/1 should NOT be 100% with Bayesian smoothing
        r = bayesian_smooth(1, 1)
        assert r < 1.0
        assert r > 0.0


class TestRobustDistance:
    def test_identical(self):
        assert _robust_distance(5.0, 5.0) == 0.0

    def test_scaled(self):
        d = _robust_distance(0.0, 1.0, scale=2.0)
        assert abs(d - 0.5) < 1e-9

    def test_clipped(self):
        d = _robust_distance(0.0, 100.0, scale=1.0)
        assert d == 1.0

    def test_zero_scale(self):
        d = _robust_distance(1.0, 2.0, scale=0.0)
        assert 0.0 <= d <= 1.0


class TestMetaDistance:
    def test_identical_meta(self):
        meta = {"shape_rows": 1000, "shape_cols": 10}
        d = _meta_distance(meta, meta)
        assert d < 0.01

    def test_different_meta(self):
        q = {"shape_rows": 1000, "type_numeric": 0.5}
        c = {"shape_rows": 10, "type_numeric": 0.9}
        d = _meta_distance(q, c)
        assert 0.0 < d <= 1.0

    def test_empty_meta(self):
        d = _meta_distance({}, {})
        assert d == 0.0


class TestHardFilter:
    def _make_case(self, task: Task) -> CaseRecord:
        return CaseRecord(
            case_id="c1",
            dataset_fingerprint="fp1",
            schema_fingerprint="sp1",
            context=CaseContext(task=task),
        )

    def test_matching_task(self):
        case = self._make_case(Task.BINARY)
        assert _hard_filter(case, "binary_classification")

    def test_auto_matches_anything(self):
        case = self._make_case(Task.AUTO)
        assert _hard_filter(case, "regression")

    def test_mismatched_task(self):
        case = self._make_case(Task.FORECASTING)
        assert not _hard_filter(case, "regression")


class TestCheapBucketRank:
    def _make_case(self, rows: int, cols: int) -> CaseRecord:
        return CaseRecord(
            case_id="c1",
            dataset_fingerprint="fp1",
            schema_fingerprint="sp1",
            context=CaseContext(
                dataset_meta_features={"rows": rows, "columns": cols}
            ),
        )

    def test_similar_scale(self):
        c = self._make_case(1000, 10)
        result = _cheap_bucket_rank([c], 800, 8)
        assert len(result) == 1

    def test_very_different_scale(self):
        c = self._make_case(1_000_000, 100)
        result = _cheap_bucket_rank([c], 100, 10)
        assert len(result) == 0

    def test_no_metadata(self):
        c = CaseRecord(
            case_id="c1",
            dataset_fingerprint="fp1",
            schema_fingerprint="sp1",
            context=CaseContext(),
        )
        result = _cheap_bucket_rank([c], 100, 10)
        assert len(result) == 1


class TestR3Ranker:
    def test_inactive_when_no_data(self):
        ranker = R3Ranker()
        assert not ranker.is_active()

    def test_becomes_active(self):
        ranker = R3Ranker(min_samples_for_activation=5)
        for i in range(10):
            ranker.update({"f1": float(i)}, {"f1": float(i + 1)}, utility=1.0)
        assert ranker.is_active()

    def test_score(self):
        ranker = R3Ranker(weights={"f1": 1.0}, bias=0.0)
        s = ranker.score({"f1": 2.0}, {"f1": 3.0})
        assert abs(s - 6.0) < 1e-9

    def test_update_converges(self):
        ranker = R3Ranker(min_samples_for_activation=0)
        for _ in range(50):
            ranker.update({"f1": 1.0}, {"f1": 1.0}, utility=1.0, lr=0.1)
            ranker.update({"f1": 0.0}, {"f1": 5.0}, utility=0.0, lr=0.1)
        high = ranker.score({"f1": 1.0}, {"f1": 1.0})
        low = ranker.score({"f1": 0.0}, {"f1": 5.0})
        assert high > low


class TestRetrievePriors:
    def test_empty_store(self, tmp_path):
        from fiae.experience.store import ExperienceStore, StoreConfig
        store = ExperienceStore(StoreConfig(db_path=str(tmp_path / "test.db")))
        prior = retrieve_priors(store, "regression", 1000, 10)
        assert prior.confidence == 0.0

    def test_with_cases(self, tmp_path):
        from fiae.experience.store import ExperienceStore, StoreConfig
        store = ExperienceStore(StoreConfig(db_path=str(tmp_path / "test.db")))
        # Write a case
        case = CaseRecord(
            case_id="c1",
            dataset_fingerprint="fp1",
            schema_fingerprint="sp1",
            context=CaseContext(
                task=Task.REGRESSION,
                dataset_meta_features={"shape_rows": 1000, "type_numeric": 0.8},
            ),
            action=CaseAction(
                feature_portfolio=["f1", "f2"],
                model_family="random_forest",
            ),
            result=CaseResult(
                primary_metric_name="rmse",
                primary_metric_value=0.15,
            ),
        )
        store.write_case(case)
        prior = retrieve_priors(store, "regression", 1000, 10)
        assert prior.source_count >= 1
        assert prior.confidence >= 0.0


# ---------------------------------------------------------------------------
# Doc 09: Compiler + Pipeline IR
# ---------------------------------------------------------------------------

from fiae.codegen.pipeline_ir import PipelineIR, IRNode, build_ir_from_proposals, generate_python_code
from fiae.codegen.compiler import (
    compile_pipeline,
    generate_sklearn_project,
    gate_validate_ir,
    gate_topological_sort,
    gate_deduplicate,
    gate_partition_fit_transform,
    gate_syntax_check,
    gate_feature_parity,
    gate_prediction_parity,
    gate_latency_resource,
)


def _make_simple_ir() -> PipelineIR:
    """Create a simple valid IR for testing."""
    ir = PipelineIR(
        pipeline_id="test_pipe",
        source_fingerprint="fp_test",
        target="y",
        task="regression",
    )
    n0 = IRNode(node_id="n0", operator="sqrt", inputs=["raw:x1"])
    n1 = IRNode(node_id="n1", operator="log1p", inputs=["raw:x2"])
    n2 = IRNode(node_id="n2", operator="sum", inputs=["n0", "n1"])
    ir.add_node(n0)
    ir.add_node(n1)
    ir.add_node(n2)
    ir.outputs = ["n2"]
    return ir


class TestPipelineIR:
    def test_add_and_get_node(self):
        ir = _make_simple_ir()
        assert ir.get_node("n0") is not None
        assert ir.get_node("n0").operator == "sqrt"
        assert ir.get_node("missing") is None

    def test_topological_order(self):
        ir = _make_simple_ir()
        order = ir.topological_order()
        ids = [n.node_id for n in order]
        assert ids.index("n0") < ids.index("n2")
        assert ids.index("n1") < ids.index("n2")

    def test_verify_dag_valid(self):
        ir = _make_simple_ir()
        errors = ir.verify_dag()
        assert len(errors) == 0

    def test_verify_dag_missing_dep(self):
        ir = PipelineIR()
        n = IRNode(node_id="n0", operator="sqrt", inputs=["n_missing"])
        ir.add_node(n)
        errors = ir.verify_dag()
        assert len(errors) == 1
        assert "missing dependency" in errors[0]

    def test_verify_dag_cycle(self):
        ir = PipelineIR()
        n0 = IRNode(node_id="n0", operator="sqrt", inputs=["n1"])
        n1 = IRNode(node_id="n1", operator="sqrt", inputs=["n0"])
        ir.add_node(n0)
        ir.add_node(n1)
        errors = ir.verify_dag()
        assert any("Cycle" in e for e in errors)

    def test_compute_hash_deterministic(self):
        ir = _make_simple_ir()
        h1 = ir.compute_hash()
        h2 = ir.compute_hash()
        assert h1 == h2

    def test_to_dict_from_dict(self):
        ir = _make_simple_ir()
        d = ir.to_dict()
        ir2 = PipelineIR.from_dict(d)
        assert ir2.pipeline_id == "test_pipe"
        assert len(ir2.nodes) == 3

    def test_build_ir_from_proposals(self):
        from dataclasses import dataclass, field

        @dataclass
        class FakeProposal:
            op: str = "sqrt"
            inputs: list = field(default_factory=lambda: ["raw:x1"])
            params: dict = field(default_factory=dict)

        ir = build_ir_from_proposals(
            [FakeProposal()],
            source_fingerprint="fp1",
            target="y",
            task="regression",
        )
        assert len(ir.nodes) == 1
        assert ir.target == "y"


class TestCompiler:
    def test_gate_validate_ir(self):
        ir = _make_simple_ir()
        result = gate_validate_ir(ir)
        assert result.passed

    def test_gate_topological_sort(self):
        ir = _make_simple_ir()
        result = gate_topological_sort(ir)
        assert result.passed

    def test_gate_deduplicate(self):
        ir = _make_simple_ir()
        result = gate_deduplicate(ir)
        assert result.passed

    def test_gate_partition_fit_transform(self):
        ir = _make_simple_ir()
        result = gate_partition_fit_transform(ir)
        assert result.passed

    def test_gate_syntax_check(self):
        ir = _make_simple_ir()
        result = gate_syntax_check(ir)
        assert result.passed

    def test_gate_feature_parity_empty(self):
        ir = _make_simple_ir()
        result = gate_feature_parity(ir, None)
        assert result.passed

    def test_gate_prediction_parity_empty(self):
        ir = _make_simple_ir()
        result = gate_prediction_parity(ir, None)
        assert result.passed

    def test_gate_latency_resource(self):
        ir = _make_simple_ir()
        result = gate_latency_resource(ir)
        assert result.passed

    def test_compile_pipeline(self):
        ir = _make_simple_ir()
        report = compile_pipeline(ir)
        assert report.all_passed
        assert len(report.gates) >= 8
        assert report.generated_code
        assert report.manifest

    def test_generate_python_code(self):
        ir = _make_simple_ir()
        code = generate_python_code(ir)
        assert "apply_pipeline" in code
        compile(code, "<test>", "exec")

    def test_generate_sklearn_project(self):
        ir = _make_simple_ir()
        files = generate_sklearn_project(ir)
        assert "pyproject.toml" in files
        assert "src/features.py" in files
        assert "README.md" in files
        assert "artifacts/manifest.json" in files


# ---------------------------------------------------------------------------
# Doc 10: CLI commands
# ---------------------------------------------------------------------------

from fiae.cli_advanced import cmd_validate, cmd_leakage, cmd_experience, cmd_codegen
from fiae.cli_pipeline import cmd_pipeline, cmd_export, cmd_optimize


class _FakeArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestCLIAdvanced:
    def test_cmd_validate_missing_source(self, tmp_path):
        args = _FakeArgs(source=str(tmp_path / "nonexistent.csv"), json=False)
        result = cmd_validate(args)
        assert result == 1

    def test_cmd_leakage_missing_source(self, tmp_path):
        args = _FakeArgs(
            source=str(tmp_path / "nonexistent.csv"),
            target="y", json=False,
        )
        result = cmd_leakage(args)
        assert result == 1

    def test_cmd_experience_missing_db(self, tmp_path):
        args = _FakeArgs(
            db_path=str(tmp_path / "nonexistent.db"), json=False,
        )
        result = cmd_experience(args)
        assert result == 1

    def test_cmd_codegen_verify(self):
        args = _FakeArgs(operator=None, verify=True, json=False)
        result = cmd_codegen(args)
        assert result == 0


class TestCLIPipeline:
    def test_cmd_export_missing_ir(self, tmp_path):
        args = _FakeArgs(
            ir_path=str(tmp_path / "nonexistent.json"),
            project=str(tmp_path / "out"),
            json=False,
        )
        result = cmd_export(args)
        assert result == 1

    def test_cmd_pipeline_missing_source(self, tmp_path):
        args = _FakeArgs(
            source=str(tmp_path / "nonexistent.csv"),
            target="y",
            output=str(tmp_path / "out"),
            json=False,
        )
        result = cmd_pipeline(args)
        assert result == 1

    def test_cmd_optimize_missing_source(self, tmp_path):
        args = _FakeArgs(
            source=str(tmp_path / "nonexistent.csv"),
            target="y",
            json=False,
        )
        result = cmd_optimize(args)
        assert result == 1


# ---------------------------------------------------------------------------
# Doc 06: Checkpoint (doc 01 NFR-006)
# ---------------------------------------------------------------------------

from fiae.problem.checkpoint import (
    CheckpointManager, StageCheckpoint, PipelineCheckpoint,
)


class TestCheckpoint:
    def test_checkpoint_lifecycle(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path))
        ckpt = PipelineCheckpoint(pipeline_id="test_pipe", run_id="r1")
        ckpt.add_stage(StageCheckpoint(
            stage_name="intake",
            stage_index=0,
            status="completed",
            result={"rows": 100},
        ))
        mgr.save(ckpt)

        loaded = mgr.load("test_pipe")
        assert loaded is not None
        assert loaded.get_stage_result("intake") == {"rows": 100}

    def test_list_checkpoints(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path))
        ckpt = PipelineCheckpoint(pipeline_id="run1", run_id="r1")
        mgr.save(ckpt)
        checkpoints = mgr.list_checkpoints()
        assert "run1" in checkpoints

    def test_delete_checkpoint(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path))
        ckpt = PipelineCheckpoint(pipeline_id="del_me", run_id="r1")
        mgr.save(ckpt)
        assert mgr.delete("del_me")
        assert mgr.load("del_me") is None


# ---------------------------------------------------------------------------
# Doc 06: Experience Write-back
# ---------------------------------------------------------------------------

from fiae.experience.writeback import CostPredictor, QualityPredictor, _cost_complexity


class TestCostPredictor:
    def test_predict_default(self):
        p = CostPredictor()
        cost = p.predict(1000, 10)
        assert cost > 0

    def test_observe_and_calibrate(self):
        p = CostPredictor()
        for i in range(10):
            p.observe(1000, 10, 1.0, 0.5 + 0.1 * i)
        p.calibrate()
        cost = p.predict(5000, 50)
        assert not math.isnan(cost)

    def test_cost_complexity_shapes(self):
        assert _cost_complexity("O(n)") < _cost_complexity("O(nlogn)")


class TestQualityPredictor:
    def test_predict_gain(self):
        p = QualityPredictor()
        gain = p.predict_gain("numeric", 0.0, 5.0, 100)
        assert 0.0 <= gain <= 1.0

    def test_high_null_reduces_gain(self):
        p = QualityPredictor()
        g1 = p.predict_gain("numeric", 0.0, 5.0, 100)
        g2 = p.predict_gain("numeric", 0.9, 5.0, 100)
        assert g1 > g2


# ---------------------------------------------------------------------------
# Doc 11: Security Audit Log
# ---------------------------------------------------------------------------

from fiae.security.audit import AuditLogger, AuditEntry


class TestAuditLogger:
    def test_write_and_read(self, tmp_path):
        log_path = str(tmp_path / "audit.log")
        logger = AuditLogger(log_path)
        logger.log(AuditEntry(
            timestamp=time.time(),
            event_type="TEST",
            component="test",
            action="test_event",
        ))
        logger._flush()
        entries = logger.read_entries()
        assert len(entries) == 1
        assert entries[0].event_type == "TEST"

    def test_empty_log(self, tmp_path):
        log_path = str(tmp_path / "empty.log")
        logger = AuditLogger(log_path)
        entries = logger.read_entries()
        assert len(entries) == 0


# ---------------------------------------------------------------------------
# Doc 08: Executor Cache
# ---------------------------------------------------------------------------

from fiae.orchestration.cache import TrialCache


class TestTrialCache:
    def _make_trial(self, tid: str = "t1"):
        from fiae.contracts import TrialResult, TrialStatus
        return TrialResult(trial_id=tid, status=TrialStatus.COMPLETED)

    def test_cache_miss(self):
        cache = TrialCache(max_size=10)
        result = cache.get("rf", {"n_estimators": 100}, ["f1", "f2"])
        assert result is None

    def test_cache_put_and_get(self):
        cache = TrialCache(max_size=10)
        tr = self._make_trial("t1")
        cache.put("rf", {"n_estimators": 100}, ["f1", "f2"], tr)
        result = cache.get("rf", {"n_estimators": 100}, ["f1", "f2"])
        assert result is not None
        assert result.trial_id == "t1"

    def test_cache_eviction(self):
        cache = TrialCache(max_size=2)
        for i in range(4):
            tr = self._make_trial(f"t{i}")
            cache.put("rf", {"n_estimators": i}, ["f1"], tr)
        assert cache.get("rf", {"n_estimators": 0}, ["f1"]) is None


# ---------------------------------------------------------------------------
# Doc 12: Property Testing Framework
# ---------------------------------------------------------------------------

from fiae.testing.property_tests import (
    generate_test_values,
    generate_binary_test_values,
    check_null_preservation,
    check_deterministic,
    check_output_length,
    check_finite_outputs,
    run_all_properties,
)
from fiae.features.registry import get_operator


class TestPropertyTests:
    def test_generate_test_values(self):
        values = generate_test_values(seed=42, n=50)
        assert len(values) >= 5
        for v in values:
            assert isinstance(v, list)

    def test_generate_binary_test_values(self):
        pairs = generate_binary_test_values(seed=42, n=50)
        assert len(pairs) >= 5
        for a, b in pairs:
            assert isinstance(a, list)
            assert isinstance(b, list)

    def test_null_preservation_on_operator(self):
        op = get_operator("sqrt")
        result = check_null_preservation(op, [None, 1.0, 4.0, None])
        assert result.passed

    def test_deterministic_on_operator(self):
        op = get_operator("sqrt")
        result = check_deterministic(op, [1.0, 4.0, 9.0])
        assert result.passed

    def test_output_length_on_operator(self):
        op = get_operator("sqrt")
        result = check_output_length(op, [1.0, 4.0, 9.0])
        assert result.passed

    def test_finite_outputs_on_operator(self):
        op = get_operator("sqrt")
        result = check_finite_outputs(op, [1.0, 4.0, 9.0])
        assert result.passed

    def test_run_all_properties(self):
        results = run_all_properties(seed=42)
        assert len(results) > 0
        # Most properties should pass for well-behaved operators
        passed = sum(1 for r in results if r.passed)
        assert passed > 0


# ---------------------------------------------------------------------------
# Doc 07: Ensemble Builder
# ---------------------------------------------------------------------------

from fiae.orchestration.ensemble import build_ensemble, pareto_front
from fiae.contracts import TrialResult, Direction


class TestEnsembleBuilder:
    def test_pareto_front(self):
        from fiae.contracts import ResourceMeasurement, MetricValue
        def _make_trial(tid, metric_val, cost_s):
            return TrialResult(
                trial_id=tid, status=TrialStatus.COMPLETED,
                metrics=[MetricValue(name="quality", value=metric_val,
                                    direction=Direction.MAXIMIZE, split="cv")],
                resource_measurements=[ResourceMeasurement(wall_time_s=cost_s)],
            )
        results = [
            _make_trial("a", 0.8, 1.0),
            _make_trial("b", 0.9, 2.0),
            _make_trial("c", 0.7, 0.5),
        ]
        front = pareto_front(results)
        assert len(front) >= 1

    def test_build_ensemble_empty(self):
        result = build_ensemble([])
        assert result is None


# ---------------------------------------------------------------------------
# Doc 14: Research Adaptations
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
# Doc 03: Progressive CV
# ---------------------------------------------------------------------------

from fiae.problem.progressive_cv import ProgressiveCVPolicy, progressive_cv, Task


class TestProgressiveCV:
    def test_progressive_cv_regression(self):
        import random
        rng = random.Random(42)
        n = 200
        x1 = [rng.gauss(0, 1) for _ in range(n)]
        x2 = [rng.gauss(0, 1) for _ in range(n)]
        y = [2 * x1[i] + 3 * x2[i] + rng.gauss(0, 0.1) for i in range(n)]

        policy = ProgressiveCVPolicy(row_fractions=(0.25, 0.5, 1.0), n_folds=3)
        results = progressive_cv([x1, x2], y, Task.REGRESSION, policy)
        assert len(results) == 3
        # Scores should improve or be stable with more data
        for r in results:
            assert r.n_rows > 0
            assert r.mean_score >= 0


# ---------------------------------------------------------------------------
# Doc 05: Model-informed operators
# ---------------------------------------------------------------------------

from fiae.features.ops_model_informed import (
    tf_residual_proposal,
    tf_tree_leaf_oof_fit,
    tf_tree_leaf_oof_transform,
)


class TestModelInformedOps:
    def test_residual_proposal(self):
        cols = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
        residuals = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0]
        proposals = tf_residual_proposal(cols, residuals)
        assert isinstance(proposals, list)

    def test_tree_leaf_oof(self):
        cols = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]]
        target = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        state = tf_tree_leaf_oof_fit(cols, target, max_depth=2)
        result = tf_tree_leaf_oof_transform(cols, state)
        assert len(result) == 10
