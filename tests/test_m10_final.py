"""M10 tests: executor, ensemble, security, codegen, CLI (docs 07-11)."""

from __future__ import annotations

import time


from fiae.contracts import (
    Direction, MetricValue, ResourceMeasurement,
    TrialResult, TrialSpec, TrialStatus, ModelSpec, FidelitySpec,
)
from fiae.orchestration.executor import (
    ExecutionToken, TokenPool, SchedulerExecutor, TrialTask,
)
from fiae.orchestration.ensemble import (
    EnsemblePolicy, check_eligibility, compute_diversity,
    optimize_weights, build_ensemble, pareto_front,
)
from fiae.security.sandbox import (
    SandboxPolicy, run_in_sandbox, validate_input_code,
    validate_numeric_input, validate_column_name,
)
from fiae.codegen.auto_tests import (
    generate_operator_test, generate_all_scaffolds,
    verify_operator_contract, verify_all_contracts, generate_compliance_report,
)


# ============================================================================
# Doc 08: Executor
# ============================================================================
class TestTokenPool:
    def test_acquire_release(self):
        pool = TokenPool(max_tokens=2)
        assert pool.acquire("t1")
        assert pool.acquire("t2")
        assert not pool.acquire("t3")  # full
        pool.release("t1")
        assert pool.acquire("t3")
        assert pool.active_count() == 2

    def test_available(self):
        pool = TokenPool(max_tokens=1)
        assert pool.available()
        pool.acquire("t1")
        assert not pool.available()


class TestExecutionToken:
    def test_active(self):
        token = ExecutionToken(trial_id="t1")
        assert token.active
        token.release()
        assert not token.active


class TestSchedulerExecutor:
    def test_submit_and_wait(self):
        executor = SchedulerExecutor(max_workers=2)
        spec = TrialSpec(
            trial_id="t1", portfolio_id="p",
            model_spec=ModelSpec(family="linear", backend="sklearn"),
            fidelity=FidelitySpec(),
        )
        def dummy_fn(s):
            return TrialResult(
                trial_id=s.trial_id, status=TrialStatus.COMPLETED,
                metrics=[MetricValue("score", 0.5, Direction.MAXIMIZE, "cv")],
            )
        task = TrialTask(trial_spec=spec, fn=dummy_fn)
        accepted = executor.submit_task(task)
        assert accepted
        results = executor.wait_all(timeout_s=5)
        assert "t1" in results
        assert results["t1"].status == TrialStatus.COMPLETED
        executor.shutdown()

    def test_submit_rejected_when_full(self):
        executor = SchedulerExecutor(max_workers=1)
        # Use a slow task to keep the token occupied
        def slow_fn(s):
            time.sleep(0.5)
            return TrialResult(trial_id=s.trial_id, status=TrialStatus.COMPLETED)
        spec = TrialSpec(
            trial_id="t1", portfolio_id="p",
            model_spec=ModelSpec(family="linear", backend="sklearn"),
            fidelity=FidelitySpec(),
        )
        executor.submit_task(TrialTask(trial_spec=spec, fn=slow_fn))
        time.sleep(0.05)  # let it start
        # Second should be rejected
        spec2 = TrialSpec(
            trial_id="t_reject", portfolio_id="p",
            model_spec=ModelSpec(family="linear", backend="sklearn"),
            fidelity=FidelitySpec(),
        )
        assert not executor.submit_task(TrialTask(trial_spec=spec2, fn=slow_fn))
        executor.shutdown()

    def test_summary(self):
        executor = SchedulerExecutor(max_workers=2)
        s = executor.summary()
        assert s["max_workers"] == 2
        assert s["completed"] == 0
        executor.shutdown()


# ============================================================================
# Doc 07: Ensemble
# ============================================================================
class TestCheckEligibility:
    def test_filters_ineligible(self):
        results = [
            TrialResult(trial_id="t1", status=TrialStatus.COMPLETED,
                        fold_metrics=[MetricValue("rmse", 0.5, Direction.MINIMIZE, "cv")]),
            TrialResult(trial_id="t2", status=TrialStatus.FAILED),
            TrialResult(trial_id="t3", status=TrialStatus.COMPLETED, fold_metrics=[]),
        ]
        eligible = check_eligibility(results, EnsemblePolicy())
        assert len(eligible) == 1

    def test_all_eligible(self):
        results = [
            TrialResult(trial_id="t1", status=TrialStatus.COMPLETED,
                        fold_metrics=[MetricValue("rmse", 0.5, Direction.MINIMIZE, "cv")]),
            TrialResult(trial_id="t2", status=TrialStatus.COMPLETED,
                        fold_metrics=[MetricValue("rmse", 0.6, Direction.MINIMIZE, "cv")]),
        ]
        eligible = check_eligibility(results, EnsemblePolicy())
        assert len(eligible) == 2


class TestComputeDiversity:
    def test_identical_results(self):
        results = [
            TrialResult(trial_id="t1", status=TrialStatus.COMPLETED,
                        fold_metrics=[MetricValue("rmse", 0.5, Direction.MINIMIZE, "cv")]),
            TrialResult(trial_id="t2", status=TrialStatus.COMPLETED,
                        fold_metrics=[MetricValue("rmse", 0.5, Direction.MINIMIZE, "cv")]),
        ]
        assert compute_diversity(results) == 0.0

    def test_diverse_results(self):
        results = [
            TrialResult(trial_id="t1", status=TrialStatus.COMPLETED,
                        fold_metrics=[MetricValue("rmse", 0.2, Direction.MINIMIZE, "cv")]),
            TrialResult(trial_id="t2", status=TrialStatus.COMPLETED,
                        fold_metrics=[MetricValue("rmse", 0.8, Direction.MINIMIZE, "cv")]),
        ]
        assert compute_diversity(results) > 0.3


class TestOptimizeWeights:
    def test_equal(self):
        results = [
            TrialResult(trial_id="t1", status=TrialStatus.COMPLETED,
                        fold_metrics=[MetricValue("rmse", 0.5, Direction.MINIMIZE, "cv")]),
            TrialResult(trial_id="t2", status=TrialStatus.COMPLETED,
                        fold_metrics=[MetricValue("rmse", 0.5, Direction.MINIMIZE, "cv")]),
        ]
        weights = optimize_weights(results, EnsemblePolicy(weight_method="equal"))
        assert len(weights) == 2
        assert abs(weights[0] - 0.5) < 1e-10

    def test_inverse_error(self):
        results = [
            TrialResult(trial_id="t1", status=TrialStatus.COMPLETED,
                        fold_metrics=[MetricValue("rmse", 0.1, Direction.MINIMIZE, "cv")]),
            TrialResult(trial_id="t2", status=TrialStatus.COMPLETED,
                        fold_metrics=[MetricValue("rmse", 0.9, Direction.MINIMIZE, "cv")]),
        ]
        weights = optimize_weights(results, EnsemblePolicy(weight_method="inverse_error"))
        assert weights[0] > weights[1]  # lower error = higher weight


class TestBuildEnsemble:
    def test_too_few_trials(self):
        results = [
            TrialResult(trial_id="t1", status=TrialStatus.COMPLETED,
                        fold_metrics=[MetricValue("rmse", 0.5, Direction.MINIMIZE, "cv")]),
        ]
        assert build_ensemble(results, EnsemblePolicy(min_trials=2)) is None

    def test_successful_ensemble(self):
        results = [
            TrialResult(trial_id="t1", status=TrialStatus.COMPLETED,
                        fold_metrics=[MetricValue("rmse", 0.2, Direction.MINIMIZE, "cv")]),
            TrialResult(trial_id="t2", status=TrialStatus.COMPLETED,
                        fold_metrics=[MetricValue("rmse", 0.8, Direction.MINIMIZE, "cv")]),
        ]
        spec = build_ensemble(results, EnsemblePolicy(
            min_diversity=0.0, stack_overfit_threshold=10.0))
        assert spec is not None
        assert len(spec.member_trial_ids) == 2
        assert len(spec.weights) == 2


class TestParetoFront:
    def test_basic(self):
        results = [
            TrialResult(trial_id="t1", status=TrialStatus.COMPLETED,
                        fold_metrics=[MetricValue("score", 0.9, Direction.MAXIMIZE, "cv")],
                        resource_measurements=[ResourceMeasurement(wall_time_s=10.0)]),
            TrialResult(trial_id="t2", status=TrialStatus.COMPLETED,
                        fold_metrics=[MetricValue("score", 0.5, Direction.MAXIMIZE, "cv")],
                        resource_measurements=[ResourceMeasurement(wall_time_s=5.0)]),
            TrialResult(trial_id="t3", status=TrialStatus.COMPLETED,
                        fold_metrics=[MetricValue("score", 0.7, Direction.MAXIMIZE, "cv")],
                        resource_measurements=[ResourceMeasurement(wall_time_s=8.0)]),
        ]
        front = pareto_front(results)
        # t1 (high quality, high cost) and t2 (low quality, low cost) are on front
        # t3 might be dominated
        assert len(front) >= 2


# ============================================================================
# Doc 11: Security Sandbox
# ============================================================================
class TestValidateInputCode:
    def test_safe_code(self):
        warnings = validate_input_code("x = a + b")
        assert len(warnings) == 0

    def test_forbidden_import(self):
        warnings = validate_input_code("import os")
        assert len(warnings) > 0
        assert any("os" in w for w in warnings)

    def test_forbidden_eval(self):
        warnings = validate_input_code("eval('1+1')")
        assert len(warnings) > 0

    def test_syntax_error(self):
        warnings = validate_input_code("def foo(")
        assert len(warnings) > 0


class TestRunInSandbox:
    def test_successful_execution(self):
        def add(a, b):
            return a + b
        result = run_in_sandbox(add, args=(2, 3))
        assert result.success
        assert result.result == 5

    def test_exception_handling(self):
        def fail():
            raise ValueError("test error")
        result = run_in_sandbox(fail)
        assert not result.success
        assert "test error" in result.error

    def test_timeout(self):
        def slow():
            time.sleep(10)
        result = run_in_sandbox(
            slow,
            policy=SandboxPolicy(max_execution_time_s=0.1),
        )
        assert not result.success
        assert "timed out" in result.error.lower()


class TestValidateNumericInput:
    def test_clean(self):
        warnings = validate_numeric_input([1.0, 2.0, 3.0])
        assert len(warnings) == 0

    def test_nan(self):
        warnings = validate_numeric_input([1.0, float("nan"), 3.0])
        assert len(warnings) > 0

    def test_inf(self):
        warnings = validate_numeric_input([1.0, float("inf")])
        assert len(warnings) > 0


class TestValidateColumnName:
    def test_clean(self):
        assert len(validate_column_name("price")) == 0

    def test_empty(self):
        assert len(validate_column_name("")) > 0

    def test_sql_injection(self):
        warnings = validate_column_name("price; DROP TABLE")
        assert len(warnings) > 0


# ============================================================================
# Doc 09: Codegen
# ============================================================================
class TestGenerateOperatorTest:
    def test_unary_operator(self):
        from fiae.features.registry import get_operator
        op = get_operator("sqrt")
        scaffold = generate_operator_test(op)
        assert scaffold.operator_name == "sqrt"
        assert "null_preservation" in scaffold.test_code
        assert "output_length" in scaffold.test_code
        assert "deterministic" in scaffold.test_code
        assert "functional" in scaffold.test_code

    def test_binary_operator(self):
        from fiae.features.registry import get_operator
        op = get_operator("sum")
        scaffold = generate_operator_test(op)
        assert scaffold.operator_name == "sum"
        assert "op.transform([1.0, None, 3.0], [4.0, 5.0, 6.0])" in scaffold.test_code


class TestGenerateAllScaffolds:
    def test_all_operators(self):
        scaffolds = generate_all_scaffolds()
        assert len(scaffolds) == 95
        for s in scaffolds:
            assert s.operator_name
            assert s.test_code
            assert "null_preservation" in s.test_code


class TestVerifyOperatorContract:
    def test_valid_operator(self):
        from fiae.features.registry import get_operator
        op = get_operator("sqrt")
        violations = verify_operator_contract(op)
        assert len(violations) == 0

    def test_all_compliant(self):
        violations = verify_all_contracts()
        # Target-aware operators intentionally have 2 inputs but are 'unary'
        # (one data input + one target input)
        expected_violations = {"numeric_to_cat_target", "target_mean_crossfit", "woe_crossfit"}
        unexpected = {k: v for k, v in violations.items() if k not in expected_violations}
        assert len(unexpected) == 0, f"Unexpected violations: {unexpected}"


class TestComplianceReport:
    def test_report(self):
        report = generate_compliance_report()
        assert report["total_operators"] == 95
        assert report["compliance_rate"] > 0.95  # target-aware operators have 2 inputs
