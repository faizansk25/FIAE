"""M39 regression tests: semantic-correctness invariants in the canonical path.

Each test targets one verified external-audit finding; each fails against
the pre-M39 code.
"""

import types

import pytest

from fiae.fitted_pipeline import FittedPipeline, canonical_feature_id
from fiae.pipeline.canonical import (
    CanonicalResult,
    EvalResult,
    HOResult,
    RunContext,
    phase_codegen,
    phase_evaluate,
    phase_hpo,
    run_canonical_pipeline,
)
from fiae.search.triggers import FeatureProposal


class TestFeatureIdIncludesParams:
    def test_same_op_same_input_different_params_are_distinct(self):
        a = FeatureProposal(op="winsorize", inputs=["raw:age"],
                            params={"lower": 0.01, "upper": 0.99})
        b = FeatureProposal(op="winsorize", inputs=["raw:age"],
                            params={"lower": 0.05, "upper": 0.95})
        assert canonical_feature_id(a) != canonical_feature_id(b)

    def test_fit_does_not_overwrite_state_across_params(self):
        data = {"age": [float(v) for v in range(1, 101)]}
        pipe = FittedPipeline()
        out = pipe.fit([
            FeatureProposal(op="winsorize", inputs=["raw:age"],
                            params={"lower": 0.01, "upper": 0.99}),
            FeatureProposal(op="winsorize", inputs=["raw:age"],
                            params={"lower": 0.05, "upper": 0.95}),
        ], data)
        # Pre-M39: both proposals collapsed to one key "winsorize(age)" and
        # the second fit state silently overwrote the first.
        assert len(out) == 2, f"feature outputs collided: {list(out)}"
        assert len(pipe.states) == 2, "fit states collided"
        # Deterministic ordering: params serialized sorted.
        assert canonical_feature_id(
            FeatureProposal(op="winsorize", inputs=["raw:age"],
                            params={"upper": 0.95, "lower": 0.05})
        ) == canonical_feature_id(
            FeatureProposal(op="winsorize", inputs=["raw:age"],
                            params={"lower": 0.05, "upper": 0.95})
        )


class TestHpoNeverFabricatesBestModel:
    def test_no_training_data_leaves_best_model_empty(self):
        class F:
            portfolio_proposals = []
            portfolio_size = 0

        class V:
            task = "classification"

        result = phase_hpo(None, F(), V())
        assert result.best_model == "", (
            "fabricated a best model with no training data")
        assert result.errors, "degenerate HPO must record why"

    def test_all_trials_failed_leaves_best_model_empty(self, monkeypatch):
        from fiae.orchestration import model_training as mt

        class FailedTrial:
            status = types.SimpleNamespace(name="FAILED")
            metrics = []

        monkeypatch.setattr(
            mt.TrialRunner, "run_trial",
            lambda self, spec, X, y, task: FailedTrial())

        class F:
            portfolio_proposals = [FeatureProposal(
                op="identity", inputs=["raw:x"], params={})]
            portfolio_size = 1

        class V:
            task = "classification"

        result = phase_hpo(None, F(), V(), source_path=__file__, target="y")
        assert result.best_model == "", (
            "fabricated a best model although every trial failed")


class TestEvaluateHandlesNegativeScores:
    def test_negative_regression_score_is_reported_not_zeroed(self):
        hpo = HOResult(best_model="ridge", best_score=-0.42)
        result = phase_evaluate(None, hpo, None)
        assert result.primary_value == pytest.approx(-0.42)
        assert not any("No model trained" in e for e in result.errors)

    def test_missing_model_is_explicit(self):
        hpo = HOResult(best_model="", best_score=0.0)
        result = phase_evaluate(None, hpo, None)
        assert result.primary_value == 0.0
        assert any("No model trained" in e for e in result.errors)


class TestCodegenFailsClosed:
    def test_failed_gates_do_not_promote_export(self, tmp_path, monkeypatch):
        from fiae.codegen import compiler as compiler_mod

        failing_report = types.SimpleNamespace(
            all_passed=False,
            generated_code="def apply_pipeline(data):\n    return {}\n",
            gates=[types.SimpleNamespace(gate_name="feature_parity",
                                         passed=False, message="x")],
            summary=lambda: {"passed": 0, "total_gates": 1},
        )
        monkeypatch.setattr(compiler_mod, "compile_pipeline",
                            lambda ir, **kw: failing_report)

        ctx = RunContext(run_id="t", run_dir=str(tmp_path),
                         config_snapshot={"source": "", "target": "y"})
        class F:
            portfolio_proposals = [FeatureProposal(
                op="identity", inputs=["raw:x"], params={})]
            portfolio_size = 1

        result = phase_codegen(ctx, F(), EvalResult())
        assert result.export_path == ""
        assert result.export_code_path == ""
        assert not (tmp_path / "export" / "features.py").exists()
        assert any("verification gates failed" in e for e in result.errors)


class TestCanonicalRunStatus:
    def test_status_field_exists_and_flows_to_summary(self):
        result = CanonicalResult(run_id="t", phases_completed=10)
        assert result.status in ("SUCCESS", "PARTIAL", "FAILED")
        assert "status" in result.summary()

    def test_missing_source_is_never_silent_success(self, tmp_path):
        # Reality check: a missing source does not raise — phases record
        # structured errors. Either way the run must not report SUCCESS.
        result = run_canonical_pipeline(
            str(tmp_path / "does_not_exist.csv"), "y")
        assert result.status in ("PARTIAL", "FAILED")
        assert result.errors
