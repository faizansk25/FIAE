"""M39.1 regression tests: the four external-audit refinements to M39.

1. Feature identity must canonicalize *values*, not just sort keys.
2. One cross-layer identity rule (logical ancestry, never IR node ids).
3. ``RunStatus`` is an enum, and completion is independent of outcome.
4. Export promotion is atomic (staging + rename), not "write at the end".
"""

import json

import pytest

from fiae.codegen.pipeline_ir import build_ir_from_proposals
from fiae.fitted_pipeline import (
    FittedPipeline,
    canonical_feature_id,
    feature_identity,
)
from fiae.pipeline.canonical import (
    CodegenResult,
    EvalResult,
    RunContext,
    RunStatus,
    phase_codegen,
)
from fiae.search.triggers import FeatureProposal


def _p(op="f", inputs=("raw:c",), **params):
    return FeatureProposal(op=op, inputs=list(inputs), params=params)


class TestIdentityCanonicalizesValues:
    @pytest.mark.parametrize("left,right", [
        ({"a": "x,y=z"}, {"a": "x", "y": "z"}),          # separator ambiguity
        ({"a": 1}, {"a": True}),                          # bool vs int
        ({"a": 0.1}, {"a": 0.10000000000000002}),         # float repr precision
    ])
    def test_distinct_param_payloads_never_share_an_identity(self, left, right):
        # Pre-M39.1 these pairs rendered to the same string.
        assert feature_identity(_p(**left)) != feature_identity(_p(**right))
        assert canonical_feature_id(_p(**left)) != canonical_feature_id(_p(**right))

    @pytest.mark.parametrize("left,right", [
        ({"a": {1, 2}}, {"a": {2, 1}}),                    # unordered set
        ({"a": {"x": 1, "y": 2}}, {"a": {"y": 2, "x": 1}}),  # dict key order
    ])
    def test_equal_values_share_an_identity_across_container_order(self, left, right):
        # Equal by value: a set and an unordered dict have no order, so a
        # *correct* canonicalizer must agree with itself here.
        assert feature_identity(_p(**left)) == feature_identity(_p(**right))
        assert canonical_feature_id(_p(**left)) == canonical_feature_id(_p(**right))

    def test_equal_values_share_an_identity_regardless_of_insertion_order(self):
        assert feature_identity(_p(a=1, b=2)) == feature_identity(_p(b=2, a=1))

    def test_identity_is_stable_for_nan_and_unhashable_params(self):
        nan = float("nan")
        a, b = _p(cut=nan), _p(cut=nan)
        # NaN != NaN, so an identity built with dict/set equality would
        # disagree with itself; a canonical hash must not.
        assert feature_identity(a) == feature_identity(b)
        assert canonical_feature_id(_p(cut=nan, tags={"x", "y"})) \
            == canonical_feature_id(_p(cut=nan, tags={"y", "x"}))

    def test_fit_state_is_keyed_by_identity_and_keeps_a_readable_name(self):
        pipe = FittedPipeline()
        p = FeatureProposal(op="standardize", inputs=["raw:x"])
        out = pipe.fit([p], {"x": [1.0, 2.0, 3.0]})

        # Output columns keep the human-readable name (consumers depend on it)
        assert "standardize(x)" in out
        # Persisted state is keyed by the collision-free identity.
        assert list(pipe.states) == [feature_identity(p)]
        assert pipe.states[feature_identity(p)].display_name == "standardize(x)"
        assert pipe.get_state_dict()[feature_identity(p)]["display_name"] \
            == "standardize(x)"

    def test_state_dict_round_trip_is_keyed_by_identity(self):
        pipe = FittedPipeline()
        p = FeatureProposal(op="standardize", inputs=["raw:x"])
        pipe.fit([p], {"x": [1.0, 2.0, 3.0]})
        restored = FittedPipeline()
        restored.load_state_dict(json.loads(json.dumps(pipe.get_state_dict())))
        assert restored.states[feature_identity(p)].fit_state == \
            pipe.states[feature_identity(p)].fit_state


class TestOneIdentityRuleAcrossLayers:
    def test_chained_proposal_resolves_to_the_ancestor_node_id(self):
        base = FeatureProposal(op="log1p", inputs=["raw:x"], params={})
        chained = FeatureProposal(
            op="sqrt",
            inputs=[canonical_feature_id(base), "raw:y"],
            params={},
        )
        ir = build_ir_from_proposals([base, chained])

        base_node = ir.nodes[0]
        assert base_node.node_id == "n_0"
        # The chained input must resolve to the ancestor's IR node id --
        # proof the IR builder hashes the *logical* name, not "n_0" vs the
        # name (a mismatch here is the cross-layer identity bug).
        assert ir.nodes[1].inputs == ["n_0", "raw:y"]
        assert ir.verify_dag() == []

    def test_runtime_and_ir_agree_on_the_same_name_for_the_same_proposal(self):
        p = FeatureProposal(op="winsorize", inputs=["raw:age"],
                            params={"lower": 0.05, "upper": 0.95})
        pipe = FittedPipeline()
        out = pipe.fit([p], {"age": [float(v) for v in range(100)]})
        ir = build_ir_from_proposals([p])
        assert list(out) == [canonical_feature_id(p)]
        assert ir.nodes[0].operator == p.op
        assert ir.nodes[0].params == dict(p.params)


class TestRunStatusEnum:
    def test_status_is_an_enum_value_that_still_compares_to_strings(self):
        assert RunStatus.PARTIAL.value == "PARTIAL"
        assert RunStatus("PARTIAL") is RunStatus.PARTIAL

    def test_completion_and_outcome_are_independent(self):
        # 10 phases executed, structured errors present -> PARTIAL, not
        # FAILED, and phases_completed still says 10.
        from fiae.pipeline.canonical import CanonicalResult

        result = CanonicalResult(phases_completed=10,
                                 status=RunStatus.PARTIAL)
        summary = result.summary()
        assert summary["phases_completed"] == 10
        assert summary["status"] == "PARTIAL"
        assert result.status == "PARTIAL"  # str mixin keeps callers working


class TestAtomicExportPromotion:
    def test_promoted_export_leaves_no_staging_directory(self, tmp_path,
                                                          monkeypatch):
        import types

        ok_report = types.SimpleNamespace(
            all_passed=True,
            generated_code="def apply_pipeline(data):\n    return {}\n",
            gates=[types.SimpleNamespace(gate_name="feature_parity",
                                         passed=True, message="ok")],
            summary=lambda: {"passed": 1, "total_gates": 1},
        )
        from fiae.codegen import compiler as compiler_mod
        monkeypatch.setattr(compiler_mod, "compile_pipeline",
                            lambda ir, **kw: ok_report)

        ctx = RunContext(run_id="t", run_dir=str(tmp_path),
                         config_snapshot={"source": "", "target": "y"})

        class F:
            portfolio_proposals = [FeatureProposal(op="identity",
                                                    inputs=["raw:x"], params={})]
            portfolio_size = 1

        result = phase_codegen(ctx, F(), EvalResult())
        assert result.export_code_path == str(tmp_path / "export" / "features.py")
        assert (tmp_path / "export" / "features.py").exists()
        # Atomic promotion: the staging path must not survive.
        assert not (tmp_path / "export" / ".staging").exists()

    def test_failed_gates_leave_no_artifact_at_all(self, tmp_path,
                                                   monkeypatch):
        import types

        bad = types.SimpleNamespace(
            all_passed=False,
            generated_code="def apply_pipeline(data):\n    return {}\n",
            gates=[types.SimpleNamespace(gate_name="feature_parity",
                                         passed=False, message="x")],
            summary=lambda: {"passed": 0, "total_gates": 1},
        )
        from fiae.codegen import compiler as compiler_mod
        monkeypatch.setattr(compiler_mod, "compile_pipeline",
                            lambda ir, **kw: bad)

        ctx = RunContext(run_id="t", run_dir=str(tmp_path),
                         config_snapshot={"source": "", "target": "y"})

        class F:
            portfolio_proposals = [FeatureProposal(op="identity",
                                                    inputs=["raw:x"], params={})]
            portfolio_size = 1

        result = phase_codegen(ctx, F(), EvalResult())
        assert isinstance(result, CodegenResult)
        assert result.export_code_path == ""
        assert not (tmp_path / "export" / "features.py").exists()
