"""Funnel gates F0-F2 (doc 04) and semantic-hint proposals (source 3)."""

import dataclasses

import pandas as pd

from fiae.contracts import ColumnProfile, DatasetProfile, SemanticType
from fiae.features.registry import get_operator
from fiae.funnel import FunnelPolicy, f0_preconditions, f1_static_resources, f2_materialize, run_funnel
from fiae.intake import CsvDataSourceAdapter, ProfileConfig, profile_source
from fiae.search.hints import build_hint_candidates, propose_hints
from fiae.search.records import StageVerdict
from fiae.search.triggers import FeatureProposal


def _profile(df: pd.DataFrame) -> tuple[DatasetProfile, dict[str, list]]:
    df.to_csv("_t.csv", index=False)
    try:
        prof = profile_source(CsvDataSourceAdapter("_t.csv"), ProfileConfig())
    finally:
        import os

        os.remove("_t.csv")
    sample = {c.name: df[c.name].tolist() for c in prof.columns}
    return prof, sample


def _proposal(op: str, inputs: list[str]) -> FeatureProposal:
    return FeatureProposal(op=op, inputs=inputs, source="test")


def test_f0_rejects_unknown_operator():
    prof, _ = _profile(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
    v = f0_preconditions(_proposal("no_such_op", ["raw:a"]), prof, FunnelPolicy())
    assert not v.passed and "unknown operator" in v.reason


def test_f0_rejects_missing_input():
    prof, _ = _profile(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
    v = f0_preconditions(_proposal("log1p", ["raw:zzz"]), prof, FunnelPolicy())
    assert not v.passed and "not available" in v.reason


def test_f0_rejects_arity_mismatch():
    prof, _ = _profile(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
    v = f0_preconditions(_proposal("difference", ["raw:a"]), prof, FunnelPolicy())
    assert not v.passed and "arity" in v.reason


def test_f0_rejects_depth_over_policy():
    prof, _ = _profile(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
    p = dataclasses.replace(_proposal("log1p", ["raw:a"]), depth=3)
    v = f0_preconditions(p, prof, FunnelPolicy(max_depth=1))
    assert not v.passed and "depth" in v.reason


def test_f0_rejects_semantic_mismatch():
    prof, _ = _profile(pd.DataFrame({"cat": ["x", "y", "x"]}))
    v = f0_preconditions(_proposal("log1p", ["raw:cat"]), prof, FunnelPolicy())
    assert not v.passed and "incompatible" in v.reason


def test_f0_accepts_valid_numeric():
    prof, _ = _profile(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
    v = f0_preconditions(_proposal("log1p", ["raw:a"]), prof, FunnelPolicy())
    assert v.passed


def test_f1_rejects_when_budget_exhausted():
    _prof, _ = _profile(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
    op = get_operator("log1p")
    p = _proposal("log1p", ["raw:a"])
    tight = FunnelPolicy(max_estimated_cpu_seconds=0.0)
    v = f1_static_resources(p, op, 10_000_000, tight)
    assert not v.passed and "cpu" in v.reason
    ok = f1_static_resources(p, op, 1_000, FunnelPolicy())
    assert ok.passed


def test_f2_rejects_constant_output():
    p = _proposal("abs", ["raw:a"])
    op = get_operator("abs")
    v = f2_materialize(p, op, {"a": [-2.0, -2.0, -2.0]}, FunnelPolicy())
    assert not v.passed and "constant" in v.reason


def test_f2_rejects_high_invalid_rate():
    p = _proposal("sqrt", ["raw:a"])
    op = get_operator("sqrt")
    v = f2_materialize(p, op, {"a": [-1.0, -2.0, -3.0, -4.0]}, FunnelPolicy())
    assert not v.passed and "invalid_rate" in v.reason


def test_f2_accepts_healthy_output():
    p = _proposal("log1p", ["raw:a"])
    op = get_operator("log1p")
    v = f2_materialize(p, op, {"a": [1.0, 2.0, 3.0, 4.0]}, FunnelPolicy())
    assert v.passed


def test_run_funnel_end_to_end_pass():
    prof, sample = _profile(
        pd.DataFrame({"price": [10.0, 20.0, 30.0, 40.0], "qty": [2, 4, 5, 8]})
    )
    rec = run_funnel(_proposal("safe_ratio", ["raw:price", "raw:qty"]), prof, sample)
    assert rec.passed
    assert [s.stage for s in rec.stages] == ["F0", "F1", "F2"]
    assert rec.final_reason == ""
    assert rec.lineage == ["price", "qty"]
    d = rec.to_dict()
    assert d["stages"][2]["passed"] is True


def test_run_funnel_stops_at_f0_on_bad_input():
    prof, _ = _profile(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
    rec = run_funnel(_proposal("difference", ["raw:a", "raw:ghost"]), prof, {"a": [1.0]})
    assert not rec.passed
    assert len(rec.stages) == 1 and rec.stages[0].stage == "F0"
    assert "not available" in rec.final_reason


# --- source 3: semantic name hints -------------------------------------------


def _col(name: str, sem: SemanticType) -> ColumnProfile:
    return ColumnProfile(
        name=name, physical_dtype="float", semantic_type=sem,
        null_fraction=0.0, distinct_estimate=3, distinct_ratio=0.5,
        statistics={},
    )


def test_hints_price_column():
    props = propose_hints(_col("total_price", SemanticType.CURRENCY))
    ops = {p.op for p in props}
    assert ops == {"log1p", "safe_ratio"}
    assert all(p.source == "hints:name" for p in props)


def test_hints_skip_identifier():
    assert propose_hints(_col("user_id", SemanticType.IDENTIFIER)) == []


def test_hints_flag_prefix():
    props = propose_hints(_col("is_active", SemanticType.BOOLEAN))
    assert [p.op for p in props] == ["identity"]


def test_hints_bounded_to_first_pattern():
    # "date" matches the datetime pattern; nothing else may pile on
    props = propose_hints(_col("order_date", SemanticType.DATETIME))
    assert {p.op for p in props} == {"year", "month", "day_of_week"}


def test_build_hint_candidates_over_profile():
    prof, _ = _profile(
        pd.DataFrame({"total_price": [1.0, 2.0], "user_id": ["a", "b"]})
    )
    props = build_hint_candidates(prof)
    assert any(p.op == "log1p" and p.inputs == ["raw:total_price"] for p in props)
    assert all("user_id" not in p.inputs for p in props)


def test_stage_verdict_serialization_still_fine():
    v = StageVerdict("F0", True, None, {"k": 1})
    assert v.to_dict() == {"stage": "F0", "passed": True, "reason": None, "metrics": {"k": 1}}
