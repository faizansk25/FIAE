"""Tests for F4 progressive evaluation and F6 final stability (evaluate.py)."""


from fiae.contracts import Task
from fiae.evaluate import EvaluatePolicy, f4_progressive_eval, f6_final_stability
from fiae.search.records import FeatureAcceptanceRecord


def _record(op="f_x"):
    return FeatureAcceptanceRecord(
        feature_id=f"cand_{op}", operator=op, inputs=["a"], params={},
        source="test", trigger_reason="test",
    )


def _data(n=120, noise=0.01):
    xs = [float(i) for i in range(n)]
    y = [2.0 * x + 1.0 + (i % 7) * noise for i, x in enumerate(xs)]
    return [xs], y


def test_f4_passes_stable_signal():
    xs, y = _data()
    rec = _record()
    v = f4_progressive_eval(rec, [], xs[0], y, Task.REGRESSION)
    assert v.passed
    assert v.stage == "F4"
    assert len(metrics_stages(v)) == 3
    assert rec.incremental_gain is None  # F4 does not overwrite F3 evidence


def metrics_stages(v):
    return v.metrics["stages"]


def test_f4_rejects_vanishing_gain():
    n = 120
    noise = [1.0 if i < n // 4 else 0.0 for i in range(n)]  # helps only early
    xs = [float(i % 10) for i in range(n)]
    y = [x + 100.0 * e for x, e in zip(xs, noise)]
    rec = _record()
    v = f4_progressive_eval(rec, [], xs, y, Task.REGRESSION)
    assert not v.passed
    assert rec.final_reason


def test_f4_rejects_length_mismatch():
    rec = _record()
    v = f4_progressive_eval(rec, [], [1.0, 2.0], [1.0, 2.0, 3.0], Task.REGRESSION)
    assert not v.passed
    assert "length" in (v.reason or "")


def test_f6_passes_consistent_signal():
    xs, y = _data(noise=0.05)
    rec = _record()
    v = f6_final_stability(rec, [], xs[0], y, Task.REGRESSION)
    assert v.passed
    assert len(v.metrics["seeds"]) == 3
    assert rec.fold_stability is not None and rec.fold_stability > 0
    assert "f6_gain_mean" in rec.scores


def test_f6_rejects_inconsistent_signal():
    # gain only under one seed's fold partition is unlikely; use pure noise
    n = 120
    y = [(i * 37) % 11 - 5.0 for i in range(n)]
    xs = [float((i * 53) % 7) for i in range(n)]
    rec = _record()
    v = f6_final_stability(rec, [], xs, y, Task.REGRESSION)
    assert not v.passed
    assert rec.final_reason


def test_f6_rejects_length_mismatch():
    rec = _record()
    v = f6_final_stability(rec, [], [1.0], [1.0, 2.0], Task.REGRESSION)
    assert not v.passed


def test_f4_policy_custom_stages():
    xs, y = _data()
    rec = _record()
    pol = EvaluatePolicy(f4_stages=(1.0,))
    v = f4_progressive_eval(rec, [], xs[0], y, Task.REGRESSION, pol)
    assert v.passed
    assert len(v.metrics["stages"]) == 1
