"""F3 incremental probe and greedy portfolio selection (doc 04)."""

import random

import pytest

from fiae.contracts import Task
from fiae.probe import (
    ProbePolicy,
    f3_incremental_probe,
    pearson,
    select_portfolio,
)
from fiae.search.records import FeatureAcceptanceRecord


def _rec() -> FeatureAcceptanceRecord:
    return FeatureAcceptanceRecord(feature_id="cand_x", operator="log1p")


def test_f3_passes_predictive_candidate():
    rng = random.Random(7)
    x = [rng.uniform(0, 10) for _ in range(200)]
    y = [3.0 * v + rng.gauss(0, 0.1) for v in x]
    rec = _rec()
    v = f3_incremental_probe(rec, [], x, y, Task.REGRESSION, ProbePolicy())
    assert v.passed
    assert rec.incremental_gain > 1.0
    assert rec.scores["f3_gain"] == rec.incremental_gain
    assert rec.stages[-1].stage == "F3"


def test_f3_rejects_noise_candidate():
    rng = random.Random(11)
    x = [rng.uniform(0, 10) for _ in range(200)]
    noise = [rng.uniform(0, 10) for _ in range(200)]
    y = [3.0 * v for v in x]
    rec = _rec()
    v = f3_incremental_probe(rec, [], noise, y, Task.REGRESSION, ProbePolicy())
    assert not v.passed
    assert "no incremental gain" in rec.final_reason


def test_f3_length_mismatch_rejected():
    rec = _rec()
    v = f3_incremental_probe(rec, [], [1.0, 2.0], [1.0, 2.0, 3.0],
                             Task.REGRESSION, ProbePolicy())
    assert not v.passed and "mismatch" in v.reason


def test_f3_redundant_candidate_rejected():
    """A candidate duplicating a base column adds no gain."""
    rng = random.Random(13)
    x = [rng.uniform(0, 10) for _ in range(200)]
    y = [2.0 * v + rng.gauss(0, 0.05) for v in x]
    rec = _rec()
    v = f3_incremental_probe(rec, [x], list(x), y, Task.REGRESSION, ProbePolicy())
    assert not v.passed


def test_f3_binary_task_brier():
    rng = random.Random(17)
    x = [rng.uniform(0, 10) for _ in range(200)]
    y = [1.0 if v > 5 else 0.0 for v in x]
    rec = _rec()
    v = f3_incremental_probe(rec, [], x, y, Task.BINARY, ProbePolicy())
    assert v.passed
    assert v.metrics["task"] == "binary_classification"


def test_pearson_basics():
    assert pearson([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert pearson([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    assert pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0
    assert pearson([1.0], [1.0]) == 0.0


def test_portfolio_picks_signal_skips_noise_and_duplicates():
    rng = random.Random(23)
    x = [rng.uniform(0, 10) for _ in range(300)]
    y = [2.0 * v + rng.gauss(0, 0.1) for v in x]
    candidates = {
        "signal": x,
        "signal_copy": list(x),                      # near-duplicate of signal
        "noise": [rng.uniform(0, 10) for _ in range(300)],
        "noise2": [rng.uniform(0, 10) for _ in range(300)],
    }
    selected, gains = select_portfolio(candidates, y, Task.REGRESSION,
                                       ProbePolicy(n_folds=3))
    assert selected == ["signal"]
    assert gains["signal"] > 0


def test_portfolio_empty_and_max_features():
    rng = random.Random(29)
    y = [rng.uniform(0, 1) for _ in range(60)]
    assert select_portfolio({}, y, Task.REGRESSION, ProbePolicy()) == ([], {})
    xs = {f"f{i}": [rng.uniform(0, 1) for _ in range(60)] for i in range(20)}
    selected, _ = select_portfolio(xs, y, Task.REGRESSION, ProbePolicy(),
                                   max_features=3)
    assert len(selected) <= 3


def test_portfolio_adds_second_independent_signal():
    rng = random.Random(31)
    a = [rng.uniform(0, 10) for _ in range(400)]
    b = [rng.uniform(0, 10) for _ in range(400)]
    y = [2.0 * u + 1.5 * w + rng.gauss(0, 0.1) for u, w in zip(a, b)]
    candidates = {"a": a, "b": b, "noise": [rng.uniform(0, 10) for _ in range(400)]}
    selected, gains = select_portfolio(candidates, y, Task.REGRESSION,
                                       ProbePolicy())
    assert "a" in selected and "b" in selected
    assert "noise" not in selected
    assert gains["b"] > 0
