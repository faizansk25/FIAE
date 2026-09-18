import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fiae.problem.calibration import (
    BinnedCalibrator,
    brier,
    distribution_shift_probe,
    freeze_decision_policy,
    log_loss_binary,
    optimize_threshold,
)


def test_brier_known_value():
    y = [0.0, 1.0]
    p = [0.2, 0.8]
    assert abs(brier(y, p) - 0.04) < 1e-12


def test_log_loss_known_value():
    import math

    y = [0.0, 1.0]
    p = [0.2, 0.8]
    expected = -0.5 * (
        math.log(0.8) + math.log(0.8)
    )  # -(log(1-0.2) + log(0.8))/2
    assert abs(log_loss_binary(y, p) - expected) < 1e-9


def test_log_loss_clips_extremes():
    assert abs(log_loss_binary([1.0], [0.0]) - log_loss_binary([1.0], [1e-15])) < 1e-6


def test_binned_calibrator_is_monotone():
    import random

    rng = random.Random(0)
    scores = [rng.random() for _ in range(3000)]
    y = [1.0 if s > 0.5 else 0.0 for s in scores]
    cal = BinnedCalibrator(n_bins=10).fit(scores, y)
    probe = [i / 100 for i in range(101)]
    out = cal.transform(probe)
    assert all(out[i] <= out[i + 1] + 1e-9 for i in range(len(out) - 1))
    assert out[0] < 0.1  # low scores map near 0
    assert out[-1] > 0.9  # high scores map near 1


def test_calibrator_requires_fit():
    cal = BinnedCalibrator()
    try:
        cal.transform([0.5])
        raise AssertionError("transform without fit must raise")
    except RuntimeError:
        pass


def test_threshold_f1_sanity():
    scores = [i / 100 for i in range(101)]
    y = [1.0 if s >= 0.5 else 0.0 for s in scores]
    res = optimize_threshold(scores, y, objective="f1")
    assert res.threshold > 0.3
    assert res.value > 0.9
    assert res.n_pos == 51 and res.n_neg == 50


def test_threshold_precision_floor():
    scores = [i / 100 for i in range(101)]
    y = [1.0 if s >= 0.7 else 0.0 for s in scores]
    res = optimize_threshold(scores, y, objective="precision_floor", precision_floor=0.9)
    assert res.at_precision >= 0.9


def test_threshold_utility_weighting():
    scores = [i / 100 for i in range(101)]
    y = [1.0 if s >= 0.5 else 0.0 for s in scores]
    utility = {"tp": 1.0, "fp": -5.0, "tn": 0.0, "fn": -1.0}
    res = optimize_threshold(scores, y, objective="utility", utility=utility)
    # heavy false-positive penalty pushes threshold above naive 0.5
    assert res.threshold >= 0.5


def test_threshold_no_feasible_returns_default():
    scores = [i / 100 for i in range(101)]
    y = [1.0 if s >= 0.9 else 0.0 for s in scores]
    res = optimize_threshold(scores, y, objective="precision_floor", precision_floor=1.0)
    # No threshold reaches perfect precision over all rows; default warned.
    assert res.warnings or res.at_precision >= 1.0 - 1e-9


def test_freeze_policy_deterministic_and_sensitive():
    a = freeze_decision_policy(
        optimize_threshold([0.1, 0.9], [0.0, 1.0], objective="f1")
    )
    b = freeze_decision_policy(
        optimize_threshold([0.1, 0.9], [0.0, 1.0], objective="f1")
    )
    assert a.fingerprint == b.fingerprint
    c = freeze_decision_policy(
        optimize_threshold([0.1, 0.9], [0.0, 1.0], objective="f1"),
        calibrator_ref="cal#1",
    )
    assert a.fingerprint != c.fingerprint


def test_distribution_shift_probe():
    probe = distribution_shift_probe(
        [float(i) for i in range(50)], [100.0 + i for i in range(50)]
    )
    assert probe["separation_auc"] == 1.0
    assert probe["high_shift_risk"] is True
    same = distribution_shift_probe(
        [float(i % 10) for i in range(200)], [float(i % 10) for i in range(200)]
    )
    assert same["high_shift_risk"] is False
