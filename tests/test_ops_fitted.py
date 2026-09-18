"""Tests for fitted-state numeric operations (doc 05 entries 12-16)."""

from __future__ import annotations



from fiae.features.ops_fitted import (
    tf_minmax_scale_fit,
    tf_minmax_scale_transform,
    tf_quantile_normal_fit,
    tf_quantile_normal_transform,
    tf_robust_scale_fit,
    tf_robust_scale_transform,
    tf_standardize_fit,
    tf_standardize_transform,
    tf_winsorize_fit,
    tf_winsorize_transform,
)
from fiae.features.registry import all_operators


# ---------------------------------------------------------------------------
# standardize
# ---------------------------------------------------------------------------
class TestStandardize:
    def test_fit(self):
        state = tf_standardize_fit([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(state["mean"] - 3.0) < 1e-10
        assert state["std"] > 0

    def test_transform(self):
        state = {"mean": 3.0, "std": 1.0}
        result = tf_standardize_transform([1.0, 3.0, 5.0], state)
        assert abs(result[0] - (-2.0)) < 1e-10
        assert abs(result[1] - 0.0) < 1e-10
        assert abs(result[2] - 2.0) < 1e-10

    def test_null_preserved(self):
        state = {"mean": 0.0, "std": 1.0}
        result = tf_standardize_transform([1.0, None, 3.0], state)
        assert result[0] is not None
        assert result[1] is None
        assert result[2] is not None

    def test_constant(self):
        state = tf_standardize_fit([5.0, 5.0, 5.0])
        assert state["std"] == 1.0  # safe fallback

    def test_too_few(self):
        state = tf_standardize_fit([1.0])
        assert state["mean"] == 0.0
        assert state["std"] == 1.0


# ---------------------------------------------------------------------------
# robust_scale
# ---------------------------------------------------------------------------
class TestRobustScale:
    def test_fit(self):
        state = tf_robust_scale_fit([1.0, 2.0, 3.0, 4.0, 100.0])
        assert state["median"] == 3.0
        assert state["iqr"] > 0

    def test_transform(self):
        state = tf_robust_scale_fit([1.0, 2.0, 3.0, 4.0, 5.0])
        result = tf_robust_scale_transform([3.0], state)
        assert abs(result[0]) < 1e-10  # median -> 0

    def test_null_preserved(self):
        state = {"median": 0.0, "iqr": 1.0}
        result = tf_robust_scale_transform([1.0, None], state)
        assert result[1] is None


# ---------------------------------------------------------------------------
# minmax_scale
# ---------------------------------------------------------------------------
class TestMinMaxScale:
    def test_fit(self):
        state = tf_minmax_scale_fit([0.0, 50.0, 100.0])
        assert state["min"] == 0.0
        assert state["max"] == 100.0

    def test_transform(self):
        state = tf_minmax_scale_fit([0.0, 100.0])
        result = tf_minmax_scale_transform([0.0, 50.0, 100.0], state)
        assert abs(result[0] - 0.0) < 1e-10
        assert abs(result[1] - 0.5) < 1e-10
        assert abs(result[2] - 1.0) < 1e-10

    def test_constant(self):
        state = tf_minmax_scale_fit([5.0, 5.0])
        result = tf_minmax_scale_transform([5.0], state)
        assert result[0] == 0.5  # midpoint for constant

    def test_null_preserved(self):
        state = {"min": 0.0, "max": 1.0}
        result = tf_minmax_scale_transform([0.5, None], state)
        assert result[1] is None


# ---------------------------------------------------------------------------
# winsorize
# ---------------------------------------------------------------------------
class TestWinsorize:
    def test_fit(self):
        state = tf_winsorize_fit(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0],
            lower_quantile=0.05, upper_quantile=0.95,
        )
        assert state["lower"] <= state["upper"]

    def test_transform_clips(self):
        state = {"lower": 2.0, "upper": 8.0}
        result = tf_winsorize_transform([1.0, 5.0, 10.0], state)
        assert abs(result[0] - 2.0) < 1e-10
        assert abs(result[1] - 5.0) < 1e-10
        assert abs(result[2] - 8.0) < 1e-10

    def test_null_preserved(self):
        state = {"lower": 0.0, "upper": 1.0}
        result = tf_winsorize_transform([0.5, None], state)
        assert result[1] is None


# ---------------------------------------------------------------------------
# quantile_normal
# ---------------------------------------------------------------------------
class TestQuantileNormal:
    def test_fit(self):
        state = tf_quantile_normal_fit([1.0, 2.0, 3.0, 4.0, 5.0])
        assert state["n"] == 5
        assert len(state["values"]) == 5

    def test_transform_monotonic(self):
        state = tf_quantile_normal_fit([1.0, 2.0, 3.0, 4.0, 5.0])
        result = tf_quantile_normal_transform([1.0, 3.0, 5.0], state)
        assert result[0] < result[1] < result[2]

    def test_null_preserved(self):
        state = tf_quantile_normal_fit([1.0, 2.0])
        result = tf_quantile_normal_transform([1.5, None], state)
        assert result[1] is None

    def test_too_few_values(self):
        state = tf_quantile_normal_fit([1.0])
        result = tf_quantile_normal_transform([1.0], state)
        # With n < 3, it still works but may not be accurate
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Registry verification
# ---------------------------------------------------------------------------
class TestRegistryPresence:
    def test_all_fitted_ops_registered(self):
        ops = {o.name: o for o in all_operators()}
        expected = [
            "standardize", "robust_scale", "minmax_scale",
            "winsorize", "quantile_normal",
        ]
        for name in expected:
            assert name in ops, f"{name} not registered"

    def test_fitted_ops_are_l1(self):
        ops = {o.name: o for o in all_operators()}
        for name in ["standardize", "robust_scale", "minmax_scale",
                      "winsorize", "quantile_normal"]:
            assert ops[name].leakage_class.value == "L1"
            assert ops[name].fit_scope.value == "training_fold"
