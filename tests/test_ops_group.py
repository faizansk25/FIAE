"""Tests for group aggregate feature operations (doc 05 entries 73-80)."""

from __future__ import annotations

import math


from fiae.features.ops_group import (
    tf_group_count_fit,
    tf_group_count_transform,
    tf_group_mean_fit,
    tf_group_mean_transform,
    tf_group_std_fit,
    tf_group_std_transform,
    tf_group_min_fit,
    tf_group_min_transform,
    tf_group_max_fit,
    tf_group_max_transform,
    tf_group_median_fit,
    tf_group_median_transform,
    tf_group_nunique_fit,
    tf_group_nunique_transform,
    tf_group_missing_rate_fit,
    tf_group_missing_rate_transform,
)
from fiae.features.registry import all_operators


# ---------------------------------------------------------------------------
# group_count
# ---------------------------------------------------------------------------
class TestGroupCount:
    def test_basic(self):
        state = tf_group_count_fit(keys=["a", "a", "b", "b", "b"])
        result = tf_group_count_transform(["a", "b", "c"], state)
        assert result[0] == 2
        assert result[1] == 3
        assert result[2] == 0

    def test_null_key(self):
        state = tf_group_count_fit(keys=["a", None, "a"])
        result = tf_group_count_transform([None], state)
        assert result[0] is None


# ---------------------------------------------------------------------------
# group_mean
# ---------------------------------------------------------------------------
class TestGroupMean:
    def test_basic(self):
        state = tf_group_mean_fit(
            values=[10.0, 20.0, 30.0],
            keys=["a", "a", "b"],
        )
        result = tf_group_mean_transform(["a", "b", "c"], state)
        assert abs(result[0] - 15.0) < 1e-10
        assert abs(result[1] - 30.0) < 1e-10
        assert result[2] == 0.0

    def test_null_values(self):
        state = tf_group_mean_fit(
            values=[10.0, None, 30.0],
            keys=["a", "a", "a"],
        )
        result = tf_group_mean_transform(["a"], state)
        assert abs(result[0] - 20.0) < 1e-10


# ---------------------------------------------------------------------------
# group_std
# ---------------------------------------------------------------------------
class TestGroupStd:
    def test_basic(self):
        state = tf_group_std_fit(
            values=[1.0, 3.0, 5.0],
            keys=["a", "a", "a"],
        )
        result = tf_group_std_transform(["a"], state)
        assert result[0] > 0
        expected = math.sqrt(8.0 / 3.0)
        assert abs(result[0] - expected) < 1e-10

    def test_single_value(self):
        state = tf_group_std_fit(values=[5.0], keys=["a"])
        result = tf_group_std_transform(["a"], state)
        assert result[0] == 0.0


# ---------------------------------------------------------------------------
# group_min / group_max
# ---------------------------------------------------------------------------
class TestGroupMinMax:
    def test_min(self):
        state = tf_group_min_fit(
            values=[3.0, 1.0, 2.0], keys=["a", "a", "b"]
        )
        result = tf_group_min_transform(["a", "b"], state)
        assert result[0] == 1.0
        assert result[1] == 2.0

    def test_max(self):
        state = tf_group_max_fit(
            values=[3.0, 1.0, 2.0], keys=["a", "a", "b"]
        )
        result = tf_group_max_transform(["a", "b"], state)
        assert result[0] == 3.0
        assert result[1] == 2.0


# ---------------------------------------------------------------------------
# group_median
# ---------------------------------------------------------------------------
class TestGroupMedian:
    def test_odd(self):
        state = tf_group_median_fit(
            values=[3.0, 1.0, 2.0], keys=["a", "a", "a"]
        )
        result = tf_group_median_transform(["a"], state)
        assert result[0] == 2.0

    def test_even(self):
        state = tf_group_median_fit(
            values=[1.0, 2.0, 3.0, 4.0], keys=["a", "a", "a", "a"]
        )
        result = tf_group_median_transform(["a"], state)
        assert result[0] == 2.5


# ---------------------------------------------------------------------------
# group_nunique
# ---------------------------------------------------------------------------
class TestGroupNunique:
    def test_basic(self):
        state = tf_group_nunique_fit(
            values=["x", "y", "x", "z"], keys=["a", "a", "a", "b"]
        )
        result = tf_group_nunique_transform(["a", "b"], state)
        assert result[0] == 2
        assert result[1] == 1

    def test_null_values(self):
        state = tf_group_nunique_fit(
            values=["x", None, "x"], keys=["a", "a", "a"]
        )
        result = tf_group_nunique_transform(["a"], state)
        assert result[0] == 1


# ---------------------------------------------------------------------------
# group_missing_rate
# ---------------------------------------------------------------------------
class TestGroupMissingRate:
    def test_basic(self):
        state = tf_group_missing_rate_fit(
            values=[1.0, None, None, 4.0], keys=["a", "a", "b", "b"]
        )
        result = tf_group_missing_rate_transform(["a", "b"], state)
        assert abs(result[0] - 0.5) < 1e-10
        assert abs(result[1] - 0.5) < 1e-10

    def test_no_missing(self):
        state = tf_group_missing_rate_fit(
            values=[1.0, 2.0, 3.0], keys=["a", "a", "a"]
        )
        result = tf_group_missing_rate_transform(["a"], state)
        assert result[0] == 0.0


# ---------------------------------------------------------------------------
# Registry verification
# ---------------------------------------------------------------------------
class TestRegistryPresence:
    def test_all_group_ops_registered(self):
        ops = {o.name: o for o in all_operators()}
        expected_names = [
            "group_count", "group_mean", "group_std", "group_min",
            "group_max", "group_median", "group_nunique", "group_missing_rate",
        ]
        for name in expected_names:
            assert name in ops, f"{name} not registered"

    def test_group_ops_are_l1(self):
        ops = {o.name: o for o in all_operators()}
        expected_names = [
            "group_count", "group_mean", "group_std", "group_min",
            "group_max", "group_median", "group_nunique", "group_missing_rate",
        ]
        for name in expected_names:
            assert ops[name].leakage_class.value == "L1"
            assert ops[name].fit_scope.value == "training_fold"
            assert ops[name].family == "group aggregate"
