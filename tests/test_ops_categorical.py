"""Tests for categorical feature operations (doc 05 entries 33-43)."""

from __future__ import annotations

import math


from fiae.features.ops_categorical import (
    _cat,
    _RARE_TOKEN,
    tf_category_cross,
    tf_count_encode_fit,
    tf_count_encode_transform,
    tf_frequency_encode_fit,
    tf_frequency_encode_transform,
    tf_hash_encode,
    tf_numeric_to_cat_target,
    tf_ordinal_encode_fit,
    tf_ordinal_encode_transform,
    tf_ordinal_true_scale,
    tf_one_hot_fit,
    tf_one_hot_transform,
    tf_rare_group_fit,
    tf_rare_group_transform,
    tf_target_mean_crossfit,
    tf_woe_crossfit,
)
from fiae.features.registry import all_operators


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
class TestHelpers:
    def test_cat_none(self):
        assert _cat(None) is None

    def test_cat_nan(self):
        assert _cat(float("nan")) is None

    def test_cat_empty(self):
        assert _cat("") is None
        assert _cat("  ") is None

    def test_cat_string(self):
        assert _cat("hello") == "hello"

    def test_cat_number(self):
        assert _cat(42) == "42"


# ---------------------------------------------------------------------------
# L0 operators
# ---------------------------------------------------------------------------
class TestHashEncode:
    def test_basic(self):
        result = tf_hash_encode(["a", "b", "c", "a"], n_buckets=4)
        assert len(result) == 4
        assert result[0] == result[3]  # 'a' hashes same
        assert all(0 <= v < 4 for v in result if v is not None)

    def test_null_preserved(self):
        result = tf_hash_encode(["a", None, "c"])
        assert result[0] is not None
        assert result[1] is None
        assert result[2] is not None

    def test_deterministic(self):
        r1 = tf_hash_encode(["x", "y", "z"], seed=42)
        r2 = tf_hash_encode(["x", "y", "z"], seed=42)
        assert r1 == r2

    def test_different_seeds(self):
        r1 = tf_hash_encode(["x"], seed=0)
        r2 = tf_hash_encode(["x"], seed=1)
        assert r1 != r2


class TestCategoryCross:
    def test_basic(self):
        result = tf_category_cross(["a", "b"], ["x", "y"])
        assert result == ["a|x", "b|y"]

    def test_null_propagation(self):
        result = tf_category_cross(["a", None], ["x", "y"])
        assert result[0] == "a|x"
        assert result[1] is None

    def test_escaping(self):
        result = tf_category_cross(["a|b"], ["x"])
        assert result[0] == "a\\|b|x"


class TestOrdinalTrueScale:
    def test_basic(self):
        result = tf_ordinal_true_scale(
            ["low", "medium", "high", "low"],
            order=["low", "medium", "high"],
        )
        assert result == [0, 1, 2, 0]

    def test_unknown_category(self):
        result = tf_ordinal_true_scale(["unknown"], order=["a", "b"])
        assert result == [None]

    def test_null(self):
        result = tf_ordinal_true_scale([None], order=["a"])
        assert result == [None]


# ---------------------------------------------------------------------------
# L1 fitted-state operators
# ---------------------------------------------------------------------------
class TestOneHot:
    def test_fit_and_transform(self):
        state = tf_one_hot_fit(["a", "b", "a", "c"])
        assert set(state["categories"]) == {"a", "b", "c"}
        result = tf_one_hot_transform(["a", "c", "b", "d"], state)
        assert len(result) == 4
        # 'a' is index 0 (most frequent)
        assert result[0][0] is True
        # 'c' is index 2
        assert result[1][2] is True
        # 'd' is unknown -> all False
        assert all(v is False for v in result[3])


class TestOrdinalEncode:
    def test_fit_and_transform(self):
        state = tf_ordinal_encode_fit(["cat", "dog", "cat", "bird"])
        assert state["mapping"]["cat"] == 0  # most frequent
        assert state["mapping"]["dog"] == 1
        assert state["mapping"]["bird"] == 2
        result = tf_ordinal_encode_transform(["cat", "unknown"], state)
        assert result[0] == 0
        assert result[1] == -1  # unknown


class TestFrequencyEncode:
    def test_fit_and_transform(self):
        state = tf_frequency_encode_fit(["a", "b", "a", "a"])
        assert abs(state["frequency_map"]["a"] - 0.75) < 1e-10
        assert abs(state["frequency_map"]["b"] - 0.25) < 1e-10
        result = tf_frequency_encode_transform(["a", "c"], state)
        assert abs(result[0] - 0.75) < 1e-10
        assert result[1] == 0.0  # unknown


class TestCountEncode:
    def test_fit_and_transform(self):
        state = tf_count_encode_fit(["a", "b", "a", "a", "b"])
        assert state["count_map"]["a"] == 3
        assert state["count_map"]["b"] == 2
        result = tf_count_encode_transform(["a", "c"], state)
        assert result[0] == 3
        assert result[1] == 0  # unknown


class TestRareGroup:
    def test_fit_and_transform(self):
        values = ["a"] * 10 + ["b"] * 3 + ["c"] * 1
        state = tf_rare_group_fit(values, min_count=5)
        assert "a" in state["retained"]
        assert "b" not in state["retained"]
        result = tf_rare_group_transform(["a", "b", "c", "x"], state)
        assert result[0] == "a"
        assert result[1] == _RARE_TOKEN
        assert result[2] == _RARE_TOKEN
        assert result[3] == _RARE_TOKEN


# ---------------------------------------------------------------------------
# L2 target-aware operators
# ---------------------------------------------------------------------------
class TestTargetMeanCrossfit:
    def test_basic(self):
        cats = ["a", "a", "b", "b"]
        target = [1.0, 1.0, 0.0, 0.0]
        result = tf_target_mean_crossfit(cats, target, smoothing=1.0)
        assert len(result) == 4
        # 'a' mean=1.0, 'b' mean=0.0, global=0.5
        # Smoothed 'a': (2*1.0 + 1.0*0.5) / (2+1.0) = 2.5/3 ≈ 0.833
        assert abs(result[0] - 0.833) < 0.01
        # Smoothed 'b': (2*0.0 + 1.0*0.5) / (2+1.0) = 0.5/3 ≈ 0.167
        assert abs(result[2] - 0.167) < 0.01

    def test_null(self):
        result = tf_target_mean_crossfit([None], [1.0])
        assert result == [None]


class TestWoeCrossfit:
    def test_basic(self):
        cats = ["a", "a", "b", "b"]
        target = [1.0, 1.0, 0.0, 0.0]
        result = tf_woe_crossfit(cats, target, smoothing=1.0)
        assert len(result) == 4
        # All values should be finite
        assert all(math.isfinite(v) for v in result if v is not None)
        # 'a' has more positives -> positive WoE
        assert result[0] > 0
        # 'b' has more negatives -> negative WoE
        assert result[2] < 0

    def test_null(self):
        result = tf_woe_crossfit([None], [1.0])
        assert result == [None]


class TestNumericToCatTarget:
    def test_basic(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        target = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        result = tf_numeric_to_cat_target(values, target, n_bins=3)
        assert len(result) == 10
        # All non-None
        assert all(v is not None for v in result)

    def test_too_few_values(self):
        result = tf_numeric_to_cat_target([1.0], [1.0], n_bins=5)
        assert result == [None]


# ---------------------------------------------------------------------------
# Registry verification
# ---------------------------------------------------------------------------
class TestRegistryPresence:
    def test_all_new_ops_registered(self):
        ops = {o.name: o for o in all_operators()}
        expected = [
            "hash_encode", "category_cross", "ordinal_true_scale",
            "one_hot", "ordinal_encode", "frequency_encode", "count_encode",
            "rare_group", "target_mean_crossfit", "woe_crossfit",
            "numeric_to_cat_target",
        ]
        for name in expected:
            assert name in ops, f"{name} not registered"

    def test_new_ops_families(self):
        ops = {o.name: o for o in all_operators()}
        assert ops["hash_encode"].family == "categorical"
        assert ops["one_hot"].family == "categorical"
        assert ops["target_mean_crossfit"].family == "target-aware categorical"
        assert ops["category_cross"].family == "categorical interaction"
