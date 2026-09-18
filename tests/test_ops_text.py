"""Tests for text feature operations (doc 05 entries 81-88)."""

from __future__ import annotations


from fiae.features.ops_text import (
    tf_text_length_chars,
    tf_text_length_tokens,
    tf_text_digit_ratio,
    tf_text_upper_ratio,
    tf_text_punctuation_ratio,
    tf_text_hashing,
    tf_tfidf_fit,
    tf_tfidf_transform,
)
from fiae.features.registry import all_operators


# ---------------------------------------------------------------------------
# text_length_chars
# ---------------------------------------------------------------------------
class TestTextLengthChars:
    def test_basic(self):
        result = tf_text_length_chars(["hello", "world!", ""])
        assert result == [5, 6, 0]

    def test_null(self):
        result = tf_text_length_chars([None, "ab"])
        assert result[0] is None
        assert result[1] == 2


# ---------------------------------------------------------------------------
# text_length_tokens
# ---------------------------------------------------------------------------
class TestTextLengthTokens:
    def test_basic(self):
        result = tf_text_length_tokens(["hello world", "one two three", "single"])
        assert result == [2, 3, 1]

    def test_empty(self):
        result = tf_text_length_tokens([""])
        assert result == [0]


# ---------------------------------------------------------------------------
# text_digit_ratio
# ---------------------------------------------------------------------------
class TestTextDigitRatio:
    def test_all_digits(self):
        result = tf_text_digit_ratio(["12345"])
        assert abs(result[0] - 1.0) < 1e-10

    def test_no_digits(self):
        result = tf_text_digit_ratio(["hello"])
        assert result[0] == 0.0

    def test_mixed(self):
        result = tf_text_digit_ratio(["abc123"])
        assert abs(result[0] - 0.5) < 1e-10

    def test_null(self):
        result = tf_text_digit_ratio([None])
        assert result[0] is None


# ---------------------------------------------------------------------------
# text_upper_ratio
# ---------------------------------------------------------------------------
class TestTextUpperRatio:
    def test_all_upper(self):
        result = tf_text_upper_ratio(["HELLO"])
        assert result[0] == 1.0

    def test_all_lower(self):
        result = tf_text_upper_ratio(["hello"])
        assert result[0] == 0.0

    def test_mixed(self):
        result = tf_text_upper_ratio(["HeLLo"])
        # H, L, L are uppercase = 3/5
        assert abs(result[0] - 0.6) < 1e-10

    def test_no_letters(self):
        result = tf_text_upper_ratio(["12345"])
        assert result[0] == 0.0


# ---------------------------------------------------------------------------
# text_punctuation_ratio
# ---------------------------------------------------------------------------
class TestTextPunctuationRatio:
    def test_all_punct(self):
        result = tf_text_punctuation_ratio(["!@#$%"])
        assert abs(result[0] - 1.0) < 1e-10

    def test_no_punct(self):
        result = tf_text_punctuation_ratio(["hello"])
        assert result[0] == 0.0

    def test_mixed(self):
        result = tf_text_punctuation_ratio(["hi!"])
        # 1 punct / 3 total
        assert abs(result[0] - 1.0 / 3.0) < 1e-10


# ---------------------------------------------------------------------------
# text_hashing
# ---------------------------------------------------------------------------
class TestTextHashing:
    def test_output_shape(self):
        result = tf_text_hashing(["hello world"], n_features=8)
        assert len(result) == 1
        assert len(result[0]) == 8

    def test_null_text(self):
        result = tf_text_hashing([None], n_features=4)
        assert result[0] == [0] * 4

    def test_deterministic(self):
        r1 = tf_text_hashing(["test"], n_features=8, seed=42)
        r2 = tf_text_hashing(["test"], n_features=8, seed=42)
        assert r1 == r2

    def test_different_texts(self):
        r1 = tf_text_hashing(["hello"], n_features=8)
        r2 = tf_text_hashing(["world"], n_features=8)
        # Different texts should produce different hashes
        assert r1 != r2


# ---------------------------------------------------------------------------
# tfidf
# ---------------------------------------------------------------------------
class TestTfidf:
    def test_fit_and_transform(self):
        docs = ["hello world", "hello there", "world peace"]
        state = tf_tfidf_fit(docs, max_features=10)
        assert "hello" in state["vocab"]
        assert "world" in state["vocab"]
        result = tf_tfidf_transform(["hello world"], state)
        assert len(result) == 1
        assert len(result[0]) == state["n_features"]
        # 'hello world' has 2 tokens -> TF=0.5 each * IDF
        assert any(v > 0 for v in result[0])

    def test_null_text(self):
        state = tf_tfidf_fit(["hello world"])
        result = tf_tfidf_transform([None], state)
        assert result[0] == [0.0] * state["n_features"]

    def test_unknown_token(self):
        state = tf_tfidf_fit(["hello world"])
        result = tf_tfidf_transform(["unknown_token"], state)
        assert all(v == 0.0 for v in result[0])


# ---------------------------------------------------------------------------
# Registry verification
# ---------------------------------------------------------------------------
class TestRegistryPresence:
    def test_all_text_ops_registered(self):
        ops = {o.name: o for o in all_operators()}
        expected = [
            "text_length_chars", "text_length_tokens", "text_digit_ratio",
            "text_upper_ratio", "text_punctuation_ratio", "text_hashing",
            "tfidf_word", "tfidf_char",
        ]
        for name in expected:
            assert name in ops, f"{name} not registered"

    def test_text_l0_ops(self):
        ops = {o.name: o for o in all_operators()}
        for name in ["text_length_chars", "text_digit_ratio", "text_hashing"]:
            assert ops[name].family == "text"
            assert ops[name].leakage_class.value == "L0"
            assert ops[name].fit_scope.value == "none"

    def test_text_l1_ops(self):
        ops = {o.name: o for o in all_operators()}
        for name in ["tfidf_word", "tfidf_char"]:
            assert ops[name].family == "text representation"
            assert ops[name].leakage_class.value == "L1"
            assert ops[name].fit_scope.value == "training_fold"
