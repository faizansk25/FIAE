"""Text feature operations (doc 05 entries 81-88).

Simple L0 row-wise text metrics plus L1 bag-of-words / hash-vectorizer
representations.  All transforms are pure functions on string lists.

Operators:
- text_length_chars:    character count
- text_length_tokens:   whitespace-token count
- text_digit_ratio:     fraction of digit characters
- text_upper_ratio:     fraction of uppercase letters
- text_punctuation_ratio: fraction of punctuation characters
- tfidf_word:           word-level TF-IDF (L1 fitted)
- tfidf_char:           character n-gram TF-IDF (L1 fitted)
- text_hashing:         feature hashing of tokens (L0, fixed seed)
"""

from __future__ import annotations

import hashlib
import math
import re
import string
from collections import Counter
from typing import Any, Optional

from ..contracts import FitScope, LeakageClass, TargetPermission
from .registry import FeatureOperator, register


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _text(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return str(v)


def _word_tokenize(s: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\w+", s.lower())


def _char_ngrams(s: str, n: int = 3) -> list[str]:
    """Character n-grams from lowercase string."""
    s_low = s.lower()
    return [s_low[i : i + n] for i in range(max(0, len(s_low) - n + 1))]


# ---------------------------------------------------------------------------
# L0 text metrics (entries 81-85)
# ---------------------------------------------------------------------------
def tf_text_length_chars(values: list, **_: Any) -> list:
    out: list[Optional[int]] = []
    for v in values:
        t = _text(v)
        out.append(len(t) if t is not None else None)
    return out


def tf_text_length_tokens(values: list, **_: Any) -> list:
    out: list[Optional[int]] = []
    for v in values:
        t = _text(v)
        out.append(len(_word_tokenize(t)) if t is not None else None)
    return out


def tf_text_digit_ratio(values: list, **_: Any) -> list:
    out: list[Optional[float]] = []
    for v in values:
        t = _text(v)
        if t is None:
            out.append(None)
        else:
            length = max(len(t), 1)
            digits = sum(c.isdigit() for c in t)
            out.append(digits / length)
    return out


def tf_text_upper_ratio(values: list, **_: Any) -> list:
    out: list[Optional[float]] = []
    for v in values:
        t = _text(v)
        if t is None:
            out.append(None)
        else:
            letters = [c for c in t if c.isalpha()]
            if not letters:
                out.append(0.0)
            else:
                upper = sum(c.isupper() for c in letters)
                out.append(upper / len(letters))
    return out


def tf_text_punctuation_ratio(values: list, **_: Any) -> list:
    out: list[Optional[float]] = []
    for v in values:
        t = _text(v)
        if t is None:
            out.append(None)
        else:
            length = max(len(t), 1)
            punct = sum(c in string.punctuation for c in t)
            out.append(punct / length)
    return out


# ---------------------------------------------------------------------------
# L0 text feature hashing (entry 88)
# ---------------------------------------------------------------------------
def tf_text_hashing(
    values: list, n_features: int = 16, ngram_range: int = 1, seed: int = 0, **_: Any
) -> list:
    """Feature hashing of text tokens / character n-grams (L0, no fit).

    Returns list of lists, each of length n_features.
    """
    out: list = []
    for v in values:
        t = _text(v)
        if t is None:
            out.append([0] * n_features)
            continue
        if ngram_range <= 1:
            tokens = _word_tokenize(t)
        else:
            tokens = _char_ngrams(t, ngram_range)
        vec = [0] * n_features
        for tok in tokens:
            # md5 (usedforsecurity=False) keeps buckets stable across processes;
            # built-in hash() is salted per process and would break determinism.
            digest = hashlib.md5(
                f"{seed}:{tok}".encode(), usedforsecurity=False
            ).hexdigest()
            h = int(digest, 16) % n_features
            # Sign hash for unbiased projection
            sign_digest = hashlib.md5(
                f"s:{seed}:{tok}".encode(), usedforsecurity=False
            ).hexdigest()
            sign = 1 if int(sign_digest, 16) % 2 == 0 else -1
            vec[h] += sign
        out.append(vec)
    return out


# ---------------------------------------------------------------------------
# L1 fitted TF-IDF (entries 86-87)
# ---------------------------------------------------------------------------
def tf_tfidf_fit(
    values: list, max_features: int = 100, ngram_range: int = 1, **_: Any
) -> dict:
    """Fit vocabulary and IDF weights on training text.

    Returns state dict with vocab (token -> index), idf (list), and config.
    """
    # Count document frequencies
    doc_freq: Counter[str] = Counter()
    total_docs = 0
    for v in values:
        t = _text(v)
        if t is None:
            continue
        total_docs += 1
        if ngram_range <= 1:
            tokens = set(_word_tokenize(t))
        else:
            tokens = set(_char_ngrams(t, ngram_range))
        doc_freq.update(tokens)

    # Select top features by document frequency (moderate frequency preferred)
    if ngram_range <= 1:
        all_tokens = sorted(doc_freq.keys())
    else:
        # For char n-grams, prefer less frequent but still present
        all_tokens = sorted(doc_freq.keys())

    # Take top max_features by DF
    top = sorted(all_tokens, key=lambda t: doc_freq[t], reverse=True)[:max_features]

    vocab = {tok: i for i, tok in enumerate(top)}
    idf = []
    for tok in top:
        df = doc_freq[tok]
        idf.append(math.log((1 + total_docs) / (1 + df)) + 1.0)

    return {
        "vocab": vocab,
        "idf": idf,
        "n_features": len(vocab),
        "ngram_range": ngram_range,
    }


def tf_tfidf_transform(values: list, state: dict, **_: Any) -> list:
    """Transform text to TF-IDF vectors using fitted vocabulary."""
    vocab = state.get("vocab", {})
    idf = state.get("idf", [])
    ngram_range = state.get("ngram_range", 1)
    n = len(vocab)
    if n == 0:
        return [[] for _ in values]

    out: list = []
    for v in values:
        t = _text(v)
        if t is None:
            out.append([0.0] * n)
            continue
        if ngram_range <= 1:
            tokens = _word_tokenize(t)
        else:
            tokens = _char_ngrams(t, ngram_range)
        if not tokens:
            out.append([0.0] * n)
            continue
        tf_count: Counter[str] = Counter(tokens)
        vec = [0.0] * n
        for tok, count in tf_count.items():
            if tok in vocab:
                idx = vocab[tok]
                vec[idx] = (count / len(tokens)) * idf[idx]
        out.append(vec)
    return out


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------
def _text_op(
    name: str, purpose: str, tf, trigger: str, rejects: tuple, tests: tuple,
    output_type: str = "numeric", fit_scope: FitScope = FitScope.NONE,
    leakage_class: LeakageClass = LeakageClass.L0,
) -> None:
    register(
        FeatureOperator(
            name=name,
            family="text",
            arity="unary",
            input_types=("free_text",),
            output_type=output_type,
            purpose=purpose,
            preconditions=("text or coerceable input",),
            fit_scope=fit_scope,
            null_policy="missing -> 0",
            leakage_class=leakage_class,
            target_permission=TargetPermission.P0_NONE,
            cost_shape="O(total chars)",
            generation_trigger=trigger,
            rejection_conditions=rejects,
            validation="incremental CV" if fit_scope == FitScope.NONE else "inside-fold",
            inference_requirement="text available" if fit_scope == FitScope.NONE else "fitted state",
            transform=tf,
            mandatory_tests=tests,
        )
    )


# ---------------------------------------------------------------------------
# L0 registrations (entries 81-85, 88)
# ---------------------------------------------------------------------------
_text_op(
    "text_length_chars", "character count of text field",
    tf_text_length_chars, "text exists",
    ("no variation",), ("unicode length policy",),
)

_text_op(
    "text_length_tokens", "whitespace-token count of text field",
    tf_text_length_tokens, "text exists",
    ("cost", "no variation"), ("tokenizer reproducibility",),
)

_text_op(
    "text_digit_ratio", "fraction of digit characters in text",
    tf_text_digit_ratio, "codes/messages",
    ("no variation",), ("empty string handling",),
)

_text_op(
    "text_upper_ratio", "fraction of uppercase letters in text",
    tf_text_upper_ratio, "style signal",
    ("no variation",), ("unicode case",),
)

_text_op(
    "text_punctuation_ratio", "fraction of punctuation characters in text",
    tf_text_punctuation_ratio, "messages/logs",
    ("no variation",), ("unicode punctuation",),
)

_text_op(
    "text_hashing", "feature hashing of text tokens",
    tf_text_hashing, "large text vocabulary",
    ("collision", "interpretability"), ("determinism",),
    output_type="numeric",
)


# ---------------------------------------------------------------------------
# L1 registrations (entries 86-87)
# ---------------------------------------------------------------------------
register(
    FeatureOperator(
        name="tfidf_word",
        family="text representation",
        arity="unary",
        input_types=("free_text",),
        output_type="sparse_numeric",
        purpose="word-level TF-IDF bag-of-words",
        preconditions=("enough text support",),
        fit_scope=FitScope.TRAINING_FOLD,
        null_policy="unknown tokens ignored",
        leakage_class=LeakageClass.L1,
        target_permission=TargetPermission.P0_NONE,
        cost_shape="O(tokens) + sparse matrix",
        generation_trigger="predictive text and memory allows",
        rejection_conditions=("vocab too large", "memory limit"),
        validation="inside-fold",
        inference_requirement="stored vocab/idf",
        transform=tf_tfidf_transform,
        mandatory_tests=("fold-only vocab",),
    )
)

register(
    FeatureOperator(
        name="tfidf_char",
        family="text representation",
        arity="unary",
        input_types=("free_text",),
        output_type="sparse_numeric",
        purpose="character n-gram TF-IDF subword representation",
        preconditions=("enough text",),
        fit_scope=FitScope.TRAINING_FOLD,
        null_policy="unknown ngrams ignored",
        leakage_class=LeakageClass.L1,
        target_permission=TargetPermission.P0_NONE,
        cost_shape="high",
        generation_trigger="misspellings/codes",
        rejection_conditions=("cost", "memory"),
        validation="inside-fold",
        inference_requirement="stored vocab/idf",
        transform=tf_tfidf_transform,
        mandatory_tests=("fold-only vocab",),
    )
)
