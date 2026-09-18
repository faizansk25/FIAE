"""Streaming statistics for bounded profiling (doc 02).

- Welford online variance exactly as specified in doc 02.
- Cardinality: exact set up to a cap, then a bounded KMV-style sketch with
  estimate / exact flag / configuration (FR-002, doc 02).
- Missingness separates true null, empty string, whitespace, configured
  missing tokens, and suspicious numeric sentinels (warnings until confirmed).
- No raw values are retained; only counts and safe statistics (doc 10 privacy).
"""

from __future__ import annotations

import hashlib
import heapq
import math
from typing import Optional

SUSPICIOUS_SENTINELS: frozenset[str] = frozenset(
    {"-999", "-9999", "9999", "99999", "-1", "1900-01-01", "1970-01-01"}
)


class Welford:
    """Online mean/variance per doc 02."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def merge(self, other: "Welford") -> None:
        """Parallel/streaming merge (Chan et al. pairwise formula)."""
        if other.n == 0:
            return
        if self.n == 0:
            self.n, self.mean, self.m2 = other.n, other.mean, other.m2
            return
        total = self.n + other.n
        delta = other.mean - self.mean
        self.mean += delta * other.n / total
        self.m2 += other.m2 + delta * delta * self.n * other.n / total
        self.n = total

    @property
    def variance(self) -> Optional[float]:
        if self.n < 2:
            return None
        return self.m2 / (self.n - 1)

    @property
    def stddev(self) -> Optional[float]:
        v = self.variance
        return None if v is None else math.sqrt(max(v, 0.0))


def stable_hash64(value: str) -> int:
    """Deterministic across processes (unlike built-in hash())."""
    return int.from_bytes(
        hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big"
    )


class CardinalitySketch:
    """Exact set up to cap, then KMV-style bounded sketch (doc 02)."""

    def __init__(self, exact_cap: int = 4096, k: int = 512) -> None:
        self.exact_cap = exact_cap
        self.k = k
        self._exact: set[int] = set()
        self._heap: list[int] = []  # max-heap (negated) of k smallest hashes
        self._overflow = False

    def add(self, value: str) -> None:
        h = stable_hash64(value)
        if not self._overflow:
            self._exact.add(h)
            if len(self._exact) > self.exact_cap:
                smallest = heapq.nsmallest(self.k, self._exact)
                self._heap = [-v for v in smallest]
                heapq.heapify(self._heap)
                self._exact.clear()
                self._overflow = True
            return
        if h < -self._heap[0]:
            heapq.heapreplace(self._heap, -h)

    def estimate(self) -> tuple[int, bool]:
        """Returns (distinct_estimate, exact_flag)."""
        if not self._overflow:
            return len(self._exact), True
        kth = -self._heap[0]
        p = kth / 2**64
        if p <= 0:
            return self.k, False
        est = round((self.k - 1) / p)
        return max(est, self.k), False

    def merge(self, other: "CardinalitySketch") -> None:
        """Merge another sketch; result is a valid sketch of the union."""
        if not self._overflow and not other._overflow:
            self._exact |= other._exact
            if len(self._exact) > self.exact_cap:
                smallest = heapq.nsmallest(self.k, self._exact)
                self._heap = [-v for v in smallest]
                heapq.heapify(self._heap)
                self._exact.clear()
                self._overflow = True
            return
        # At least one side overflowed: keep the k smallest hashes of the union.
        candidates: set[int] = set()
        if self._overflow:
            candidates.update(-v for v in self._heap)
        else:
            candidates.update(self._exact)
        if other._overflow:
            candidates.update(-v for v in other._heap)
        else:
            candidates.update(other._exact)
        smallest = heapq.nsmallest(self.k, candidates)
        self._heap = [-v for v in smallest]
        heapq.heapify(self._heap)
        self._exact.clear()
        self._overflow = True


class Missingness:
    """Distinguishes null / empty / whitespace / token / sentinel (doc 02)."""

    def __init__(self, missing_tokens: frozenset[str] = frozenset()) -> None:
        self.true_null = 0
        self.empty_string = 0
        self.whitespace = 0
        self.token = 0
        self.sentinel = 0
        self.n_observed = 0
        self._missing_tokens = missing_tokens

    def update(self, value: Optional[str]) -> None:
        self.n_observed += 1
        if value is None:
            self.true_null += 1
            return
        if value == "":
            self.empty_string += 1
            return
        if value.strip() == "":
            self.whitespace += 1
            return
        if value in self._missing_tokens:
            self.token += 1
            return
        if value in SUSPICIOUS_SENTINELS:
            # Suspicious sentinels are warnings until confirmed (doc 02).
            self.sentinel += 1

    def merge(self, other: "Missingness") -> None:
        """Merge counters from another accumulator (same token config)."""
        self.true_null += other.true_null
        self.empty_string += other.empty_string
        self.whitespace += other.whitespace
        self.token += other.token
        self.sentinel += other.sentinel
        self.n_observed += other.n_observed

    @property
    def missing_total(self) -> int:
        return self.true_null + self.empty_string + self.whitespace + self.token

    def to_dict(self) -> dict[str, int]:
        return {
            "true_null": self.true_null,
            "empty_string": self.empty_string,
            "whitespace": self.whitespace,
            "missing_token": self.token,
            "suspicious_sentinel": self.sentinel,
            "n_observed": self.n_observed,
        }


class NumericAccumulator:
    """Streaming numeric statistics (doc 02)."""

    def __init__(self) -> None:
        self.welford = Welford()
        self.minimum: Optional[float] = None
        self.maximum: Optional[float] = None
        self.nan_count = 0
        self.inf_count = 0
        self.zero_count = 0
        self.negative_count = 0
        self.n_numeric = 0

    def update(self, x: float) -> None:
        self.n_numeric += 1
        if math.isnan(x):
            self.nan_count += 1
            return
        if math.isinf(x):
            self.inf_count += 1
            return
        self.welford.update(x)
        self.minimum = x if self.minimum is None else min(self.minimum, x)
        self.maximum = x if self.maximum is None else max(self.maximum, x)
        if x == 0:
            self.zero_count += 1
        if x < 0:
            self.negative_count += 1

    def merge(self, other: "NumericAccumulator") -> None:
        """Merge another numeric accumulator."""
        self.welford.merge(other.welford)
        if other.minimum is not None:
            self.minimum = other.minimum if self.minimum is None else min(self.minimum, other.minimum)
        if other.maximum is not None:
            self.maximum = other.maximum if self.maximum is None else max(self.maximum, other.maximum)
        self.nan_count += other.nan_count
        self.inf_count += other.inf_count
        self.zero_count += other.zero_count
        self.negative_count += other.negative_count
        self.n_numeric += other.n_numeric

    def to_dict(self) -> dict[str, Optional[float]]:
        return {
            "count": self.n_numeric,
            "mean": self.welford.mean if self.n_numeric else None,
            "stddev": self.welford.stddev,
            "min": self.minimum,
            "max": self.maximum,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "zero_count": self.zero_count,
            "negative_count": self.negative_count,
        }
