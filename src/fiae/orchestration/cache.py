"""Trial result cache (doc 08).

Memoizes completed trial results to avoid redundant computation.
Keyed by (model_family, hyperparameters_hash, feature_set_hash).

Normative source: doc 08 section "Cache".
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional

from ..contracts import TrialResult, TrialStatus


@dataclass
class TrialCache:
    """Cache for completed trial results (doc 08)."""

    max_size: int = 1000
    _entries: dict[str, TrialResult] = field(default_factory=dict, init=False)
    _access_order: list[str] = field(default_factory=list, init=False)

    def _key(self, model_family: str, params: dict, features: list[str]) -> str:
        """Generate cache key from trial characteristics."""
        data = {
            "family": model_family,
            "params": params,
            "features": sorted(features),
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def get(self, model_family: str, params: dict, features: list[str]) -> Optional[TrialResult]:
        """Look up a cached result."""
        key = self._key(model_family, params, features)
        if key in self._entries:
            # Move to end (most recently used)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return self._entries[key]
        return None

    def put(self, model_family: str, params: dict, features: list[str],
            result: TrialResult) -> None:
        """Store a result in the cache."""
        if result.status != TrialStatus.COMPLETED:
            return  # only cache completed results

        key = self._key(model_family, params, features)

        # Evict oldest if at capacity
        if len(self._entries) >= self.max_size and key not in self._entries:
            oldest = self._access_order.pop(0)
            self._entries.pop(oldest, None)

        self._entries[key] = result
        if key not in self._access_order:
            self._access_order.append(key)

    def contains(self, model_family: str, params: dict, features: list[str]) -> bool:
        key = self._key(model_family, params, features)
        return key in self._entries

    def size(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self._access_order.clear()

    def hit_rate(self) -> float:
        """Return cache hit rate (for monitoring)."""
        return 0.0  # requires access tracking
