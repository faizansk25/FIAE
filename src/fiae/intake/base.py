"""Source adapter contract and row primitives (doc 02).

The core operates on a DataSourceAdapter, not direct CSV assumptions (FR-001).
RowBatch is columnar; values arrive as raw strings plus explicit None for true
nulls so downstream missingness accounting can separate null vs empty string.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Protocol, runtime_checkable


@dataclass
class RowBatch:
    """Columnar batch of raw (unparsed) values."""

    columns: dict[str, list[Any]] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        if not self.columns:
            return 0
        return len(next(iter(self.columns.values())))


class SampleRegion(str, enum.Enum):
    HEAD = "head"
    MIDDLE = "middle"
    TAIL = "tail"
    RANDOM = "random"


@dataclass
class SamplePlan:
    """Deterministic representative fast sample (doc 02).

    head block + middle block(s) + tail block + deterministic pseudo-random blocks.
    """

    block_bytes: int = 262_144
    include_head: bool = True
    n_middle: int = 1
    include_tail: bool = True
    n_random: int = 1
    random_seed: int = 0
    max_total_bytes: int = 1_048_576


@runtime_checkable
class DataSourceAdapter(Protocol):
    """Normative source adapter contract (doc 02)."""

    def source_id(self) -> str: ...

    def schema_hint(self) -> Optional[dict[str, Any]]: ...

    def estimate_rows(self) -> Optional[int]: ...

    def estimate_bytes(self) -> Optional[int]: ...

    def sample(self, plan: SamplePlan) -> Iterable[RowBatch]: ...

    def scan(
        self,
        projection: Optional[list[str]] = None,
        predicate: Optional[Any] = None,
    ) -> Iterable[RowBatch]: ...

    def supports_seek_sampling(self) -> bool: ...

    def supports_pushdown(self) -> bool: ...

    def fingerprint_material(self) -> bytes: ...
