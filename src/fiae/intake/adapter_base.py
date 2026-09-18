"""Base adapter class for all FIAE data source adapters.

Every adapter inherits from BaseAdapter which provides:
- Shared batch building utilities
- Default implementations for optional protocol methods
- Deterministic fingerprinting helpers
- Type coercion for columnar data

Concrete adapters implement:
- source_id(), scan(), sample(), fingerprint_material()
- _connect(), _disconnect() for resource management

Normative source: doc 02 DataSourceAdapter protocol.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterator, Optional

from .base import RowBatch, SamplePlan


class BaseAdapter:
    """Base class for all FIAE data source adapters.

    Provides shared utilities so concrete adapters only need to implement
    the core data access methods.
    """

    def __init__(self, source_type: str, **kwargs: Any) -> None:
        self._source_type = source_type
        self._kwargs = kwargs
        self._connected = False
        self._batch_rows = kwargs.get("batch_rows", 2048)

    # ── Protocol methods (subclasses implement) ──────────────────────────

    def source_id(self) -> str:
        """Unique identifier for this source."""
        raise NotImplementedError

    def schema_hint(self) -> Optional[dict[str, Any]]:
        """Optional schema hint (None for self-describing formats)."""
        return None

    def estimate_rows(self) -> Optional[int]:
        """Estimate total row count. None if unknown."""
        return None

    def estimate_bytes(self) -> Optional[int]:
        """Estimate total byte size. None if unknown."""
        return None

    def scan(
        self,
        projection: Optional[list[str]] = None,
        predicate: Optional[Any] = None,
    ) -> Iterator[RowBatch]:
        """Full sequential scan. Subclasses MUST implement."""
        raise NotImplementedError

    def sample(self, plan: SamplePlan) -> Iterator[RowBatch]:
        """Deterministic multi-region sample. Default uses scan with row limiting."""
        count = 0
        max_rows = plan.max_total_bytes // 200  # rough estimate
        for batch in self.scan():
            yield batch
            count += batch.n_rows
            if count >= max_rows:
                break

    def supports_seek_sampling(self) -> bool:
        return False

    def supports_pushdown(self) -> bool:
        return False

    def fingerprint_material(self) -> bytes:
        """Deterministic content fingerprint. Default: hash of source_id."""
        return hashlib.sha256(self.source_id().encode()).digest()

    # ── Shared utilities ─────────────────────────────────────────────────

    @staticmethod
    def _to_row_batches(
        columns: dict[str, list],
        batch_rows: int = 2048,
    ) -> Iterator[RowBatch]:
        """Convert columnar dict to batches of RowBatch."""
        if not columns:
            return
        n = len(next(iter(columns.values())))
        for start in range(0, n, batch_rows):
            end = min(start + batch_rows, n)
            batch_cols = {k: v[start:end] for k, v in columns.items()}
            yield RowBatch(columns=batch_cols)

    @staticmethod
    def _coerce_column(values: list) -> list:
        """Coerce a column of values to string representation with None for nulls."""
        out = []
        for v in values:
            if v is None or (isinstance(v, float) and (v != v)):
                out.append(None)
            else:
                out.append(str(v))
        return out

    @staticmethod
    def _columns_to_batches(
        data: dict[str, list],
        batch_rows: int = 2048,
    ) -> Iterator[RowBatch]:
        """Generic columnar-to-batch conversion."""
        if not data:
            return
        n = len(next(iter(data.values())))
        for start in range(0, n, batch_rows):
            end = min(start + batch_rows, n)
            cols = {}
            for k, v in data.items():
                cols[k] = v[start:end] if isinstance(v, list) else [v] * (end - start)
            yield RowBatch(columns=cols)

    def _table_exists(self, table_name: str) -> bool:
        """Override in SQL adapters to check if table exists."""
        return True

    def _count_rows(self) -> Optional[int]:
        """Override to provide fast row count."""
        return None
