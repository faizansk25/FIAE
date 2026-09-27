"""SQLite-backed persistence for the Experience Store (doc 06).

Cases are stored in a local SQLite database with JSON-serialized fields.
Large artifacts remain as files, not database blobs.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .case import CaseRecord


# -----------------------------------------------------------------------------
# Schema
# -----------------------------------------------------------------------------
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cases (
    identity_hash TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    dataset_fingerprint TEXT NOT NULL,
    schema_fingerprint TEXT NOT NULL,
    engine_commit TEXT DEFAULT 'dev',
    split_fingerprint TEXT,
    seed INTEGER,
    context_json TEXT NOT NULL,
    action_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    cost_json TEXT NOT NULL,
    diagnosis_json TEXT NOT NULL,
    metadata_json TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_cases_case_id ON cases(case_id);
CREATE INDEX IF NOT EXISTS idx_cases_dataset ON cases(dataset_fingerprint);
CREATE INDEX IF NOT EXISTS idx_cases_schema ON cases(schema_fingerprint);
"""


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
class StoreConfig:
    """Configuration for the ExperienceStore."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            db_path = str(Path.cwd() / ".fiae" / "experience.sqlite")
        self.db_path = db_path


# -----------------------------------------------------------------------------
# Store
# -----------------------------------------------------------------------------
class ExperienceStore:
    """Thread-safe SQLite-backed store for CaseRecords."""

    def __init__(self, config: Optional[StoreConfig] = None) -> None:
        self.config = config or StoreConfig()
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            db_dir = os.path.dirname(self.config.db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
            self._conn = sqlite3.connect(self.config.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_db(self) -> None:
        conn = self._get_conn()
        with self._lock:
            conn.executescript(_SCHEMA_SQL)
            conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            with self._lock:
                self._conn.close()
            self._conn = None

    def write_case(self, case: CaseRecord) -> str:
        """Persist a CaseRecord. Returns the identity hash.

        Storage key is the identity hash, so records that differ in engine
        version, schema fingerprint, or hyperparameters are stored side by
        side instead of overwriting each other (record versioning, doc 06).
        Re-writing an identical record is a no-op (deduplication).
        """
        identity = case.identity_hash()
        conn = self._get_conn()
        with self._lock:
            conn.execute(
                """
                INSERT OR REPLACE INTO cases
                (identity_hash, case_id, dataset_fingerprint, schema_fingerprint,
                 engine_commit, split_fingerprint, seed,
                 context_json, action_json, result_json, cost_json,
                 diagnosis_json, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identity,  # storage key: versioned record identity
                    case.case_id,
                    case.dataset_fingerprint,
                    case.schema_fingerprint,
                    case.engine_commit,
                    case.split_fingerprint,
                    case.seed,
                    json.dumps(asdict(case.context), default=str),
                    json.dumps(asdict(case.action), default=str),
                    json.dumps(asdict(case.result), default=str),
                    json.dumps(asdict(case.cost), default=str),
                    json.dumps(asdict(case.diagnosis), default=str),
                    json.dumps(case.metadata, default=str),
                ),
            )
            conn.commit()
        return identity


    def get_case(self, case_id: str) -> Optional[CaseRecord]:
        """Retrieve a case by its case_id."""
        conn = self._get_conn()
        with self._lock:
            row = conn.execute(
                "SELECT * FROM cases WHERE case_id = ?", (case_id,)
            ).fetchone()
        if row is None:
            return None
        return _safe_row_to_case(row)

    def get_by_identity(self, identity_hash: str) -> Optional[CaseRecord]:
        """Retrieve a case by its identity hash."""
        conn = self._get_conn()
        with self._lock:
            row = conn.execute(
                "SELECT * FROM cases WHERE identity_hash = ?", (identity_hash,)
            ).fetchone()
        if row is None:
            return None
        return _safe_row_to_case(row)

    def count(self) -> int:
        """Return the total number of stored cases."""
        conn = self._get_conn()
        with self._lock:
            row = conn.execute("SELECT COUNT(*) FROM cases").fetchone()
        return row[0] if row else 0

    def all_cases(self) -> list[CaseRecord]:
        """Return all readable stored cases (corrupt rows are skipped)."""
        conn = self._get_conn()
        with self._lock:
            rows = conn.execute("SELECT * FROM cases ORDER BY created_at").fetchall()
        cases = [_safe_row_to_case(r) for r in rows]
        return [c for c in cases if c is not None]

    def find_by_dataset(self, dataset_fingerprint: str) -> list[CaseRecord]:
        """Return all cases for a given dataset fingerprint."""
        conn = self._get_conn()
        with self._lock:
            rows = conn.execute(
                "SELECT * FROM cases WHERE dataset_fingerprint = ?",
                (dataset_fingerprint,),
            ).fetchall()
        cases = [_safe_row_to_case(r) for r in rows]
        return [c for c in cases if c is not None]

    def clear(self) -> None:
        """Delete all cases (for testing)."""
        conn = self._get_conn()
        with self._lock:
            conn.execute("DELETE FROM cases")
            conn.commit()


# -----------------------------------------------------------------------------
# Serialization helpers
# -----------------------------------------------------------------------------
def _hydrate_meta_features(value):
    """Rehydrate a persisted DatasetMetaFeatures (dict form or instance)."""
    from .metafeatures import DatasetMetaFeatures, MetaFeatureFamily

    if value is None or isinstance(value, DatasetMetaFeatures):
        return value
    if isinstance(value, dict):
        families_raw = value.get("families") or {}
        try:
            families = {
                (k if isinstance(k, MetaFeatureFamily) else MetaFeatureFamily(k)): v
                for k, v in families_raw.items()
            }
        except (KeyError, ValueError):
            families = {}
        return DatasetMetaFeatures(
            dataset_fingerprint=value.get("dataset_fingerprint", ""),
            families=families,
            raw_stats=value.get("raw_stats") or {},
        )
    return value


def _row_to_case(row: sqlite3.Row) -> CaseRecord:
    """Reconstruct a CaseRecord from a database row.

    Tolerant by design:
    - Unknown JSON keys (written by a newer engine) are dropped instead of
      raising TypeError -- schema drift must not make history unreadable.
    - Rows with corrupt JSON are skipped by callers via ``_safe_row_to_case``.
    """
    from .case import (
        CaseContext,
        CaseAction,
        CaseResult,
        CaseCost,
        CaseDiagnosis,
    )

    def _load(dataclass_type, raw: str):
        known = set(dataclass_type.__dataclass_fields__)
        data = json.loads(raw)
        if isinstance(data, dict):
            data = {k: v for k, v in data.items() if k in known}
        return dataclass_type(**data)

    ctx = _load(CaseContext, row["context_json"])
    ctx.dataset_meta_features = _hydrate_meta_features(ctx.dataset_meta_features)

    return CaseRecord(
        case_id=row["case_id"],
        dataset_fingerprint=row["dataset_fingerprint"],
        schema_fingerprint=row["schema_fingerprint"],
        engine_commit=row["engine_commit"],
        split_fingerprint=row["split_fingerprint"],
        seed=row["seed"],
        context=ctx,
        action=_load(CaseAction, row["action_json"]),
        result=_load(CaseResult, row["result_json"]),
        cost=_load(CaseCost, row["cost_json"]),
        diagnosis=_load(CaseDiagnosis, row["diagnosis_json"]),
        metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
    )


def _safe_row_to_case(row: sqlite3.Row) -> Optional[CaseRecord]:
    """Read a row, returning None instead of raising on corrupt data.

    A single damaged record must not make the whole experience store
    unreadable (reliability, doc 11).
    """
    try:
        return _row_to_case(row)
    except Exception:
        return None
