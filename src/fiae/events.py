"""Event bus: structured events, JSONL durability, SQLite index, async writer.

Normative sources: doc 10 (observability) and doc 01 (audit events cannot be
dropped; NFR-008 minimal observability overhead).

Design:
- Events flow through a bounded queue to a background batch writer (backpressure
  by blocking, never dropping).
- Critical/audit events can be written synchronously and durably flushed.
- Subscribers receive live copies of every envelope.
"""

from __future__ import annotations

import contextlib
import json
import queue
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

SCHEMA_VERSION = 1


class EventType:
    """Event-type catalog from doc 10 (extensible; stable string values)."""

    # Run
    RUN_CREATED = "RUN_CREATED"
    RUN_STARTED = "RUN_STARTED"
    RUN_CANCELLED = "RUN_CANCELLED"
    RUN_FAILED = "RUN_FAILED"
    RUN_COMPLETED = "RUN_COMPLETED"
    RUN_INTERRUPTED = "RUN_INTERRUPTED"
    # Data
    SOURCE_OPENED = "SOURCE_OPENED"
    PROFILE_STARTED = "PROFILE_STARTED"
    PROFILE_COMPLETED = "PROFILE_COMPLETED"
    DATA_QUALITY_FINDING = "DATA_QUALITY_FINDING"
    # Problem
    TASK_INFERRED = "TASK_INFERRED"
    VALIDATION_PLAN_CREATED = "VALIDATION_PLAN_CREATED"
    LEAKAGE_FINDING = "LEAKAGE_FINDING"
    # Feature
    FEATURE_PROPOSED = "FEATURE_PROPOSED"
    FEATURE_DUPLICATE = "FEATURE_DUPLICATE"
    FEATURE_PRUNED = "FEATURE_PRUNED"
    FEATURE_MATERIALIZED = "FEATURE_MATERIALIZED"
    FEATURE_EVALUATED = "FEATURE_EVALUATED"
    FEATURE_ACCEPTED = "FEATURE_ACCEPTED"
    FEATURE_REJECTED = "FEATURE_REJECTED"

    # Trial
    TRIAL_QUEUED = "TRIAL_QUEUED"
    TRIAL_STARTED = "TRIAL_STARTED"
    TRIAL_CHECKPOINT = "TRIAL_CHECKPOINT"
    TRIAL_PRUNED = "TRIAL_PRUNED"
    TRIAL_FAILED = "TRIAL_FAILED"
    TRIAL_COMPLETED = "TRIAL_COMPLETED"
    TRIAL_PROMOTED = "TRIAL_PROMOTED"
    # Ensemble
    ENSEMBLE_CANDIDATE = "ENSEMBLE_CANDIDATE"
    ENSEMBLE_MEMBER_ADDED = "ENSEMBLE_MEMBER_ADDED"
    ENSEMBLE_MEMBER_REJECTED = "ENSEMBLE_MEMBER_REJECTED"
    ENSEMBLE_STACKING_DISABLED = "ENSEMBLE_STACKING_DISABLED"
    # Codegen
    CODEGEN_STARTED = "CODEGEN_STARTED"
    CODEGEN_TEST_FAILED = "CODEGEN_TEST_FAILED"
    CODEGEN_PARITY_PASSED = "CODEGEN_PARITY_PASSED"
    CODEGEN_EXPORT_COMPLETED = "CODEGEN_EXPORT_COMPLETED"
    # Experience
    CASE_WRITTEN = "CASE_WRITTEN"
    RETRIEVAL_COMPLETED = "RETRIEVAL_COMPLETED"
    # Generic
    DECISION_RECORDED = "DECISION_RECORDED"
    STAGE_COMPLETED = "STAGE_COMPLETED"


@dataclass
class EventEnvelope:
    run_id: Optional[str]
    event_type: str
    component: str
    level: str = "INFO"
    stage_id: Optional[str] = None
    trial_id: Optional[str] = None
    feature_id: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["timestamp"] is None:
            d["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        return d


Subscriber = Callable[[EventEnvelope], None]


class EventBus:
    """Bounded-queue, batch-writing durable event bus for a single run."""

    def __init__(
        self,
        events_path: Path,
        sqlite_path: Optional[Path] = None,
        *,
        max_queue: int = 10_000,
        batch_size: int = 100,
        flush_interval_s: float = 0.2,
    ) -> None:
        self._events_path = events_path
        self._sqlite_path = sqlite_path
        self._max_queue = max_queue
        self._batch_size = batch_size
        self._flush_interval_s = flush_interval_s
        self._queue: queue.Queue[Optional[EventEnvelope]] = queue.Queue(maxsize=max_queue)
        self._subscribers: list[Subscriber] = []
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._writer = threading.Thread(target=self._write_loop, daemon=True)
        self._conn: Optional[sqlite3.Connection] = None
        if sqlite_path is not None:
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(sqlite_path), check_same_thread=False)
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    run_id TEXT,
                    stage_id TEXT,
                    trial_id TEXT,
                    feature_id TEXT,
                    level TEXT,
                    component TEXT,
                    event_type TEXT,
                    schema_version INTEGER,
                    payload TEXT
                )
                """
            )
            self._conn.commit()
        self._writer.start()

    # -- public API ---------------------------------------------------------
    def subscribe(self, fn: Subscriber) -> None:
        self._subscribers.append(fn)

    def emit(
        self,
        envelope: Optional[EventEnvelope] = None,
        *,
        durable: bool = False,
        **fields: Any,
    ) -> EventEnvelope:
        """Emit an event. durable=True writes synchronously (audit-safe)."""
        if envelope is None:
            envelope = EventEnvelope(**fields)
        for fn in list(self._subscribers):
            fn(envelope)
        if durable:
            self._write_batch([envelope])
            self._flush_handles()
        else:
            # Bounded queue: block (pause producer) instead of dropping (doc 08).
            self._queue.put(envelope, block=True)
        return envelope

    def flush(self, timeout_s: float = 10.0) -> None:
        """Wait until queued events are durably written, then fsync handles."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._queue.empty():
                break
            time.sleep(0.01)
        self._flush_handles()

    def close(self) -> None:
        self.flush()
        self._closed.set()
        with contextlib.suppress(queue.Full):
            self._queue.put(None, timeout=5.0)
        self._writer.join(timeout=10.0)
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- internals -----------------------------------------------------------
    def _write_batch(self, batch: list[EventEnvelope]) -> None:
        lines = [json.dumps(e.to_dict(), ensure_ascii=False, default=str) for e in batch]
        with self._lock:
            with open(self._events_path, "a", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
            if self._conn is not None:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    [
                        (
                            e.event_id,
                            e.to_dict()["timestamp"],
                            e.run_id,
                            e.stage_id,
                            e.trial_id,
                            e.feature_id,
                            e.level,
                            e.component,
                            e.event_type,
                            e.schema_version,
                            json.dumps(e.payload, ensure_ascii=False, default=str),
                        )
                        for e in batch
                    ],
                )
                self._conn.commit()

    def _flush_handles(self) -> None:
        with self._lock:
            try:
                with open(self._events_path, "a", encoding="utf-8") as fh:
                    fh.flush()
            except FileNotFoundError:
                pass
            if self._conn is not None:
                self._conn.commit()

    def _write_loop(self) -> None:
        while True:
            batch: list[EventEnvelope] = []
            try:
                item = self._queue.get(timeout=self._flush_interval_s)
            except queue.Empty:
                if self._closed.is_set() and self._queue.empty():
                    return
                continue
            if item is None:
                # Drain any remaining queued events before exiting.
                while True:
                    try:
                        extra = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if extra is not None:
                        batch.append(extra)
                if batch:
                    self._write_batch(batch)
                return
            batch.append(item)
            while len(batch) < 500:
                try:
                    nxt = self._queue.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    continue
                batch.append(nxt)
            self._write_batch(batch)
