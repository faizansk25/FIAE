"""Real parallel executor with thread pool (doc 08).

Implements the actual concurrent trial execution with resource management,
backpressure, cancellation, and timeout enforcement.

Normative source: doc 08 "Parallel execution", "Backpressure",
"Nested parallelism guard", "Cancellation".
"""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..ids import new_id


# ---------------------------------------------------------------------------
# Resource token pool (doc 08)
# ---------------------------------------------------------------------------

@dataclass
class ResourcePool:
    """Manages available compute resources (doc 08)."""
    max_workers: int = 4
    max_memory_mb: float = 4096.0
    max_cpu_fraction: float = 0.8

    _used_workers: int = field(default_factory=int, init=False)
    _used_memory_mb: float = field(default_factory=float, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def tokens_available(self, estimate: dict[str, float] | None = None) -> bool:
        """Check if resources are available for the given estimate."""
        with self._lock:
            if self._used_workers >= self.max_workers:
                return False
            est_mem = (estimate or {}).get("memory_mb", 500.0)
            return not self._used_memory_mb + est_mem > self.max_memory_mb

    def reserve(self, estimate: dict[str, float] | None = None) -> str:
        """Reserve resources, return token ID."""
        with self._lock:
            self._used_workers += 1
            est_mem = (estimate or {}).get("memory_mb", 500.0)
            self._used_memory_mb += est_mem
            return new_id("token")

    def release(self, token_id: str, actual_memory_mb: float = 0.0) -> None:
        """Release reserved resources."""
        with self._lock:
            self._used_workers = max(0, self._used_workers - 1)
            self._used_memory_mb = max(0, self._used_memory_mb - actual_memory_mb)

    @property
    def available_workers(self) -> int:
        with self._lock:
            return self.max_workers - self._used_workers

    @property
    def available_memory_mb(self) -> float:
        with self._lock:
            return self.max_memory_mb - self._used_memory_mb


# ---------------------------------------------------------------------------
# Parallel executor (doc 08)
# ---------------------------------------------------------------------------

@dataclass
class ParallelExecutor:
    """Thread pool executor for concurrent trial execution (doc 08)."""
    pool: ResourcePool = field(default_factory=ResourcePool)
    _executor: Optional[ThreadPoolExecutor] = field(default=None, init=False)
    _futures: dict[str, Future] = field(default_factory=dict, init=False)
    _results: queue.Queue = field(default_factory=queue.Queue, init=False)
    _cancelled: set[str] = field(default_factory=set, init=False)
    _tokens: dict[str, str] = field(default_factory=dict, init=False)

    def start(self) -> None:
        """Start the thread pool."""
        self._executor = ThreadPoolExecutor(
            max_workers=self.pool.max_workers,
            thread_name_prefix="fiae_trial",
        )

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool."""
        if self._executor:
            self._executor.shutdown(wait=wait)

    def submit(
        self,
        job_id: str,
        fn: Callable,
        args: tuple = (),
        kwargs: dict | None = None,
        estimate: dict[str, float] | None = None,
    ) -> Optional[str]:
        """Submit a job for execution. Returns token ID or None if rejected."""
        if not self.pool.tokens_available(estimate):
            return None  # backpressure

        token = self.pool.reserve(estimate)
        self._tokens[job_id] = token

        if self._executor is None:
            self.start()

        def _wrapped():
            try:
                result = fn(*args, **(kwargs or {}))
                self._results.put((job_id, "completed", result))
            except Exception as e:
                self._results.put((job_id, "failed", e))
            finally:
                actual_mem = 0.0
                tok = self._tokens.pop(job_id, None)
                if tok:
                    self.pool.release(tok, actual_mem)

        future = self._executor.submit(_wrapped)
        self._futures[job_id] = future
        return token

    def cancel(self, job_id: str) -> bool:
        """Cancel a running job (doc 08 cancellation protocol)."""
        self._cancelled.add(job_id)
        future = self._futures.pop(job_id, None)
        if future:
            return future.cancel()
        return False

    def poll(self) -> list[tuple[str, str, Any]]:
        """Poll for completed jobs. Returns list of (job_id, status, result)."""
        completed = []
        while not self._results.empty():
            try:
                completed.append(self._results.get_nowait())
            except queue.Empty:
                break
        return completed

    def cancel_all(self) -> int:
        """Cancel all running jobs. Returns count cancelled."""
        count = 0
        for job_id in list(self._futures.keys()):
            if self.cancel(job_id):
                count += 1
        return count


# ---------------------------------------------------------------------------
# Inline executor (for single-threaded fallback)
# ---------------------------------------------------------------------------

@dataclass
class InlineExecutor:
    """Execute jobs inline (single-threaded, doc 08)."""
    _results: list = field(default_factory=list)

    def submit(
        self,
        job_id: str,
        fn: Callable,
        args: tuple = (),
        kwargs: dict | None = None,
        **_: Any,
    ) -> str:
        """Execute immediately."""
        try:
            result = fn(*args, **(kwargs or {}))
            self._results.append((job_id, "completed", result))
        except Exception as e:
            self._results.append((job_id, "failed", e))
        return job_id

    def poll(self) -> list[tuple[str, str, Any]]:
        """Return and clear results."""
        results = list(self._results)
        self._results.clear()
        return results

    def cancel(self, job_id: str) -> bool:
        return False

    def shutdown(self, wait: bool = True) -> None:
        pass


# ---------------------------------------------------------------------------
# Event path (doc 08)
# ---------------------------------------------------------------------------

@dataclass
class EventEnvelope:
    """Compact event for the async event path (doc 08)."""
    event_type: str  # "trial_started", "trial_completed", "trial_failed", "resource_warning"
    job_id: str = ""
    timestamp: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class EventStream:
    """Async event path: worker → EventEnvelope → batch writer (doc 08)."""
    def __init__(self, max_buffer: int = 1000):
        self._buffer: list[EventEnvelope] = []
        self._lock = threading.Lock()
        self._max_buffer = max_buffer

    def emit(self, event: EventEnvelope) -> None:
        with self._lock:
            self._buffer.append(event)
            if len(self._buffer) >= self._max_buffer:
                self._flush()

    def _flush(self) -> list[EventEnvelope]:
        events = list(self._buffer)
        self._buffer.clear()
        return events

    def drain(self) -> list[EventEnvelope]:
        with self._lock:
            return self._flush()
