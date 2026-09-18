"""Parallel scheduler executor (doc 08).

Thread-pool based executor for running trials concurrently with
resource tracking, cancellation, and fault isolation.

Normative source: doc 08 section "Executor" and "Token lifecycle".
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..contracts import (
    TrialResult, TrialSpec, TrialStatus,
)


@dataclass
class ExecutionToken:
    """Token representing allocated resources for one trial (doc 08)."""

    trial_id: str
    cpu_seconds: float = 0.0
    memory_bytes: int = 0
    granted_at: float = 0.0
    released_at: Optional[float] = None

    @property
    def active(self) -> bool:
        return self.released_at is None

    def release(self) -> None:
        self.released_at = time.monotonic()


@dataclass
class TokenPool:
    """Manages execution tokens for concurrent trials (doc 08).

    Tokens enforce resource limits: a trial must acquire a token before
    execution and release it when done.
    """

    max_tokens: int = 4
    max_cpu_seconds: float = 3600.0
    _tokens: dict[str, ExecutionToken] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def acquire(self, trial_id: str) -> bool:
        """Try to acquire a token. Returns True if successful."""
        with self._lock:
            active = sum(1 for t in self._tokens.values() if t.active)
            if active >= self.max_tokens:
                return False
            self._tokens[trial_id] = ExecutionToken(
                trial_id=trial_id, granted_at=time.monotonic(),
            )
            return True

    def release(self, trial_id: str) -> None:
        """Release a token."""
        with self._lock:
            token = self._tokens.get(trial_id)
            if token:
                token.release()

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for t in self._tokens.values() if t.active)

    def available(self) -> bool:
        return self.active_count() < self.max_tokens

    def total_cpu_used(self) -> float:
        with self._lock:
            return sum(t.cpu_seconds for t in self._tokens.values())


@dataclass
class TrialTask:
    """A callable task that runs one trial."""

    trial_spec: TrialSpec
    fn: Callable[[TrialSpec], TrialResult]
    timeout_s: float = 300.0


@dataclass
class SchedulerExecutor:
    """Thread-pool executor for running trials concurrently (doc 08).

    Usage::

        executor = SchedulerExecutor(max_workers=4)
        executor.submit_task(TrialTask(spec, my_fn))
        results = executor.wait_all()
        executor.shutdown()
    """

    max_workers: int = 4
    default_timeout_s: float = 300.0
    _pool: Optional[ThreadPoolExecutor] = field(default=None, init=False)
    _futures: dict[str, Future] = field(default_factory=dict, init=False)
    _results: dict[str, TrialResult] = field(default_factory=dict, init=False)
    _token_pool: TokenPool = field(default_factory=TokenPool, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self):
        self._pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self._token_pool.max_tokens = self.max_workers

    def submit_task(self, task: TrialTask) -> bool:
        """Submit a trial task for execution. Returns True if accepted."""
        if not self._token_pool.available():
            return False

        trial_id = task.trial_spec.trial_id

        def _run():
            if not self._token_pool.acquire(trial_id):
                return TrialResult(
                    trial_id=trial_id, status=TrialStatus.REJECTED,
                    warnings=["resource pool exhausted"],
                )
            t0 = time.monotonic()
            try:
                result = task.fn(task.trial_spec)
                elapsed = time.monotonic() - t0
                # Record resource usage
                token = self._token_pool._tokens.get(trial_id)
                if token:
                    token.cpu_seconds = elapsed
                return result
            except Exception as e:
                return TrialResult(
                    trial_id=trial_id, status=TrialStatus.FAILED,
                    warnings=[str(e)],
                )
            finally:
                self._token_pool.release(trial_id)

        with self._lock:
            future = self._pool.submit(_run)
            self._futures[trial_id] = future

        return True

    def wait_all(self, timeout_s: Optional[float] = None) -> dict[str, TrialResult]:
        """Wait for all submitted tasks to complete. Returns results by trial_id."""
        timeout = timeout_s or self.default_timeout_s
        with self._lock:
            futures = dict(self._futures)

        for trial_id, future in futures.items():
            try:
                result = future.result(timeout=timeout)
                self._results[trial_id] = result
            except Exception as e:
                self._results[trial_id] = TrialResult(
                    trial_id=trial_id, status=TrialStatus.TIMEOUT,
                    warnings=[f"execution error: {e}"],
                )

        return dict(self._results)

    def wait_one(self, timeout_s: float = 1.0) -> Optional[TrialResult]:
        """Wait for any one task to complete."""
        with self._lock:
            futures = dict(self._futures)

        for trial_id, future in futures.items():
            if future.done():
                try:
                    result = future.result(timeout=0)
                    self._results[trial_id] = result
                    return result
                except Exception:
                    pass
        return None

    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for f in self._futures.values() if not f.done())

    def completed_count(self) -> int:
        return len(self._results)

    def summary(self) -> dict[str, Any]:
        return {
            "max_workers": self.max_workers,
            "active_tokens": self._token_pool.active_count(),
            "pending": self.pending_count(),
            "completed": self.completed_count(),
            "total_cpu_s": round(self._token_pool.total_cpu_used(), 2),
        }

    def shutdown(self, wait: bool = True) -> None:
        if self._pool:
            self._pool.shutdown(wait=wait)
