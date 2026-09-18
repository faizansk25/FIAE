"""Trial orchestration engine (doc 07).

Manages the lifecycle of model training trials with fidelity ladders,
successive halving pruning, and resource accounting.

Normative source: doc 07 section "Fidelity ladders" and "Successive halving".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..contracts import (
    MetricValue, ResourceMeasurement,
    TrialResult, TrialSpec, TrialStatus,
)


@dataclass
class FidelityLadder:
    """Defines the progression of resource allocations for a trial (doc 07).

    Each rung is a fraction of the full budget.  Trials start at rung 0
    and are promoted only if they show sufficient improvement.
    """

    rungs: tuple = (0.1, 0.25, 0.5, 1.0)
    min_improvement: float = 0.01  # minimum gain to promote
    promotion_metric: str = "score"


@dataclass
class PruningPolicy:
    """Successive halving pruning policy (doc 07)."""

    min_promotion_fraction: float = 0.5  # bottom half pruned each rung
    min_rungs: int = 2
    aggressive_pruning: bool = False  # True = bracket pruning


@dataclass
class ResourceBudget:
    """Resource limits for trial execution (doc 07)."""

    max_wall_time_s: float = 600.0
    max_memory_bytes: int = 4 * 1024 * 1024 * 1024  # 4 GB
    max_concurrent_trials: int = 4
    max_total_cpu_seconds: float = 3600.0


@dataclass
class TrialState:
    """Runtime state of a trial during execution."""

    trial_spec: TrialSpec
    current_rung: int = 0
    status: TrialStatus = TrialStatus.QUEUED
    fold_metrics: list[MetricValue] = field(default_factory=list)
    resource_measurements: list[ResourceMeasurement] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def wall_time_s(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


@dataclass
class TrialRunner:
    """Orchestrates trial execution with fidelity ladders and pruning (doc 07).

    Usage::

        runner = TrialRunner(budget=ResourceBudget())
        runner.submit(trial_spec)
        # ... runner executes trials, applying pruning ...
        results = runner.completed_results()
    """

    budget: ResourceBudget = field(default_factory=ResourceBudget)
    fidelity: FidelityLadder = field(default_factory=FidelityLadder)
    pruning: PruningPolicy = field(default_factory=PruningPolicy)

    _active: list[TrialState] = field(default_factory=list)
    _completed: list[TrialState] = field(default_factory=list)
    _pruned: list[TrialState] = field(default_factory=list)
    _total_cpu_s: float = 0.0

    def submit(self, spec: TrialSpec) -> str:
        """Submit a trial for execution. Returns trial_id."""
        state = TrialState(trial_spec=spec)
        self._active.append(state)
        return spec.trial_id

    def can_submit(self) -> bool:
        """Check if we can accept more trials under the budget."""
        if len(self._active) >= self.budget.max_concurrent_trials:
            return False
        return not self._total_cpu_s >= self.budget.max_total_cpu_seconds

    def active_trials(self) -> list[TrialState]:
        """Return currently active trials."""
        return list(self._active)

    def completed_results(self) -> list[TrialResult]:
        """Return results of all completed trials."""
        return [
            TrialResult(
                trial_id=s.trial_spec.trial_id,
                status=s.status,
                metrics=s.fold_metrics,
                resource_measurements=s.resource_measurements,
            )
            for s in self._completed
        ]

    def select_next_rung(self) -> Optional[TrialState]:
        """Select the next trial to promote to the next fidelity rung.

        Implements successive halving: rank all active trials by their
        current metric, promote the top fraction.
        """
        if not self._active:
            return None

        # Find trials at their current rung
        candidates = [s for s in self._active if s.status == TrialStatus.RUNNING]
        if not candidates:
            candidates = self._active

        # Sort by best metric (descending for maximize)
        def _sort_key(state: TrialState):
            if not state.fold_metrics:
                return -float("inf")
            return max(m.value for m in state.fold_metrics)

        candidates.sort(key=_sort_key, reverse=True)

        # Prune bottom fraction
        n_keep = max(1, int(len(candidates) * self.pruning.min_promotion_fraction))
        pruned = candidates[n_keep:]
        for state in pruned:
            state.status = TrialStatus.PRUNED
            self._active.remove(state)
            self._pruned.append(state)

        # Return top trial for promotion
        return candidates[0] if candidates else None

    def record_completion(
        self, trial_id: str, metrics: list[MetricValue],
        resources: list[ResourceMeasurement],
    ) -> None:
        """Record trial completion and move to completed list."""
        for state in self._active:
            if state.trial_spec.trial_id == trial_id:
                state.status = TrialStatus.COMPLETED
                state.fold_metrics = metrics
                state.resource_measurements = resources
                state.end_time = time.monotonic()
                self._active.remove(state)
                self._completed.append(state)
                # Track CPU usage
                for r in resources:
                    if r.cpu_time_s:
                        self._total_cpu_s += r.cpu_time_s
                return

    def record_failure(self, trial_id: str, reason: str = "") -> None:
        """Record trial failure."""
        for state in self._active:
            if state.trial_spec.trial_id == trial_id:
                state.status = TrialStatus.FAILED
                state.end_time = time.monotonic()
                state.history.append({"event": "failed", "reason": reason})
                self._active.remove(state)
                self._completed.append(state)
                return

    def summary(self) -> dict[str, Any]:
        """Return a summary of the runner's state."""
        return {
            "active": len(self._active),
            "completed": len(self._completed),
            "pruned": len(self._pruned),
            "total_cpu_s": round(self._total_cpu_s, 2),
            "budget_remaining_s": max(0, self.budget.max_total_cpu_seconds - self._total_cpu_s),
        }
