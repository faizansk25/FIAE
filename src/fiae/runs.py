"""Run lifecycle: durable run directory, manifest, state machine, cancellation.

Normative sources: doc 10 (run identity, local persistence, crash recovery),
doc 11 (reliability states), doc 15 (bootstrap phase).

Reliability states (doc 11):
    CREATED -> PLANNING -> RUNNING -> {COMPLETED | FAILED | CANCELLED | INTERRUPTED}
A crash can never become COMPLETED automatically; COMPLETED is reachable only
from RUNNING via an explicit completion-contract check.
"""

from __future__ import annotations

import enum
import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional

from .contracts import RunManifest
from .errors import ErrorCode, FIAEError, Severity
from .events import EventBus, EventType
from .ids import content_hash, new_id


class RunState(str, enum.Enum):
    CREATED = "CREATED"
    PLANNING = "PLANNING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"


# Legal transitions. Terminal states have no outgoing transitions, and
# COMPLETED is reachable only from RUNNING (doc 11: crash is never COMPLETED).
ALLOWED_TRANSITIONS: dict[RunState, set[RunState]] = {
    RunState.CREATED: {RunState.PLANNING, RunState.FAILED, RunState.CANCELLED},
    RunState.PLANNING: {
        RunState.RUNNING,
        RunState.FAILED,
        RunState.CANCELLED,
        RunState.INTERRUPTED,
    },
    RunState.RUNNING: {
        RunState.COMPLETED,
        RunState.FAILED,
        RunState.CANCELLED,
        RunState.INTERRUPTED,
    },
    RunState.COMPLETED: set(),
    RunState.FAILED: set(),
    RunState.CANCELLED: set(),
    RunState.INTERRUPTED: set(),
}


class CancellationToken:
    """Cooperative cancellation token (doc 08 cancellation protocol)."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self.cancelled:
            raise FIAEError(
                code=ErrorCode.CANCELLED,
                safe_message="Operation cancelled by request.",
                severity=Severity.WARNING,
                component="cancellation",
                recoverable=True,
            )


class RunHandle:
    """Durable handle for one run: directory, event bus, manifest, state."""

    def __init__(self, run_id: str, path: Path, bus: EventBus, manifest: RunManifest) -> None:
        self.run_id = run_id
        self.path = path
        self.bus = bus
        self.manifest = manifest
        self._state_lock = threading.Lock()

    # -- state machine -------------------------------------------------------
    @property
    def state(self) -> RunState:
        return RunState(self._read_manifest()["completion_state"])

    def transition(self, new_state: RunState, *, reason: str = "") -> RunState:
        with self._state_lock:
            current = self.state
            if new_state not in ALLOWED_TRANSITIONS[current]:
                raise FIAEError(
                    code=ErrorCode.REPRODUCIBILITY_MISMATCH,
                    safe_message=(
                        f"Illegal run state transition {current.value} -> {new_state.value}."
                    ),
                    component="run_lifecycle",
                    run_id=self.run_id,
                    evidence={"current": current.value, "requested": new_state.value},
                )
            m = self._read_manifest()
            m["completion_state"] = new_state.value
            if reason:
                m.setdefault("state_history", []).append(
                    {"state": new_state.value, "reason": reason}
                )
            self._write_manifest(m)
        event_map = {
            RunState.RUNNING: EventType.RUN_STARTED,
            RunState.COMPLETED: EventType.RUN_COMPLETED,
            RunState.FAILED: EventType.RUN_FAILED,
            RunState.CANCELLED: EventType.RUN_CANCELLED,
            RunState.INTERRUPTED: EventType.RUN_INTERRUPTED,
        }
        event_type = event_map.get(new_state)
        if event_type is not None:
            self.bus.emit(
                run_id=self.run_id,
                event_type=event_type,
                component="run_lifecycle",
                payload={"reason": reason},
            )
        return new_state

    # -- persistence ----------------------------------------------------------
    def _manifest_path(self) -> Path:
        return self.path / "run.json"

    def _read_manifest(self) -> dict[str, Any]:
        return json.loads(self._manifest_path().read_text(encoding="utf-8"))

    def _write_manifest(self, m: dict[str, Any]) -> None:
        tmp = self._manifest_path().with_suffix(".json.tmp")
        tmp.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._manifest_path())

    def update_manifest(self, **fields: Any) -> None:
        m = self._read_manifest()
        m.update(fields)
        self._write_manifest(m)


class RunManager:
    """Creates durable runs (doc 15 bootstrap phase)."""

    def __init__(self, runs_root: Path) -> None:
        self.runs_root = Path(runs_root)
        self.runs_root.mkdir(parents=True, exist_ok=True)

    def create_run(
        self,
        config: Mapping[str, Any],
        *,
        engine_build: str = "dev",
        environment: Optional[dict[str, Any]] = None,
    ) -> RunHandle:
        run_id = new_id("run")
        run_path = self.runs_root / run_id
        (run_path / "metrics").mkdir(parents=True)
        (run_path / "reports").mkdir()
        (run_path / "artifacts").mkdir()
        (run_path / "events.jsonl").touch()
        (run_path / "decisions.jsonl").touch()

        manifest = RunManifest(
            run_id=run_id,
            completion_state=RunState.CREATED.value,
            config_hash=content_hash(dict(config)),
            engine_build=engine_build,
            environment=dict(environment or {}),
        )
        handle = RunHandle(run_id, run_path, self._make_bus(run_path, run_id), manifest)
        handle._write_manifest(asdict(manifest))
        handle.bus.emit(
            run_id=run_id,
            event_type=EventType.RUN_CREATED,
            component="run_manager",
            payload={"config_hash": manifest.config_hash},
        )
        return handle

    @staticmethod
    def _make_bus(run_path: Path, run_id: str) -> EventBus:
        return EventBus(
            events_path=run_path / "events.jsonl",
            sqlite_path=run_path / "index.sqlite",
        )

