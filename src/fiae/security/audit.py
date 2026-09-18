"""Audit logging and runtime enforcement (doc 11).

Provides persistent audit trails for all pipeline decisions, resource
monitoring with automatic enforcement, and policy violation detection.

Normative source: doc 11 section "Audit logging" and "Runtime enforcement".
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class AuditEntry:
    """A single audit log entry."""

    timestamp: float = 0.0
    event_type: str = ""  # "decision", "resource", "security", "pipeline"
    component: str = ""
    action: str = ""
    subject: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # "info" | "warning" | "error" | "critical"
    run_id: str = ""

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> AuditEntry:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class AuditLogger:
    """Persistent audit logger with thread-safe writes (doc 11)."""

    def __init__(self, log_path: str, max_entries: int = 100000):
        self.log_path = log_path
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._buffer: list[AuditEntry] = []
        self._flush_interval = 100  # flush every N entries

    def log(self, entry: AuditEntry) -> None:
        """Log an audit entry."""
        with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) >= self._flush_interval:
                self._flush()

    def log_decision(self, component: str, action: str, subject: str,
                     details: dict, run_id: str = "") -> None:
        """Log a pipeline decision."""
        self.log(AuditEntry(
            event_type="decision", component=component,
            action=action, subject=subject, details=details,
            run_id=run_id,
        ))

    def log_resource(self, component: str, action: str, details: dict,
                     severity: str = "info") -> None:
        """Log a resource event."""
        self.log(AuditEntry(
            event_type="resource", component=component,
            action=action, details=details, severity=severity,
        ))

    def log_security(self, component: str, action: str, subject: str,
                     details: dict, severity: str = "warning") -> None:
        """Log a security event."""
        self.log(AuditEntry(
            event_type="security", component=component,
            action=action, subject=subject, details=details,
            severity=severity,
        ))

    def log_pipeline(self, component: str, action: str, details: dict,
                     run_id: str = "") -> None:
        """Log a pipeline event."""
        self.log(AuditEntry(
            event_type="pipeline", component=component,
            action=action, details=details, run_id=run_id,
        ))

    def _flush(self) -> None:
        """Flush buffer to disk."""
        if not self._buffer:
            return
        entries = list(self._buffer)
        self._buffer.clear()

        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry.to_dict(), default=str) + "\n")

    def flush(self) -> None:
        """Public flush."""
        with self._lock:
            self._flush()

    def read_entries(self, limit: int = 100) -> list[AuditEntry]:
        """Read recent entries from the log file."""
        entries = []
        if not os.path.exists(self.log_path):
            return entries
        with open(self.log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[-limit:]:
            line = line.strip()
            if line:
                try:
                    entries.append(AuditEntry.from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError):
                    continue
        return entries

    def count_entries(self) -> int:
        if not os.path.exists(self.log_path):
            return 0
        with open(self.log_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)


@dataclass
class ResourceLimits:
    """Runtime resource limits (doc 11)."""

    max_memory_bytes: int = 4 * 1024 * 1024 * 1024  # 4 GB
    max_cpu_seconds: float = 3600.0
    max_disk_bytes: int = 10 * 1024 * 1024 * 1024  # 10 GB
    max_file_size: int = 1024 * 1024 * 1024  # 1 GB per file
    max_open_files: int = 256


@dataclass
class ResourceMonitor:
    """Monitors resource usage against limits (doc 11)."""

    limits: ResourceLimits = field(default_factory=ResourceLimits)
    _start_time: float = 0.0
    _violations: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self._start_time = time.time()

    def check_cpu(self) -> Optional[str]:
        """Check if CPU time limit is exceeded."""
        elapsed = time.time() - self._start_time
        if elapsed > self.limits.max_cpu_seconds:
            msg = f"CPU time {elapsed:.1f}s > {self.limits.max_cpu_seconds}s"
            self._violations.append({"type": "cpu", "message": msg})
            return msg
        return None

    def check_file_size(self, path: str) -> Optional[str]:
        """Check if a file exceeds size limits."""
        if os.path.exists(path):
            size = os.path.getsize(path)
            if size > self.limits.max_file_size:
                msg = f"File {path}: {size} bytes > {self.limits.max_file_size}"
                self._violations.append({"type": "file_size", "message": msg})
                return msg
        return None

    def violations(self) -> list[dict]:
        return list(self._violations)

    def summary(self) -> dict:
        elapsed = time.time() - self._start_time
        return {
            "elapsed_s": round(elapsed, 2),
            "cpu_budget_remaining_s": max(0, self.limits.max_cpu_seconds - elapsed),
            "violations": len(self._violations),
        }
