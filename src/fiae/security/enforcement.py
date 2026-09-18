"""Resource enforcement, input validation, and incident recording (doc 11).

Runtime enforcement of resource limits, input sanitization, path safety,
dependency verification, and incident record keeping.

Normative source: doc 11 "Denial-of-resource protections", "Path/file
safety", "Dependency security", "Audit integrity".
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Resource limits (doc 11)
# ---------------------------------------------------------------------------

@dataclass
class ResourceLimits:
    """Hard resource limits for pipeline execution (doc 11)."""
    max_line_bytes: int = 1_000_000  # 1MB per line
    max_field_bytes: int = 100_000   # 100KB per field
    max_candidate_count: int = 10_000
    max_feature_depth: int = 20
    max_pending_queue: int = 1_000
    max_text_length: int = 1_000_000  # for fast-mode text processing
    max_wall_time_s: float = 3600.0   # 1 hour
    max_ram_mb: float = 8_000.0       # 8 GB
    max_temp_disk_mb: float = 10_000.0  # 10 GB

    def check_line(self, line: str) -> tuple[bool, str]:
        """Check line size limit."""
        size = len(line.encode("utf-8", errors="replace"))
        if size > self.max_line_bytes:
            return False, f"Line too large: {size} bytes > {self.max_line_bytes}"
        return True, ""

    def check_field(self, value: str) -> tuple[bool, str]:
        """Check field size limit."""
        size = len(value.encode("utf-8", errors="replace"))
        if size > self.max_field_bytes:
            return False, f"Field too large: {size} bytes > {self.max_field_bytes}"
        return True, ""

    def check_text(self, text: str) -> tuple[bool, str]:
        """Check text length for fast-mode processing."""
        if len(text) > self.max_text_length:
            return False, f"Text too long: {len(text)} chars > {self.max_text_length}"
        return True, ""


# ---------------------------------------------------------------------------
# Input validation (doc 11)
# ---------------------------------------------------------------------------

class InputValidator:
    """Validates and sanitizes all external inputs (doc 11)."""

    # Dangerous patterns
    _PATH_TRAVERSAL = re.compile(r"\.\.[\\/]")
    _SHELL_METACHARACTERS = re.compile(r"[;&|`$(){}]")
    _SQL_INJECTION = re.compile(r"(--|;|'|\"|\\)", re.IGNORECASE)

    def validate_path(self, path: str) -> tuple[bool, str]:
        """Validate file path safety (doc 11 path/file safety)."""
        # Normalize
        normalized = os.path.normpath(path)

        # Check traversal
        if self._PATH_TRAVERSAL.search(normalized):
            return False, f"Path traversal detected: {path}"

        # Check for shell metacharacters
        if self._SHELL_METACHARACTERS.search(normalized):
            return False, f"Shell metacharacters in path: {path}"

        return True, ""

    def validate_column_name(self, name: str) -> tuple[bool, str]:
        """Validate column name against injection."""
        if not name:
            return False, "Empty column name"
        if len(name) > 256:
            return False, f"Column name too long: {len(name)}"
        if self._SQL_INJECTION.search(name):
            return False, f"Suspicious characters in column name: {name}"
        return True, ""

    def validate_filename(self, name: str) -> tuple[bool, str]:
        """Validate generated filename."""
        if not name or name.startswith("."):
            return False, f"Invalid filename: {name}"
        if "/" in name or "\\" in name:
            return False, f"Path separator in filename: {name}"
        return True, ""

    def sanitize_string(self, value: str, max_length: int = 10000) -> str:
        """Sanitize a string for safe storage/logging."""
        # Remove null bytes
        value = value.replace("\x00", "")
        # Truncate
        if len(value) > max_length:
            value = value[:max_length] + "...[truncated]"
        return value

    def verify_dependency(self, package: str, version: str = "") -> tuple[bool, str]:
        """Verify a dependency is in the allowed list (doc 11)."""
        ALLOWED = {
            "scikit-learn", "numpy", "pandas", "scipy", "joblib",
            "pytest", "hypothesis",
        }
        if package.lower() not in {a.lower() for a in ALLOWED}:
            return False, f"Package not in allowed list: {package}"
        return True, ""


# ---------------------------------------------------------------------------
# Incident record (doc 11)
# ---------------------------------------------------------------------------

@dataclass
class IncidentRecord:
    """Immutable incident record (doc 11)."""
    incident_id: str = ""
    timestamp: float = 0.0
    severity: str = "info"  # "info" | "warning" | "error" | "critical"
    category: str = ""  # "resource" | "security" | "data" | "integrity"
    component: str = ""
    description: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    remediation: str = ""
    related_ids: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.incident_id:
            self.incident_id = f"inc_{hashlib.sha256(f'{self.timestamp}{self.category}{self.component}'.encode()).hexdigest()[:12]}"
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "incident_id": self.incident_id,
            "timestamp": self.timestamp,
            "severity": self.severity,
            "category": self.category,
            "component": self.component,
            "description": self.description,
            "evidence": self.evidence,
            "remediation": self.remediation,
        }


class IncidentLog:
    """Append-only incident log (doc 11 "Audit integrity")."""

    def __init__(self, log_path: str = ""):
        self.log_path = log_path
        self._records: list[IncidentRecord] = []
        self._lock = __import__("threading").Lock()

    def record(self, incident: IncidentRecord) -> str:
        """Append an incident record. Returns the incident ID."""
        with self._lock:
            self._records.append(incident)
            if self.log_path:
                self._append_to_disk(incident)
        return incident.incident_id

    def _append_to_disk(self, incident: IncidentRecord) -> None:
        import json
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(incident.to_dict(), default=str) + "\n")

    def read(self, limit: int = 100) -> list[IncidentRecord]:
        """Read recent incidents (append-only, no rewrites)."""
        with self._lock:
            return list(self._records[-limit:])


# ---------------------------------------------------------------------------
# Plugin permission model (doc 11)
# ---------------------------------------------------------------------------

@dataclass
class PluginPermission:
    """Permission model for custom feature operators (doc 11)."""
    name: str
    input_schema: dict[str, str] = field(default_factory=dict)
    output_schema: dict[str, str] = field(default_factory=dict)
    requires_fit: bool = False
    target_access: str = "P0"  # P0|P1|P2|P3
    temporal_access: bool = False
    group_access: bool = False
    deterministic: bool = True
    estimated_memory_mb: float = 100.0
    version: str = "1.0.0"


def validate_plugin_permission(
    perm: PluginPermission, limits: ResourceLimits | None = None
) -> list[str]:
    """Validate a plugin's permission declaration. Returns list of violations."""
    violations = []
    if not perm.name:
        violations.append("Plugin must have a name")
    if perm.target_access not in ("P0", "P1", "P2", "P3"):
        violations.append(f"Invalid target_access: {perm.target_access}")
    if perm.target_access in ("P2", "P3") and not perm.temporal_access:
        violations.append(f"Target access {perm.target_access} requires temporal access")
    if limits and perm.estimated_memory_mb > limits.max_ram_mb:
        violations.append(f"Memory estimate {perm.estimated_memory_mb}MB exceeds limit")
    return violations
