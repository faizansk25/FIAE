"""Experiment tracking — optional MLflow / Weights & Biases exporters.

Design principles (doc 10, doc 14):
- Internal schema maps cleanly to run/params/metrics/artifacts/tags so
  exporters are OPTIONAL, never mandatory dependencies.
- Zero-overhead when disabled: backends are imported lazily and a missing
  package degrades to the local JSONL run log with an audit note, never a
  crash.
- Every run always produces a durable local audit trail (audit.jsonl),
  independent of any external tracking backend.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TrackingConfig:
    """Configuration for experiment tracking.

    backend : "none" (default) | "mlflow" | "wandb" | "local"
        "local" writes only the local run log + audit trail.
    experiment : str
        Experiment/project name used by the backend.
    run_name : str, optional
        Human-readable run name; defaults to run_<timestamp>.
    output_dir : str
        Directory for the local run log and audit trail.
    tags : dict
        Arbitrary key/value metadata attached to the run.
    """

    backend: str = "none"
    experiment: str = "fiae"
    run_name: Optional[str] = None
    output_dir: str = "runs"
    tags: dict[str, str] = field(default_factory=dict)


class ExperimentTracker:
    """Records a run locally (always) and optionally exports to a backend."""

    def __init__(self, config: TrackingConfig) -> None:
        self.config = config
        self.run_id = ""
        self._client: Any = None
        self._active = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self, params: dict[str, Any]) -> str:
        """Start a run. Returns the run id."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        if not self.config.run_name:
            self.config.run_name = "run_" + time.strftime("%Y%m%d_%H%M%S")
        self.run_id = self.config.run_name

        params_json = {
            k: v for k, v in params.items()
            if isinstance(v, (str, int, float, bool, type(None)))
        }
        self._write_local("run_start.json", {
            "run_id": self.run_id,
            "backend": self.config.backend,
            "started_at": time.time(),
            "params": params_json,
            "tags": self.config.tags,
        })
        self._log_audit("pipeline", "run_start", self.run_id, params_json)

        backend = self.config.backend.lower()
        if backend == "mlflow":
            self._client = self._start_mlflow(params_json)
        elif backend == "wandb":
            self._client = self._start_wandb(params_json)
        self._active = True
        return self.run_id

    def log_metrics(self, metrics: dict[str, float], step: int = 0) -> None:
        """Log scalar metrics for the run."""
        if not self._active:
            return
        clean = {k: float(v) for k, v in metrics.items()
                 if isinstance(v, (int, float))}
        self._write_local("metrics.jsonl",
                          json.dumps({"step": step, **clean}) + "\n",
                          append=True)
        if self._client is not None:
            try:
                if self.config.backend == "mlflow":
                    self._client.log_metrics(clean, step=step)
                elif self.config.backend == "wandb":
                    import wandb
                    wandb.log(clean, step=step)
            except Exception:
                pass  # tracking must never crash the pipeline

    def log_artifact(self, path: str) -> None:
        """Attach a file artifact (report, model, exported code)."""
        if not self._active or not os.path.exists(path):
            return
        self._log_audit("pipeline", "artifact", os.path.basename(path),
                        {"path": path})
        if self._client is not None:
            try:
                if self.config.backend == "mlflow":
                    self._client.log_artifact(path)
                elif self.config.backend == "wandb":
                    import wandb
                    wandb.save(path)
            except Exception:
                pass

    def finish(self) -> None:
        """End the run and flush everything."""
        if not self._active:
            return
        self._write_local("run_end.json", {
            "run_id": self.run_id,
            "ended_at": time.time(),
        })
        self._log_audit("pipeline", "run_end", self.run_id, {})
        if self._client is not None:
            try:
                if self.config.backend == "mlflow":
                    import mlflow
                    mlflow.end_run()
                elif self.config.backend == "wandb":
                    import wandb
                    wandb.finish()
            except Exception:
                pass
        self._active = False

    # ------------------------------------------------------------------
    # Backends (lazy, failure-tolerant)
    # ------------------------------------------------------------------
    def _start_mlflow(self, params: dict) -> Any:
        try:
            import mlflow
            mlflow.set_experiment(self.config.experiment)
            mlflow.start_run(run_name=self.config.run_name)
            mlflow.set_tags(self.config.tags)
            mlflow.log_params(params)
            return mlflow
        except ImportError:
            self._log_audit("pipeline", "backend_unavailable", "mlflow",
                            {"reason": "mlflow not installed"})
            return None
        except Exception as exc:
            self._log_audit("pipeline", "backend_error", "mlflow",
                            {"reason": str(exc)}, severity="warning")
            return None

    def _start_wandb(self, params: dict) -> Any:
        try:
            import wandb
            wandb.init(project=self.config.experiment,
                       name=self.config.run_name,
                       config={**params, **self.config.tags})
            return wandb
        except ImportError:
            self._log_audit("pipeline", "backend_unavailable", "wandb",
                            {"reason": "wandb not installed"})
            return None
        except Exception as exc:
            self._log_audit("pipeline", "backend_error", "wandb",
                            {"reason": str(exc)}, severity="warning")
            return None

    # ------------------------------------------------------------------
    # Local persistence
    # ------------------------------------------------------------------
    def _run_dir(self) -> str:
        return os.path.join(self.config.output_dir, self.run_id)

    def _write_local(self, name: str, data: Any, append: bool = False) -> None:
        try:
            d = self._run_dir()
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, name)
            mode = "a" if append else "w"
            with open(path, mode, encoding="utf-8") as f:
                if isinstance(data, str):
                    f.write(data)
                else:
                    json.dump(data, f, indent=2, default=str)
        except OSError:
            pass  # never crash the pipeline for local logging

    def _log_audit(self, event_type: str, action: str, subject: str,
                   details: dict, severity: str = "info") -> None:
        try:
            from ..security.audit import AuditLogger, AuditEntry
            logger = AuditLogger(
                os.path.join(self._run_dir(), "audit.jsonl"))
            logger.log(AuditEntry(
                event_type=event_type, component="experiment",
                action=action, subject=subject, details=details,
                severity=severity, run_id=self.run_id))
            logger.flush()
        except Exception:
            pass

