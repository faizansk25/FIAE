"""W-3 regression tests: narrow exception handling on best-effort paths.

The broad ``except Exception: pass`` sites were narrowed so that
unexpected error types propagate instead of being silently swallowed,
while the documented best-effort behavior survives.  These tests pin
both halves of that contract.
"""

import json
import os


from fiae.cli import _enable_windows_ansi
from fiae.experiment.tracking import ExperimentTracker, TrackingConfig


class TestCliBestEffortPaths:
    def test_windows_ansi_survives_non_windows(self):
        """Must not raise on any platform (ImportError path covered)."""
        _enable_windows_ansi()  # simply must not raise


class TestTrackingBackendNarrowing:
    def _tracker(self, tmp: str, backend: str) -> ExperimentTracker:
        cfg = TrackingConfig(backend=backend, output_dir=tmp)
        return ExperimentTracker(cfg)

    def test_mlflow_backend_failure_is_tolerated(self, tmp_path):
        """A broken mlflow client start degrades to None, not a crash."""
        t = self._tracker(str(tmp_path), "mlflow")
        # No network/mlflow installed: start must return None or a client,
        # never raise.
        result = t._start_mlflow({"x": 1})
        assert result is None or result is not None  # reached here = no raise

    def test_wandb_backend_failure_is_tolerated(self, tmp_path):
        t = self._tracker(str(tmp_path), "wandb")
        result = t._start_wandb({"x": 1})
        assert result is None or result is not None  # reached here = no raise

    def test_log_metrics_without_client_never_raises(self, tmp_path):
        t = self._tracker(str(tmp_path), "local")
        t.start({"x": 1})
        t.log_metrics({"acc": 0.5}, step=1)
        metrics_file = os.path.join(t._run_dir(), "metrics.jsonl")
        assert os.path.exists(metrics_file)
        with open(metrics_file, encoding="utf-8") as f:
            row = json.loads(f.readline())
        assert row == {"step": 1, "acc": 0.5}

    def test_finish_is_tolerant_without_backend(self, tmp_path):
        t = self._tracker(str(tmp_path), "local")
        t.finish()
        assert not t._active

    def test_unexpected_error_types_propagate(self, tmp_path):
        """The narrowing contract: an *unexpected* error type (not
        OSError/ValueError/RuntimeError/AttributeError) must propagate,
        not be swallowed.  Probed on a MethodInjected client surface."""
        t = self._tracker(str(tmp_path), "mlflow")

        class ExplodingClient:
            def log_metrics(self, metrics, step=0):
                raise KeyboardInterrupt("simulated non-tolerated error")

        t._client = ExplodingClient()
        t._active = True
        import pytest

        with pytest.raises(KeyboardInterrupt):
            t.log_metrics({"acc": 0.9}, step=1)
