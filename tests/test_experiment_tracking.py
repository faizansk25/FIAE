"""Tests for experiment tracking (Phase 4: local runs + MLflow/W&B exporters)."""

import json
import os

import pytest

from fiae.experiment.tracking import ExperimentTracker, TrackingConfig


@pytest.fixture
def run_dir(tmp_path):
    return str(tmp_path / "runs")


def _make_tracker(run_dir, backend="local", **kwargs):
    cfg = TrackingConfig(backend=backend, output_dir=run_dir,
                         run_name="test_run", **kwargs)
    return ExperimentTracker(cfg), cfg


class TestLocalTracking:
    def test_run_lifecycle_files(self, run_dir):
        tracker, _ = _make_tracker(run_dir)
        run_id = tracker.start({"target": "y", "seed": 0})
        assert run_id == "test_run"

        d = os.path.join(run_dir, "test_run")
        start = json.load(open(os.path.join(d, "run_start.json")))
        assert start["params"]["target"] == "y"
        assert start["backend"] == "local"

        tracker.log_metrics({"rows": 100, "portfolio_size": 3})
        tracker.log_metrics({"rows": 200}, step=1)
        lines = open(os.path.join(d, "metrics.jsonl")).read().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[1])["rows"] == 200

        tracker.finish()
        end = json.load(open(os.path.join(d, "run_end.json")))
        assert end["run_id"] == "test_run"

    def test_audit_trail_written(self, run_dir):
        tracker, _ = _make_tracker(run_dir)
        tracker.start({})
        tracker.finish()
        audit_path = os.path.join(run_dir, "test_run", "audit.jsonl")
        assert os.path.exists(audit_path)
        events = [json.loads(line) for line in open(audit_path)]
        actions = [e["action"] for e in events]
        assert "run_start" in actions
        assert "run_end" in actions

    def test_noop_after_finish(self, run_dir):
        tracker, _ = _make_tracker(run_dir)
        tracker.start({})
        tracker.finish()
        # Calls after finish must be safe no-ops
        tracker.log_metrics({"x": 1})
        tracker.finish()

    def test_missing_artifact_is_noop(self, run_dir):
        tracker, _ = _make_tracker(run_dir)
        tracker.start({})
        tracker.log_artifact("does_not_exist.json")  # must not raise
        tracker.finish()


class TestBackendFallback:
    def test_mlflow_missing_package_degrades_gracefully(self, run_dir):
        """If mlflow isn't installed, run still completes via local log."""
        try:
            import mlflow  # noqa: F401
            pytest.skip("mlflow installed; fallback path not exercised")
        except ImportError:
            pass
        tracker, _ = _make_tracker(run_dir, backend="mlflow")
        tracker.start({"target": "y"})
        assert tracker._client is None  # degraded, not crashed
        tracker.log_metrics({"rows": 10})
        tracker.finish()

    def test_wandb_missing_package_degrades_gracefully(self, run_dir):
        try:
            import wandb  # noqa: F401
            pytest.skip("wandb installed; fallback path not exercised")
        except ImportError:
            pass
        tracker, _ = _make_tracker(run_dir, backend="wandb")
        tracker.start({"target": "y"})
        assert tracker._client is None
        tracker.finish()


class TestLearnIntegration:
    def test_learn_with_local_tracking(self, tmp_path):
        import pandas as pd
        from fiae.learn import learn, LearnConfig
        from fiae.experiment.tracking import TrackingConfig
        from fiae.intake import auto_adapter

        df = pd.DataFrame({
            "a": [float(i) for i in range(60)],
            "b": [float(i % 7) for i in range(60)],
            "dept": [float(i % 3) for i in range(60)],
        })
        runs = str(tmp_path / "runs")
        cfg = LearnConfig(tracking=TrackingConfig(
            backend="local", output_dir=runs, run_name="learn_run"))
        report = learn(auto_adapter(df), "dept", config=cfg)

        d = os.path.join(runs, "learn_run")
        assert os.path.exists(os.path.join(d, "run_start.json"))
        assert os.path.exists(os.path.join(d, "run_end.json"))
        metrics_lines = open(os.path.join(d, "metrics.jsonl")).read()
        metrics = json.loads(metrics_lines.splitlines()[-1])
        assert metrics["rows"] == report.rows_in_source
        assert os.path.exists(os.path.join(d, "audit.jsonl"))
