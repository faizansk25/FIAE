"""CLI error hints: actionable recovery tips printed for probable error causes."""

from __future__ import annotations

import subprocess
import sys

from fiae.cli import _error_hints
from fiae.errors import ErrorCode, FIAEError


def _err(code: ErrorCode, evidence: dict | None = None) -> FIAEError:
    return FIAEError(
        code=code, safe_message="boom", component="test",
        evidence=evidence or {},
    )


class TestErrorHints:
    def test_format_error_suggests_connect_and_lists_formats(self):
        hints = _error_hints(_err(ErrorCode.DATA_FORMAT_ERROR, {"path": "d.csv"}))
        assert any("fiae connect" in h for h in hints)
        assert any("CSV" in h and "Parquet" in h for h in hints)
        assert any("d.csv" in h for h in hints)

    def test_target_missing_suggests_inspect(self):
        hints = _error_hints(_err(ErrorCode.TARGET_MISSING, {"target": "y"}))
        assert any("fiae inspect" in h for h in hints)
        assert any("'y'" in h for h in hints)
        assert any("case-sensitive" in h for h in hints)

    def test_leakage_points_to_leakage_command(self):
        hints = _error_hints(_err(ErrorCode.LEAKAGE_CONFIRMED))
        assert any("fiae leakage" in h for h in hints)

    def test_resource_codes_suggest_scope_reduction(self):
        for code in (ErrorCode.RESOURCE_PRECHECK_FAILED, ErrorCode.TRIAL_TIMEOUT, ErrorCode.TRIAL_OOM):
            assert any("reduce scope" in h for h in _error_hints(_err(code)))

    def test_unknown_code_gets_json_fallback(self):
        hints = _error_hints(_err(ErrorCode.CANCELLED))
        assert hints and all("--json" in h for h in hints)

    def test_cli_failure_prints_hint_and_docs_line(self, tmp_path):
        """End-to-end: a failing CLI command shows [CODE], a tip, and a docs link."""
        import os
        bad = tmp_path / "missing.csv"
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "..", "src")
        proc = subprocess.run(
            [sys.executable, "-m", "fiae", "inspect", str(bad)],
            capture_output=True, text=True, env=env,
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )  # run from repo root so `-m fiae` resolves; PYTHONPATH targets src/
        assert proc.returncode == 2, f"expected exit 2, got {proc.returncode}"
        combined = proc.stdout + proc.stderr
        assert "DATA_FORMAT_ERROR" in combined
        assert "tip:" in combined
        assert "github.com/faizansk25/FIAE" in combined
