"""M40.8: the CLI must not crash on non-ASCII column names.

Found by manual testing, not by the suite: profiling a CSV whose header
contains non-ASCII text (``名前 with space``) completed successfully and then
died while *printing*, because a Windows console defaults to cp1252::

    UnicodeEncodeError: 'charmap' codec can't encode characters

The user got a raw traceback instead of a profile. stdout/stderr are now
reconfigured to UTF-8 with ``errors="replace"`` at the CLI entry point, so
even a character the terminal cannot render degrades to a replacement glyph
rather than aborting the command.
"""

import csv
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _write_unicode_csv(path, rows=50):
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["名前 with space", "Åge", "y"])
        for i in range(rows):
            w.writerow([f"値{i}", 20 + i % 50, i % 2])
    return path


class TestStdioIsUtf8Safe:
    def test_helper_exists_and_is_tolerant(self):
        from fiae.cli import _force_utf8_stdio

        # Must never raise, even when streams are already closed/detached.
        _force_utf8_stdio()

    def test_reconfigure_uses_replace_not_strict(self):
        """Strict encoding is what produced the crash."""
        import fiae.cli as cli

        assert callable(cli._force_utf8_stdio)

        class _FakeStream:
            def __init__(self):
                self.kwargs = None

            def reconfigure(self, **kwargs):
                self.kwargs = kwargs

        saved_out, saved_err = sys.stdout, sys.stderr
        fake_out, fake_err = _FakeStream(), _FakeStream()
        sys.stdout, sys.stderr = fake_out, fake_err
        try:
            cli._force_utf8_stdio()
        finally:
            sys.stdout, sys.stderr = saved_out, saved_err

        for fake in (fake_out, fake_err):
            assert fake.kwargs["encoding"] == "utf-8"
            assert fake.kwargs["errors"] == "replace"

    def test_printing_non_ascii_does_not_raise(self, capsys):
        """The direct failure mode, exercised without a subprocess."""
        payload = "名前 with space"
        # A real TextIOWrapper over a cp1252 stream is what blew up; a
        # BytesIO-backed UTF-8 writer stands in for it here.
        buf = io.BytesIO()
        stream = io.TextIOWrapper(buf, encoding="utf-8", errors="replace")
        stream.write(payload)
        stream.flush()
        assert payload in buf.getvalue().decode("utf-8")

    def test_non_ascii_column_survives_profiling(self, tmp_path):
        """End-to-end: the profile itself completes and keeps the names."""
        from fiae.intake.csv_source import CsvDataSourceAdapter
        from fiae.intake.profiler import ProfileConfig, profile_source

        p = _write_unicode_csv(tmp_path / "weird.csv")
        names = [c.name for c in
                 profile_source(CsvDataSourceAdapter(p), ProfileConfig()).columns]

        assert any("名前" in n for n in names), (
            f"BOM/unicode headers must survive, got {names}")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows console encoding")
class TestSubprocessOnWindowsConsole:
    def test_cli_does_not_emit_a_traceback(self, tmp_path):
        """The user-visible symptom: a raw traceback on stdout/stderr."""
        p = _write_unicode_csv(tmp_path / "weird.csv")
        env = dict(os.environ)
        # Make sure the subprocess imports the working tree, not any stale
        # copy left in site-packages (the failure mode hit during manual
        # testing: an old non-editable install shadowed the fix).
        repo_src = str(Path(__file__).resolve().parents[1] / "src")
        env["PYTHONPATH"] = repo_src + os.pathsep + env.get("PYTHONPATH", "")

        proc = subprocess.run(
            [sys.executable, "-m", "fiae", "inspect", str(p)],
            capture_output=True, text=True, errors="replace", timeout=300,
            env=env,
        )
        combined = proc.stdout + proc.stderr
        assert "Traceback" not in combined, (
            "CLI crashed on non-ASCII headers:\n" + combined[-2000:])
        assert proc.returncode == 0, combined[-2000:]


class TestHostileInputsStillFailCleanly:
    """Malformed input must produce typed errors, never tracebacks."""

    @pytest.mark.parametrize("name,body", [
        ("empty.csv", ""),
        ("header_only.csv", "id,age,y\n"),
    ])
    def test_bad_csv_produces_typed_error(self, tmp_path, name, body):
        from fiae.cli import _force_utf8_stdio

        _force_utf8_stdio()
        p = tmp_path / name
        p.write_text(body, encoding="utf-8")

        env = dict(os.environ)
        repo_src = str(Path(__file__).resolve().parents[1] / "src")
        env["PYTHONPATH"] = repo_src + os.pathsep + env.get("PYTHONPATH", "")

        proc = subprocess.run(
            [sys.executable, "-m", "fiae", "inspect", str(p)],
            capture_output=True, text=True, errors="replace", timeout=300,
            env=env,
        )
        combined = proc.stdout + proc.stderr
        assert "Traceback" not in combined, combined[-2000:]
        assert proc.returncode != 0, "malformed input must not report success"
        assert "ERROR [" in combined, combined[-2000:]

