"""The CI failure reporter must actually report.

Job logs and artifacts on this repo's public CI both require
authentication, so ``tools/ci_failure_summary.py`` is the only thing that
turns "exit code 1" into a named failing test. If it silently produced an
empty summary, every future red run would be a dead end again - and, like
the README badge guard it lives beside, nothing else would notice.

So the reporter is tested the way it is used: on synthetic JUnit XML.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL = REPO_ROOT / "tools" / "ci_failure_summary.py"


def _load():
    spec = importlib.util.spec_from_file_location("_fiae_ci_failure_summary", TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_SUMMARY = _load()

_SUITE = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="3" errors="1" failures="1" skipped="0">
  <testcase classname="tests.test_a" name="test_passes"/>
  <testcase classname="tests.test_a" name="test_fails">
    <failure message="assert 1058 == 1151">assert 1058 == 1151</failure>
  </testcase>
  <testcase classname="tests.test_b" name="test_errors">
    <error message="NameError: boom">traceback line</error>
  </testcase>
</testsuite></testsuites>
"""


class TestFailureSummaryRendering:
    def test_names_every_failing_and_erroring_case(self, tmp_path):
        xml = tmp_path / "pytest-results.xml"
        xml.write_text(_SUITE, encoding="utf-8")
        text = _SUMMARY.render(str(xml))
        assert "tests.test_a::test_fails" in text
        assert "tests.test_b::test_errors" in text
        assert "2 failing" in text
        assert "of 3 collected" in text

    def test_carries_the_assertion_message(self, tmp_path):
        xml = tmp_path / "pytest-results.xml"
        xml.write_text(_SUITE, encoding="utf-8")
        text = _SUMMARY.render(str(xml))
        assert "assert 1058 == 1151" in text

    def test_does_not_invent_failures_for_a_green_run(self, tmp_path):
        xml = tmp_path / "pytest-results.xml"
        xml.write_text(
            '<testsuite name="pytest" tests="2"><testcase classname="t" name="a"/>'
            '<testcase classname="t" name="b"/></testsuite>',
            encoding="utf-8",
        )
        text = _SUMMARY.render(str(xml))
        assert "0 failing" in text
        assert "```" not in text

    def test_missing_report_says_so_instead_of_crashing(self, tmp_path):
        text = _SUMMARY.render(str(tmp_path / "nope.xml"))
        assert "never ran" in text

    def test_malformed_report_says_so_instead_of_crashing(self, tmp_path):
        xml = tmp_path / "pytest-results.xml"
        xml.write_text("<testsuite><testcase", encoding="utf-8")
        text = _SUMMARY.render(str(xml))
        assert "not valid XML" in text

    def test_long_messages_are_clipped(self, tmp_path):
        xml = tmp_path / "pytest-results.xml"
        huge = "x" * 9000
        xml.write_text(
            f'<testsuite tests="1"><testcase classname="t" name="a">'
            f'<failure message="{huge}">t</failure></testcase></testsuite>',
            encoding="utf-8",
        )
        text = _SUMMARY.render(str(xml))
        assert len(text) < 9000, "a 9000-char message must be clipped"

    def test_appends_to_the_step_summary_and_exits_zero(self, tmp_path):
        """The reporter must not turn a red run into a differently-red one."""
        xml = tmp_path / "pytest-results.xml"
        xml.write_text(_SUITE, encoding="utf-8")
        summary = tmp_path / "summary.md"
        summary.write_text("# existing\n", encoding="utf-8")
        import os
        env = {**os.environ, "GITHUB_STEP_SUMMARY": str(summary)}
        proc = subprocess.run(
            [sys.executable, str(TOOL), str(xml)],
            capture_output=True, text=True, env=env, timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        written = summary.read_text(encoding="utf-8")
        assert written.startswith("# existing\n"), "must append, not truncate"
        assert "test_fails" in written


@pytest.mark.parametrize("argv", [[], ["missing.xml"]])
def test_cli_never_fails(argv):
    """Exit status stays 0 whatever the report looks like."""
    assert _SUMMARY.main(argv) == 0


class TestFailureAnnotation:
    """`::error::` annotations are readable through the public API.

    The job summary needs a browser; the annotations endpoint does not, so
    a failing test name is machine-readable without authentication. This
    is the only channel that survives the logs-are-private problem.
    """

    def test_emits_an_error_annotation_naming_the_failures(self, tmp_path, capsys):
        xml = tmp_path / "pytest-results.xml"
        xml.write_text(_SUITE, encoding="utf-8")
        assert _SUMMARY.main([str(xml)]) == 0
        out = capsys.readouterr().out
        assert "::error title=pytest::" in out
        assert "tests.test_a::test_fails" in out
        assert "tests.test_b::test_errors" in out

    def test_annotation_carries_what_it_actually_asserted(self, tmp_path, capsys):
        """"exit code 1" is not a diagnosis; the message is."""
        xml = tmp_path / "pytest-results.xml"
        xml.write_text(_SUITE, encoding="utf-8")
        _SUMMARY.main([str(xml)])
        out = capsys.readouterr().out
        assert "assert 1058 == 1151" in out

    def test_emits_nothing_for_a_green_report(self, tmp_path, capsys):
        xml = tmp_path / "pytest-results.xml"
        xml.write_text(
            '<testsuite tests="1"><testcase classname="t" name="a"/></testsuite>',
            encoding="utf-8",
        )
        assert _SUMMARY.main([str(xml)]) == 0
        assert "::error" not in capsys.readouterr().out
