#!/usr/bin/env python3
"""Name the failing tests where a human can actually read them.

A red CI run on a public repo is otherwise a dead end: job logs *and* run
artifacts both require authentication, so the browser and the API show
"Process completed with exit code 1" and nothing else. The JUnit XML is
already on disk from ``--junitxml``, so this turns it into a job summary -
visible on the run page, and in the API's check-run output.

Used by the ``test`` and ``coverage`` jobs as an ``if: failure()`` step.
Never changes the exit status: reporting a failure is not the step's job.

Usage:
    python tools/ci_failure_summary.py pytest-results.xml
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET

MAX_DETAIL_CHARS = 1500


def _bad_cases(root: ET.Element) -> list[ET.Element]:
    return [
        tc for tc in root.iter("testcase")
        if tc.find("failure") is not None or tc.find("error") is not None
    ]


def _detail(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return (node.get("message") or node.text or "").strip()


def render(xml_path: str) -> str:
    """Markdown for the job summary. Never raises on a malformed report."""
    lines = ["## Test failures", ""]
    if not os.path.exists(xml_path):
        return "\n".join([
            *lines,
            f"No JUnit report at `{xml_path}` - the suite never ran.",
        ]) + "\n"
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        return "\n".join([
            *lines,
            f"`{xml_path}` is not valid XML ({exc}); cannot summarise.",
        ]) + "\n"

    bad = _bad_cases(root)
    suite = next(root.iter("testsuite"), None)
    total = (suite.get("tests") if suite is not None else None) or len(
        list(root.iter("testcase"))
    )
    lines.append(f"**{len(bad)} failing** of {total} collected.\n")
    for tc in bad:
        node = tc.find("failure")
        if node is None:
            node = tc.find("error")
        lines.append(f"- `{tc.get('classname')}::{tc.get('name')}`")
        detail = _detail(node)
        if detail:
            clipped = detail[:MAX_DETAIL_CHARS].replace("\n", "\n  ")
            lines.append("  ```\n  " + clipped + "\n  ```")
    return "\n".join(lines) + "\n"


def _fail_node_ids(xml_path: str) -> list[str]:
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError):
        return []
    return [
        f"{tc.get('classname')}::{tc.get('name')}" for tc in _bad_cases(root)
    ]


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    xml_path = args[0] if args else "pytest-results.xml"
    text = render(xml_path)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(text)
    print(text)

    # The job summary is readable in a browser, but check-run annotations are
    # also readable through the unauthenticated API - so a failing test name
    # is visible to tooling, not just to whoever is looking at the run page.
    failing = _fail_node_ids(xml_path)
    if failing:
        names = ", ".join(failing[:5])
        more = f" (+{len(failing) - 5} more)" if len(failing) > 5 else ""
        print(f"::error title=pytest::{len(failing)} failing: {names}{more}".replace("\n", " "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
