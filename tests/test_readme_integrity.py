"""README integrity lock (D-1 recurrence guard).

Every file path referenced in README.md must exist on disk, and the
headline count badges must match reality.  If this test fails, either
README.md drifted from the repository or the referenced artifact was
removed — fix whichever is wrong, never weaken the check.

Note: the design documents (md/) are intentionally *not* tracked in the
public repository (M35), so README no longer references them and there is
no design-doc count claim to verify.
"""

import os
import re

import pytest


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

BADGE_TOLERANCE = 15


def _collected_count(session) -> int | None:
    """How many tests pytest collected in this run, or None if unknown.

    Must come from ``Session.testscollected``, which pytest sets after
    collection on *every* run.  ``config.item_count`` is populated only
    under ``--collect-only``, so reading it here made this whole file skip
    its own assertion on any ordinary ``pytest tests/`` run - the guard
    passed vacuously while the badge drifted 93 tests out of date.
    """
    return getattr(session, "testscollected", None)


def _assert_badge_tracks(badge: int, collected: int | None) -> None:
    """Raise unless the README badge is within tolerance of reality."""
    if not collected or collected <= 700:
        # Subset run: the badge describes the full suite, so only a
        # full-suite collection says anything about it.
        return
    assert abs(collected - badge) <= BADGE_TOLERANCE, (
        f"README badge says {badge} tests, pytest collected {collected}"
    )


def _readme() -> str:
    with open(os.path.join(_ROOT, "README.md"), encoding="utf-8") as f:
        return f.read()


class TestReadmeFileReferences:
    def test_readme_does_not_reference_untracked_md_docs(self):
        readme = _readme()
        refs = re.findall(r"md/([A-Za-z0-9_.]+\.md)", readme)
        assert refs == [], (
            "README must not reference md/ documents — they are local-only "
            f"(not in the public repo) since M35; found: {sorted(set(refs))}"
        )

    def test_all_repo_path_references_exist(self):
        readme = _readme()
        # Markdown links and inline repo paths (src/, tests/, examples/,
        # benchmarks/, tools/, .github/)
        pattern = re.compile(
            r"\((src|tests|examples|benchmarks|tools|\.github)/[A-Za-z0-9_\-./]+\)"
        )
        refs = {m.group(0)[1:-1] for m in pattern.finditer(readme)}
        assert refs, "README should reference real repo paths"
        missing = [r for r in refs if not os.path.exists(os.path.join(_ROOT, *r.split("/")))]
        assert missing == [], f"README references missing paths: {missing}"

    def test_example_entrypoint_exists(self):
        readme = _readme()
        assert "examples/churn/run_api.py" in readme
        assert os.path.exists(os.path.join(_ROOT, "examples", "churn", "run_api.py"))


class TestReadmeCountClaims:
    def test_tests_badge_matches_collected(self, request):
        readme = _readme()
        m = re.search(r"tests-(\d+)(?:%20|-)passing", readme)
        assert m, "tests badge missing from README"
        _assert_badge_tracks(int(m.group(1)), _collected_count(request.session))


class TestBadgeGuardIsNotVacuous:
    """The badge guard is itself a guard, so it gets the same treatment.

    Its previous failure mode - reading an attribute that is None on a
    normal run, silently skipping the assertion - is invisible to a normal
    suite run, because the suite *is* the thing that skipped.  These tests
    pin the mechanism itself, so the vacuous form cannot come back.
    """

    def test_count_comes_from_testscollected_not_item_count(self):
        class FakeSession:
            # `item_count` exists only under --collect-only; a lookup that
            # reached for it (the pre-fix code) would return None here.
            testscollected = 1151

        assert _collected_count(FakeSession()) == 1151

    def test_the_guard_itself_raises_on_a_drifted_count(self):
        """Drive the real guard method under normal-run conditions.

        `item_count` is None here because that is what pytest reports on an
        ordinary run - the one case the pre-fix lookup mishandled.  Under
        those conditions a drifted collection count must raise, so restoring
        the vacuous lookup makes this test fail instead of quietly skipping.
        """

        class FakeRequest:
            class config:
                item_count = None  # a normal run: --collect-only never happened

            class session:
                testscollected = 1000  # >700, and far from the real badge

        with pytest.raises(AssertionError, match="badge says"):
            TestReadmeCountClaims().test_tests_badge_matches_collected(FakeRequest())

    def test_drift_beyond_tolerance_is_rejected(self):
        with pytest.raises(AssertionError, match="badge says 1058"):
            _assert_badge_tracks(1058, 1151)

    def test_drift_inside_tolerance_is_allowed(self):
        _assert_badge_tracks(1151, 1151 + BADGE_TOLERANCE)
        _assert_badge_tracks(1151, 1151 - BADGE_TOLERANCE)

    def test_subset_run_is_not_judged_against_the_full_suite_badge(self):
        _assert_badge_tracks(1151, 4)  # this file alone: no verdict, no raise
