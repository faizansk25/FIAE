"""README integrity lock (D-1 recurrence guard).

Every file path referenced in README.md must exist on disk, and the
headline count badges must match reality.  If this test fails, either
README.md drifted from the repository or the referenced artifact was
removed — fix whichever is wrong, never weaken the check.
"""

import os
import re


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _readme() -> str:
    with open(os.path.join(_ROOT, "README.md"), encoding="utf-8") as f:
        return f.read()


class TestReadmeFileReferences:
    def test_all_md_references_exist(self):
        readme = _readme()
        refs = set(re.findall(r"md/([A-Za-z0-9_.]+\.md)", readme))
        assert refs, "README must reference the design documents"
        missing = [r for r in refs if not os.path.exists(os.path.join(_ROOT, "md", r))]
        assert missing == [], f"README references missing docs: {missing}"

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
    def test_tests_badge_matches_collected(self, pytestconfig):
        readme = _readme()
        m = re.search(r"tests-(\d+)(?:%20|-)passing", readme)
        assert m, "tests badge missing from README"
        badge = int(m.group(1))
        collected = getattr(pytestconfig, "item_count", None)
        if collected and collected > 700:
            # Full-suite run: badge must track collection within tolerance
            # for tests added/removed between README edits.
            assert abs(collected - badge) <= 15, (
                f"README badge says {badge} tests, pytest collected {collected}"
            )

    def test_design_doc_count_claim_matches_disk(self):
        readme = _readme()
        m = re.search(r"built on (\d+) design documents", readme)
        assert m, "design-doc count claim missing"
        claimed = int(m.group(1))
        actual = len([
            f for f in os.listdir(os.path.join(_ROOT, "md"))
            if re.match(r"^\d{2}_.*\.md$", f) and not f.startswith("99_")
        ])
        assert claimed == actual, (
            f"README claims {claimed} design documents; disk has {actual}"
        )
