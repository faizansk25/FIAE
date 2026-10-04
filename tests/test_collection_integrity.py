"""A test module that vanishes takes its whole file with it, silently.

`tests/test_phase2_hardening.py` carried a module-level
``pytest.importorskip("pyarrow")``. pyarrow is in neither the `dev` nor the
`tier1` extra, so on every CI job that line skipped the entire module at
collection time: 24 tests - export/runtime equivalence, schema
preservation, the SSRF guard, the JSON depth guard, artifact-hash lineage -
simply did not exist there. Nothing failed. Nothing was reported. The suite
reported "1147 passed" while covering less than the README claimed, and the
count badge could not see it either, because a skipped module is collected
as zero items on some paths and the drift was below the tolerance.

This is the general hazard: a module-level skip is not one skip, it is every
test in the file, and it is invisible unless you count. So the rule is
mechanical - no module-level skip in `tests/`, ever. Missing optional
dependencies belong on the individual test that needs them, where the skip
is one line of output instead of a hole in the suite.

Static by design: a behavioural check would need to hide each optional
dependency in turn, which is a slower and less complete version of reading
the file.
"""

import ast
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent


def _module_level_skip_lines(tree: ast.Module) -> list[int]:
    found = []
    for node in tree.body:  # module level only - nested skips are per-test
        call = None
        if isinstance(node, ast.Expr):
            call = node.value  # a bare `pytest.importorskip(...)`
        elif isinstance(node, ast.Assign):
            # `pyarrow = pytest.importorskip("pyarrow")` is the shape the
            # original bug actually used; matching only bare calls missed it.
            call = node.value
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Name):
            continue
        if func.value.id != "pytest":
            continue
        is_importorskip = func.attr == "importorskip"
        is_module_skip = func.attr == "skip" and any(
            kw.arg == "allow_module_level" and getattr(kw.value, "value", False)
            for kw in call.keywords
        )
        if is_importorskip or is_module_skip:
            found.append(node.lineno)
    return found


def _module_level_skips(source: str, name: str = "<test>") -> list[int]:
    """Line numbers of skips that execute at import time."""
    return _module_level_skip_lines(ast.parse(source, filename=name))


def _test_modules() -> list[Path]:
    return sorted(p for p in TESTS_DIR.glob("test_*.py"))


def _skips_in(path: Path) -> list[int]:
    return _module_level_skips(path.read_text(encoding="utf-8"), path.name)


def _declares_tests(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name.startswith("test_")
        for n in ast.walk(tree)
    )


class TestNoModuleLevelSkips:
    def test_there_are_test_modules_to_check(self):
        assert len(_test_modules()) > 40, "glob found suspiciously few test files"

    def test_no_test_module_skips_itself_at_import_time(self):
        offenders = {
            path.name: _skips_in(path)
            for path in _test_modules()
            if _skips_in(path)
        }
        assert not offenders, (
            "module-level skip deletes every test in the file, silently, on "
            "any environment without that dependency - move it to the "
            f"individual test: {offenders}"
        )

    def test_the_detector_actually_detects(self):
        """A guard that finds nothing because it looks for nothing.

        The sweep above passes vacuously if this function quietly stops
        matching - so it is fed the exact shapes it exists to catch. Same
        lesson as the badge guard, one level down: the checker needs its own
        checker.
        """
        assert _module_level_skips('pyarrow = pytest.importorskip("pyarrow")\n') == [1]
        assert _module_level_skips('pytest.importorskip("pyarrow")\n') == [1]
        assert _module_level_skips(
            'import pytest\npytest.skip("nope", allow_module_level=True)\n'
        ) == [2]
        # A per-test skip is what the rule asks for: not flagged.
        assert _module_level_skips(
            "def test_x():\n    pa = pytest.importorskip('pyarrow')\n"
        ) == []
        # Nor is a module-level pytest.skip without the opt-in flag.
        assert _module_level_skips('pytest.skip("later")\n') == []

    def test_every_test_module_actually_declares_tests(self):
        """A module that cannot contribute items is the same hole, quieter.

        Renaming or emptying a file, or leaving a helper module under
        ``test_*.py``, costs coverage without costing a single red mark.
        """
        empty = [p.name for p in _test_modules() if not _declares_tests(p)]
        assert not empty, f"test_*.py files that declare no tests: {empty}"
