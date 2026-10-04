"""The mutation registry must stay honest even when it is not being run.

``tools/mutation_check.py`` is the expensive check: it mutates source and
runs pytest for every guard. This file is the cheap check CI can run on
every commit -- it proves each guard still points at real, unique source
text, so a stale anchor can never masquerade as a passing guard.

A stale anchor is the dangerous failure mode: ``_apply_mutation`` refuses
to report "caught" when the anchor is missing, but that only helps if
somebody runs the tool. Here it fails on every commit instead.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_registry():
    path = REPO_ROOT / "tools" / "mutation_check.py"
    spec = importlib.util.spec_from_file_location("_fiae_mutation_check", path)
    assert spec and spec.loader, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules, so the module has
    # to be registered before it is executed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_REGISTRY = _load_registry()
GUARDS = _REGISTRY.GUARDS


def _anchor_variants(text: str) -> list[bytes]:
    return _REGISTRY._variant(text)


class TestRegistryIsWellFormed:
    def test_the_registry_is_not_empty(self):
        assert len(GUARDS) >= 10, (
            "the registry shrank; every audited finding should stay pinned")

    def test_guard_names_are_unique(self):
        names = [g.name for g in GUARDS]
        assert len(set(names)) == len(names), "duplicate guard names"

    def test_every_guard_targets_an_existing_test_file(self):
        for guard in GUARDS:
            for target in guard.targets:
                assert (REPO_ROOT / target).is_file(), (
                    f"{guard.name}: target {target} does not exist")

    def test_every_guard_mutates_an_existing_source_file(self):
        for guard in GUARDS:
            assert (REPO_ROOT / guard.path).is_file(), (
                f"{guard.name}: {guard.path} does not exist")

    def test_every_anchor_matches_exactly_one_place_in_the_source(self):
        """The staleness canary.

        If the surrounding source is refactored and the anchor no longer
        appears, this guard would silently stop proving anything. CI fails
        here instead, at refactor time, rather than the next time somebody
        remembers to run the tool.
        """
        for guard in GUARDS:
            data = (REPO_ROOT / guard.path).read_bytes()
            counts = [data.count(v) for v in _anchor_variants(guard.fixed)]
            matched = sum(counts)
            assert matched == 1, (
                f"{guard.name}: anchor appears {matched} time(s) in "
                f"{guard.path} (LF={counts[0]}, CRLF={counts[1]}); "
                "expected exactly 1")

    def test_every_mutation_actually_changes_something(self):
        for guard in GUARDS:
            assert guard.fixed != guard.reverted, (
                f"{guard.name}: reverting is a no-op")

    def test_every_guard_explains_the_finding_it_pins(self):
        for guard in GUARDS:
            assert len(guard.finding) > 30, (
                f"{guard.name}: finding is too vague to be useful")
            assert guard.milestone, f"{guard.name}: no milestone recorded"

    def test_verify_anchors_passes_on_the_current_tree(self):
        assert _REGISTRY.verify_anchors() == []

    def test_verify_anchors_reports_a_moved_anchor(self):
        moved = (_REGISTRY.Guard(
            name="moved", milestone="T", path="src/fiae/probe.py",
            finding="x" * 40,
            fixed="THIS TEXT IS NOT IN probe.py", reverted="neither is this",
            targets=(),
        ),)
        problems = _REGISTRY.verify_anchors(moved)
        assert len(problems) == 1 and problems[0].startswith("moved:")
