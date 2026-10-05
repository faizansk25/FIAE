"""M40.2: the canonical validate phase must not swallow analysis failures.

Before this milestone, task inference, the identifier scan and the Stage A/B
leakage detectors were each wrapped in ``except Exception: pass``. A crash in
any of them produced an empty ``leakage_flags`` list and a run status
indistinguishable from a genuinely clean dataset -- the most dangerous possible
failure mode for a leakage tool, because "analysis crashed" and "no leakage
exists" produce byte-identical reports.

Each behavioural test drives the real ``phase_validate`` through a fault
injected into its collaborator and asserts the failure is *recorded*. The
structural tests pin the call sites so the swallow cannot quietly return.
"""

import csv
import inspect
import random
from pathlib import Path

import pytest

from fiae.pipeline import canonical
from fiae.pipeline.canonical import phase_validate

MARKER = "leakage detectors failed"


def _write_csv(path: Path, rows: int = 300) -> Path:
    """Synthetic churn table with one exact-copy leaker."""
    rng = random.Random(0)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["age", "tenure", "spend", "leak_copy", "churned"])
        for _ in range(rows):
            churned = rng.randint(0, 1)
            w.writerow([
                rng.randint(18, 80), rng.randint(1, 60),
                rng.random() * 500.0, churned, churned,
            ])
    return path


@pytest.fixture()
def leaky_csv(tmp_path: Path) -> Path:
    return _write_csv(tmp_path / "churn.csv")


@pytest.fixture()
def intake_for(leaky_csv):
    """Build a real intake result for the fixture CSV."""
    def _build():
        ctx = canonical.RunContext(
            run_id="m40_2", config_snapshot={"source": str(leaky_csv)})
        return ctx, canonical.phase_intake(ctx, str(leaky_csv), "churned")
    return _build


@pytest.fixture(autouse=True)
def _restore_leakage_module():
    """Fault injection is module-global; never let it leak between tests."""
    import fiae.problem.leakage as leak_mod

    saved = (leak_mod.detect_deterministic, leak_mod.statistical_triage)
    yield
    leak_mod.detect_deterministic, leak_mod.statistical_triage = saved


def _crash_detectors() -> None:
    """Make both Stage A/B leakage detectors raise on call."""
    import fiae.problem.leakage as leak_mod

    def boom(*_a, **_k):
        raise RuntimeError("injected detector fault")

    leak_mod.detect_deterministic = boom
    leak_mod.statistical_triage = boom


def _detector_flags(result) -> list:
    """Flags produced by the Stage A/B detectors (they carry leakage_class)."""
    return [f for f in result.leakage_flags if "leakage_class" in f]


class TestLeakageDetectorFailureIsRecorded:
    def test_clean_run_finds_the_real_leak(self, intake_for):
        """Happy path: the exact-copy column is detected, nothing recorded."""
        ctx, intake = intake_for()
        result = phase_validate(ctx, intake, "churned")

        assert _detector_flags(result), "fixture leak should be detected"
        assert not [e for e in result.errors if MARKER in e]

    def test_detector_crash_is_recorded_not_swallowed(self, intake_for):
        """The core regression: a crashing detector must be reported."""
        _crash_detectors()
        ctx, intake = intake_for()
        result = phase_validate(ctx, intake, "churned")

        assert _detector_flags(result) == [], "no detector flags under a crash"
        assert any(MARKER in e for e in result.errors), (
            "leakage detector crash was swallowed: "
            f"errors={result.errors!r}"
        )

    def test_crash_and_clean_are_distinguishable(self, intake_for):
        """A crashed run must not be indistinguishable from a clean one.

        Before the fix both cases yielded zero detector flags and no error,
        so no caller could tell them apart.
        """
        # Clean first: fault injection is process-global, so the faulted
        # run must come after the clean one.
        ctx, intake = intake_for()
        clean = phase_validate(ctx, intake, "churned")

        _crash_detectors()
        ctx2, intake2 = intake_for()
        faulted = phase_validate(ctx2, intake2, "churned")

        assert _detector_flags(clean), "clean run should detect the leak"
        assert not _detector_flags(faulted), "faulted run should flag nothing"
        assert any(MARKER in e for e in faulted.errors)
        assert not any(MARKER in e for e in clean.errors)


class TestValidatePhaseHasNoSilentSwallows:
    """Structural pins: the swallow shape must not return to this phase."""

    def test_no_bare_pass_handlers_in_validate(self):
        src = inspect.getsource(phase_validate)
        assert "except Exception:\n        pass" not in src, (
            "phase_validate reintroduced a silent except-and-pass handler")

    def test_every_broad_handler_records(self):
        """Each broad handler must bind the exception and record it."""
        src = inspect.getsource(phase_validate)
        handlers = [
            line.strip() for line in src.splitlines()
            if line.strip().startswith("except Exception")
        ]
        assert handlers, "expected broad handlers in phase_validate"
        for handler in handlers:
            assert handler.endswith("as e:"), (
                f"broad handler does not bind the exception: {handler!r}")

    def test_errors_field_exists_and_defaults_empty(self):
        assert canonical.ValidationResult().errors == []
        assert canonical.ValidationResult().leakage_flags == []
