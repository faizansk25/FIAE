#!/usr/bin/env python3
"""Mutation check: prove the suite catches the *reversion* of each fix.

A green test suite only says the tests pass. It does not say they pass
*because* the fix is present. That gap was found the hard way: the M40
commit that introduced ``is_better`` shipped with 1134 green tests, and
re-inserting the hard-coded ``score > best[1]`` that the fix replaced
left every one of them green. The fix was real, but nothing was pinning
it, so it could have been reverted silently.

This tool closes that class of hole. Each guard names a fix, the exact
source text that expresses it, the text that expresses the pre-fix bug,
and the tests that must fail when the bug is put back. A guard whose
tests still pass is a hole, and the tool exits non-zero.

Deliberately not general-purpose mutation testing (no random mutants, no
per-line instrumentation): those tools spend most of their time proving
that nobody wrote a test for line 87. Here every mutation is a reversion
of a defect an external audit actually found, which is the only category
that has ever mattered in this repo.

Usage:
    python tools/mutation_check.py             # run every guard
    python tools/mutation_check.py --list      # show guards, run nothing
    python tools/mutation_check.py --only NAME # run one guard (repeatable)

Safety: refuses to run on a dirty tracked working tree, restores each
mutated file from its original bytes in a finally block, and verifies the
restore byte-for-byte before reporting. It never commits or pushes.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Guard:
    """One fix, the bug it replaced, and the tests that must notice."""

    name: str
    milestone: str
    path: str
    finding: str
    # Text that expresses the fix. Must appear exactly once.
    fixed: str
    # Text that expresses the pre-fix bug, substituted for `fixed`.
    reverted: str
    # pytest args; the mutation is "caught" only if pytest exits 1
    # (tests failed). Exit 0 means nothing noticed -- that is the hole.
    targets: tuple[str, ...]


# ---------------------------------------------------------------------------
# The registry. One entry per audited finding worth keeping pinned.
# ---------------------------------------------------------------------------

GUARDS: tuple[Guard, ...] = (
    Guard(
        name="m39_feature_name_includes_params",
        milestone="M39",
        path="src/fiae/fitted_pipeline.py",
        finding="Feature identity ignored params, so two proposals over the "
                "same input collided and one fit state/output was overwritten.",
        fixed='    base = proposal.op + "(" + "_".join(_logical_inputs(proposal)) + ")"\n'
              "    if proposal.params:\n",
        reverted='    base = proposal.op + "(" + "_".join(_logical_inputs(proposal)) + ")"\n'
                 "    if False:\n",
        targets=("tests/test_m39_semantics.py",
                 "tests/test_m8_fitted_pipeline.py"),
    ),
    Guard(
        name="m39_codegen_fails_closed",
        milestone="M39",
        path="src/fiae/pipeline/canonical.py",
        finding="phase_codegen wrote export/features.py and set "
                "export_code_path before checking report.all_passed, so a "
                "failed verification still published a normal-looking artifact.",
        fixed="        if not report.all_passed:\n",
        reverted="        if False:\n",
        targets=("tests/test_m39_semantics.py",),
    ),
    Guard(
        name="m39_1_state_keyed_by_machine_identity",
        milestone="M39.1",
        path="src/fiae/fitted_pipeline.py",
        finding="Persisted state was keyed by the human-readable display "
                "name, so reformatting a name silently invalidated saved state.",
        fixed='    return feature_id_for({\n        "op": proposal.op,\n'
              '        "inputs": _logical_inputs(proposal),\n'
              '        "params": dict(proposal.params),\n    })\n',
        reverted='    return canonical_feature_id(proposal)  # SABOTAGE: display name as identity\n',
        targets=("tests/test_m39_1_identity_and_status.py",),
    ),
    Guard(
        name="m39_2_exact_stats_requires_exhaustion",
        milestone="M39.2",
        path="src/fiae/intake/profiler.py",
        finding="exact_stats came from the *requested* mode, so an EXACT run "
                "that hit max_rows or the time budget still claimed exactness.",
        fixed="        and stop_reason is StopReason.SOURCE_EXHAUSTED\n",
        reverted="        and True  # SABOTAGE: requested mode is not exhaustion\n",
        targets=("tests/test_m39_2_profiling_truth.py",),
    ),
    Guard(
        name="m39_2_estimate_rows_excludes_header",
        milestone="M39.2",
        path="src/fiae/intake/csv_source.py",
        finding="estimate_rows counted the header line, so every CSV "
                "over-reported its row count by exactly one.",
        fixed="        if getattr(self, \"has_header\", None):\n            estimate -= 1\n",
        reverted="        # SABOTAGE: header line counted as a data row\n",
        targets=("tests/test_m39_2_profiling_truth.py",),
    ),
    Guard(
        name="m39_2_learn_does_not_report_sampled_rows_as_total",
        milestone="M39.2",
        path="src/fiae/learn.py",
        finding="learn() set rows_in_source = profile.rows_observed, which is "
                "capped by the profiling budget, so a 2.5M-row file reported "
                "'Rows: 100000'.",
        fixed="    report.rows_in_source = profile.coverage.rows_total\n",
        reverted="    report.rows_in_source = profile.rows_observed  # SABOTAGE: sample as total\n",
        targets=("tests/test_m39_2_profiling_truth.py",),
    ),
    Guard(
        name="m40_folds_bounded_by_limiting_unit",
        milestone="M40",
        path="src/fiae/orchestration/model_training.py",
        finding="Fold count was capped by the number of classes, so every "
                "binary 5-fold request silently ran 2 folds.",
        fixed="        n_splits = min(n_folds, min(Counter(y).values()))\n",
        reverted="        n_splits = min(n_folds, len(set(y)))  # SABOTAGE: class count\n",
        targets=("tests/test_m40_model_correctness.py",),
    ),
    Guard(
        name="m40_hpo_selection_obeys_direction",
        milestone="M40",
        path="src/fiae/pipeline/canonical.py",
        finding="phase_hpo compared scores with a hard-coded '>', so a MINIMIZE "
                "metric selected the worst model. is_better() existed but "
                "nothing proved phase_hpo called it.",
        fixed="                if best is None or is_better(m, best[3]):\n",
        reverted="                if best is None or (m.value > best[3].value):  # SABOTAGE\n",
        targets=("tests/test_m40_model_correctness.py",),
    ),
    Guard(
        name="m40_loss_un_negated_at_sklearn_boundary",
        milestone="M40",
        path="src/fiae/orchestration/model_training.py",
        finding="FIAE published sklearn's negated loss under the name 'mse', "
                "making a positive loss look like a better model.",
        fixed="        scores = -raw if spec.negated else raw\n",
        reverted="        scores = raw  # SABOTAGE: keep sklearn's neg_ convention\n",
        targets=("tests/test_m40_model_correctness.py",),
    ),
    Guard(
        name="m40_multiclass_probe_is_one_vs_rest",
        milestone="M40",
        path="src/fiae/probe.py",
        finding="One ridge was solved against the raw class index and that "
                "single prediction vector was scored against every OVR target, "
                "so it carried no class information; pure noise looked useful.",
        fixed="            errs.append(_multiclass_ovr_sq_error(\n"
              "                _multiclass_ovr_scores(columns, y, train, val, classes, lam),\n"
              "                y, val, classes,\n            ))\n",
        reverted="            errs.append(0.0)  # SABOTAGE: shared prediction vector\n",
        targets=("tests/test_m40_model_correctness.py",),
    ),
    Guard(
        name="m40_ensemble_no_fabricated_weights",
        milestone="M40",
        path="src/fiae/pipeline/canonical.py",
        finding="When build_ensemble() rejected every trial, phase_ensemble "
                "synthesised 1/n weights over those same rejected trials and "
                "reported an ensemble that was never built.",
        fixed="                result.members = 0\n                result.weights = []\n",
        reverted="                result.members = len(trials)  # SABOTAGE: equal weights\n"
                 "                result.weights = [1.0 / len(trials)] * len(trials)\n",
        targets=("tests/test_m40_model_correctness.py",),
    ),
    Guard(
        name="m40_trial_runner_populates_fold_metrics",
        milestone="M40",
        path="src/fiae/orchestration/model_training.py",
        finding="TrialRunner never wrote fold_metrics, so ensemble "
                "eligibility rejected every genuine trial.",
        fixed="                fold_metrics=fold_metrics,\n",
        reverted="                fold_metrics=[],  # SABOTAGE: no per-fold evidence\n",
        targets=("tests/test_m40_model_correctness.py",),
    ),
    Guard(
        name="m38_5_prune_also_drops_failed_jobs",
        milestone="M38",
        path="src/fiae/server.py",
        finding="_prune_locked only pruned COMPLETED jobs, so repeated "
                "failures grew the registry without bound.",
        fixed='                    if j["state"] in ("COMPLETED", "FAILED")]\n',
        reverted='                    if j["state"] == "COMPLETED"]  # SABOTAGE\n',
        targets=("tests/test_m38_correctness.py", "tests/test_server.py"),
    ),
    Guard(
        # The only guard whose subject is a test.  Its failure mode was
        # invisible from inside a normal run: the guard skipped its own
        # assertion, so the suite that should have caught the drift was the
        # suite doing the skipping.
        name="m40_1_readme_badge_count_is_collected_not_vacuous",
        milestone="M40.1",
        path="tests/test_readme_integrity.py",
        finding="The README badge guard read pytestconfig.item_count, which "
                "is only set under --collect-only, so on every ordinary run "
                "it skipped its own assertion and the badge silently drifted "
                "93 tests out of date.",
        fixed="        _assert_badge_tracks(int(m.group(1)), _collected_count(request.session))\n",
        reverted="        _assert_badge_tracks(\n"
                 "            int(m.group(1)),\n"
                 "            getattr(request.config, \"item_count\", None),  # SABOTAGE: vacuous\n"
                 "        )\n",
        targets=("tests/test_readme_integrity.py",),
    ),
)


# ---------------------------------------------------------------------------
# Machinery
# ---------------------------------------------------------------------------

class GuardError(RuntimeError):
    """The harness could not run a guard (not the same as an uncaught bug)."""


def _variant(text: str) -> list[bytes]:
    """Anchor as bytes, LF and CRLF, because checkouts differ per platform."""
    raw = text.encode("utf-8")
    return [raw, raw.replace(b"\n", b"\r\n")]


def _git() -> str:
    """Absolute path to git, so PATH lookup is resolved once, not per call."""
    git = shutil.which("git")
    if git is None:
        raise GuardError("git is not on PATH; the mutation harness needs it")
    return git


def _dirty_tracked_files() -> list[str]:
    proc = subprocess.run(  # noqa: S603 - argv is fixed, shell=False, no interpolation
        [_git(), "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise GuardError(f"git status failed: {proc.stderr.strip()}")
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _apply_mutation(path: Path, guard: Guard) -> bytes:
    """Write the reverted file; return the original bytes for restore."""
    original = path.read_bytes()
    data = original
    hit = False
    for fixed_b, reverted_b in (
        (_variant(guard.fixed)[0], _variant(guard.reverted)[0]),
        (_variant(guard.fixed)[1], _variant(guard.reverted)[1]),
    ):
        count = data.count(fixed_b)
        if count == 1:
            data = data.replace(fixed_b, reverted_b, 1)
            hit = True
            break
        if count > 1:
            raise GuardError(
                f"anchor matches {count} times in {guard.path}; "
                "make it unique so the guard is deterministic")
    if not hit:
        raise GuardError(
            f"anchor not found in {guard.path} -- the source moved, so this "
            "guard is stale and would have reported a false pass")
    if data == original:
        raise GuardError("mutation was a no-op")
    path.write_bytes(data)
    return original


def _run_targets(guard: Guard) -> tuple[int, str, tuple[str, ...]]:
    cmd = [sys.executable, "-m", "pytest", *guard.targets,
           "-p", "no:warnings", "--tb=no", "-q", "-rf"]
    proc = subprocess.run(  # noqa: S603 - argv built from the registry, shell=False
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    # Which tests failed, not merely that pytest exited non-zero: a guard
    # "caught" by an unrelated import error is a false pass.
    failed = tuple(
        line.split(" ", 1)[1].strip()
        for line in proc.stdout.splitlines()
        if line.startswith("FAILED ")
    )
    tail = [ln for ln in proc.stdout.splitlines() if ln.strip()][-1:] or [""]
    return proc.returncode, tail[0].strip(), failed


def verify_anchors(guards: tuple[Guard, ...] = GUARDS) -> list[str]:
    """Check every anchor still matches exactly once. Mutates nothing.

    This is the cheap half of the tool, and the one CI can afford on every
    commit: a refactor that moves an anchor would otherwise turn a guard into
    a permanent no-op that still reports success.
    """
    problems: list[str] = []
    for guard in guards:
        path = REPO_ROOT / guard.path
        if not path.is_file():
            problems.append(f"{guard.name}: {guard.path} does not exist")
            continue
        data = path.read_bytes()
        counts = [data.count(v) for v in _variant(guard.fixed)]
        matched = sum(counts)
        if matched != 1:
            problems.append(
                f"{guard.name}: anchor appears {matched} time(s) in "
                f"{guard.path} (LF={counts[0]}, CRLF={counts[1]}); "
                "expected exactly 1")
    return problems


CAUGHT, MISSED, ERRORED = "caught", "MISSED", "error"


def run_guard(guard: Guard) -> tuple[str, str]:
    """Mutate, run the targets, restore. Returns (verdict, detail)."""
    path = REPO_ROOT / guard.path
    if not path.is_file():
        return ERRORED, f"{guard.path} does not exist"
    try:
        original = _apply_mutation(path, guard)
    except GuardError as exc:
        return ERRORED, str(exc)

    try:
        rc, last, failed = _run_targets(guard)
    finally:
        path.write_bytes(original)
        if path.read_bytes() != original:
            # Never continue against a file we failed to put back.
            raise GuardError(
                f"FAILED TO RESTORE {guard.path}; check it out from git")

    if rc == 1:
        if not failed:
            return ERRORED, f"pytest exited 1 but named no failing test: {last}"
        return CAUGHT, f"{len(failed)} test(s): " + ", ".join(failed[:2])
    if rc == 0:
        return MISSED, (
            f"tests still pass with the bug restored ({last or 'no output'}) "
            "-- nothing is pinning this fix")
    return ERRORED, f"pytest exited {rc} (not a test failure): {last}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--list", action="store_true",
                        help="print the registry and exit")
    parser.add_argument("--verify", action="store_true",
                        help="check every anchor still matches; mutates nothing")
    parser.add_argument("--only", action="append", default=[],
                        metavar="NAME", help="run only these guards")
    parser.add_argument("--verbose", action="store_true",
                        help="name the tests that caught each mutation")
    args = parser.parse_args(argv)

    if args.list:
        for g in GUARDS:
            print(f"{g.name:46} {g.milestone:8} {g.path}")
        print(f"\n{len(GUARDS)} guards")
        return 0

    if args.verify:
        problems = verify_anchors()
        for line in problems:
            print(f"  STALE  {line}", file=sys.stderr)
        print(f"verify: {len(GUARDS)} guards, {len(problems)} stale")
        return 1 if problems else 0

    guards = GUARDS
    if args.only:
        wanted = set(args.only)
        guards = tuple(g for g in GUARDS if g.name in wanted)
        missing = wanted - {g.name for g in GUARDS}
        if missing:
            print(f"unknown guard(s): {sorted(missing)}", file=sys.stderr)
            return 2
    if not guards:
        print("no guards selected", file=sys.stderr)
        return 2

    dirty = _dirty_tracked_files()
    if dirty:
        print("refusing to run: tracked working tree is dirty", file=sys.stderr)
        for line in dirty:
            print(f"  {line}", file=sys.stderr)
        return 2

    print(f"mutation check: {len(guards)} guards\n")
    missed, errored, caught = [], [], []
    for guard in guards:
        started = time.monotonic()
        verdict, detail = run_guard(guard)
        took = time.monotonic() - started
        bucket = {CAUGHT: caught, MISSED: missed, ERRORED: errored}[verdict]
        bucket.append((guard, detail))
        mark = {CAUGHT: "ok  ", MISSED: "MISS", ERRORED: "ERR "}[verdict]
        print(f"  {mark} {guard.name:46} {took:6.1f}s")
        if verdict != CAUGHT or args.verbose:
            print(f"       {detail}")

    after = _dirty_tracked_files()
    if after:
        print(f"\nworking tree is dirty after the run: {after}",
              file=sys.stderr)
        return 2

    print(f"\n{len(caught)} caught, {len(missed)} missed, {len(errored)} errored")
    for guard, detail in missed:
        print(f"\n  UNPINNED: {guard.name} ({guard.milestone})\n"
              f"    {guard.finding}\n    {detail}")
    return 1 if (missed or errored) else 0


if __name__ == "__main__":
    sys.exit(main())
