"""CLI dashboard and observability commands (doc 10).

All commands emit FIAE-branded, purple-colored output:
- ``fiae status`` — show current run status and progress
- ``fiae portfolio`` — display selected feature portfolio
- ``fiae report`` — generate a human-readable pipeline report
- ``fiae benchmarks`` — run built-in benchmark suite
"""

from __future__ import annotations

import json
import sys

from . import cli_colors as C


def cmd_status(args) -> int:
    """Display status of a FIAE run directory."""
    import os
    run_path = args.run_path
    if not os.path.isdir(run_path):
        print(f"\n  {C.error_header()} Not a valid run directory: {C.cyan(run_path)}\n",
              file=sys.stderr)
        return 1

    manifest_path = os.path.join(run_path, "run.json")
    if not os.path.isfile(manifest_path):
        print(f"\n  {C.error_header()} No run.json found in {C.cyan(run_path)}\n",
              file=sys.stderr)
        return 1

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if args.json:
        print(json.dumps(manifest, indent=2))
    else:
        print(C.command_header("status", "Run Status"))
        print()
        print(C.key_value("run ID", C.cyan(manifest.get('run_id', '?'))))

        state = manifest.get('completion_state', '?')
        if state == "COMPLETED":
            state_str = f"{C.green('[OK]')} {C.green(state)}"
        elif state in ("FAILED", "CANCELLED"):
            state_str = f"{C.red('[FAIL]')} {C.red(state)}"
        else:
            state_str = f"{C.yellow('[WAIT]')} {C.yellow(state)}"
        print(C.key_value("state", state_str))

        print(C.key_value("source", C.gray(manifest.get('source_fingerprint', '?'))))
        print(C.key_value("pipeline", C.gray(manifest.get('selected_pipeline_hash', '?'))))
        artifacts = manifest.get("artifacts", [])
        print(C.key_value("artifacts", str(len(artifacts))))
        for art in artifacts:
            print(f"    {C.purple('-')} {art.get('kind', '?')}: {C.cyan(art.get('path_or_uri', '?'))}")
        print()
        print(C.footer())

    return 0


def cmd_portfolio(args) -> int:
    """Display a feature portfolio from a JSON report file."""
    report_path = args.report
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    portfolio = report.get("portfolio", [])
    if args.json:
        print(json.dumps(portfolio, indent=2))
    else:
        print(C.command_header("portfolio", "Feature Portfolio"))
        print()
        print(C.key_value("features", str(report.get('portfolio_size', len(portfolio)))))
        print(C.key_value("task", report.get('task', '?')))
        print(C.key_value("target", C.cyan(report.get('target', '?'))))
        print(C.key_value("columns", str(report.get('columns', '?'))))
        print(C.key_value("rows", str(report.get('rows', '?'))))
        print()
        print(C.section("Selected Features"))
        print()
        print(C.table_header("  #", "Operator", "Gain", "F4", "F6", widths=[4, 30, 12, 6, 6]))
        print(C.table_separator([4, 30, 12, 6, 6]))
        for i, member in enumerate(portfolio, 1):
            f4 = C.badge_pass() if member.get("f4_passed") else C.badge_fail()
            f6 = C.badge_pass() if member.get("f6_passed") else C.badge_fail()
            gain = member.get("incremental_gain", 0)
            gain_str = C.green(f"{gain:.4f}") if gain > 0 else C.red(f"{gain:.4f}")
            print(C.table_row(
                f"  {i}",
                C.bold(member.get('operator', '?')),
                gain_str,
                f4,
                f6,
                widths=[4, 30, 12, 6, 6],
            ))
        print()
        print(C.footer())

    return 0


def cmd_report(args) -> int:
    """Generate a human-readable pipeline report."""
    report_path = args.report
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(C.command_header("report", "Pipeline Report"))
    print()
    print(C.key_value("source", report.get('source_id', '?')))
    print(C.key_value("fingerprint", C.gray(report.get('dataset_fingerprint', '?'))))
    print(C.key_value("columns", str(report.get('columns', '?'))))
    print(C.key_value("rows", str(report.get('rows', '?'))))
    print(C.key_value("target", C.cyan(report.get('target', '?'))))
    print(C.key_value("task", C.purple(report.get('task', '?'))))
    print(C.key_value("confidence", f"{report.get('task_confidence', 0):.0%}"))
    print()
    print(C.section("Proposals"))
    print()
    print(C.status_line("Generated", str(report.get('proposals_generated', 0))))
    print(C.status_line("After dedup", str(report.get('proposals_after_dedup', 0))))
    print(C.status_line("Passed F2", str(report.get('funnel_passed_f2', 0))))
    print()
    print(C.section("Portfolio"))
    print()
    print(C.key_value("size", str(report.get('portfolio_size', 0))))
    for i, member in enumerate(report.get("portfolio", []), 1):
        f4 = C.badge_pass() if member.get("f4_passed") else C.badge_fail()
        f6 = C.badge_pass() if member.get("f6_passed") else C.badge_fail()
        gain = member.get("incremental_gain", 0)
        gain_str = C.green(f"{gain:.4f}") if gain > 0 else C.red(f"{gain:.4f}")
        print(f"  {C.purple(f'{i:>2}.')} {C.bold(member.get('operator', '?'))}  "
              f"gain={gain_str}  F4={f4}  F6={f6}")
    print()
    print(C.divider())
    total_t = report.get('total_time_s', 0)
    print(f"  {C.purple('Total time:')} {C.cyan(f'{total_t:.2f}s')}")
    print(C.footer())

    return 0


def cmd_benchmarks(args) -> int:
    """Run built-in benchmark suite."""
    import time
    from .features import all_operators

    print(C.command_header("benchmarks", "FIAE Benchmark Suite"))
    print()

    # Benchmark 1: Operator registration
    t0 = time.monotonic()
    ops = all_operators()
    t1 = time.monotonic()
    print(C.status_line(
        "Operator catalog",
        f"{C.cyan(str(len(ops)))} ops loaded in {C.green(f'{(t1-t0)*1000:.1f}ms')}"
    ))

    # Benchmark 2: Transform throughput
    import random
    rng = random.Random(42)
    data = [rng.gauss(0, 1) for _ in range(10000)]
    t0 = time.monotonic()
    count = 0
    for op in ops:
        if op.arity == "unary" and "continuous_numeric" in op.input_types:
            try:
                op.transform(data[:100])
                count += 1
            except Exception:
                pass
    t1 = time.monotonic()
    print(C.status_line(
        "Transform throughput",
        f"10K rows × {C.cyan(str(count))} ops in {C.green(f'{(t1-t0)*1000:.1f}ms')}"
    ))

    # Benchmark 3: System info
    import sys
    print(C.status_line("Python", C.cyan(sys.version.split()[0])))

    print()
    print(C.divider())
    print(f"  {C.green('Benchmarks complete.')}")
    print(C.footer())
    return 0
