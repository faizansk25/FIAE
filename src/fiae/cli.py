"""Minimal stdlib CLI: `fiae inspect SOURCE` and `fiae analyze SOURCE --target Y`.

Normative source: doc 10 (CLI commands; machine-readable --json). The CLI calls
the same core engine and implements no ML logic itself (invariant 18).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

from .errors import ErrorCode, FIAEError
from .identity import banner, splash, version_line
from .intake import ProfileConfig, ProfileMode, profile_source
from .contracts import SemanticType
from . import cli_colors as C


def _profile_to_dict(profile) -> dict[str, Any]:
    d = dataclasses.asdict(profile)
    d["columns"] = [
        {**col, "semantic_type": col["semantic_type"].value
         if hasattr(col["semantic_type"], "value") else col["semantic_type"]}
        for col in d["columns"]
    ]
    return d


@dataclass
class _Single:
    """Wrapper so dataclasses.asdict can serialize a single ColumnProfile."""

    columns: list
    dataset_fingerprint: str = "n/a"
    rows_observed: int = 0
    rows_estimated: int = 0
    meta_features: dict = field(default_factory=dict)
    source_cost: dict = field(default_factory=dict)
    quality_findings: list = field(default_factory=list)


def _color_value(val: Any, force: Optional[bool] = None) -> str:
    """Colorize a value for display."""
    s = str(val)
    if isinstance(val, bool):
        return C.green(s, force) if val else C.red(s, force)
    if isinstance(val, (int, float)):
        return C.cyan(s, force)
    return s


def cmd_inspect(args: argparse.Namespace) -> int:
    from .intake import auto_adapter
    adapter = auto_adapter(args.source, batch_rows=args.chunk_size)
    config = ProfileConfig(mode=ProfileMode(args.mode.upper()), max_rows=args.max_rows, scheduler=args.scheduler)
    profile = profile_source(adapter, config)
    payload = {
        "source_id": adapter.source_id(),
        "dialect": adapter.dialect_report(),
        "profile": _profile_to_dict(profile),
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    else:
        p = profile
        print(C.command_header("inspect", "Data Source Profile"))
        print()
        print(C.key_value("source", adapter.source_id()))
        print(C.key_value("rows", f"{p.rows_observed} observed / {p.rows_estimated} estimated"))
        print(C.key_value("fingerprint", C.gray(p.dataset_fingerprint)))
        print(C.key_value("columns", str(len(p.columns))))
        print()
        print(C.section("Column Profiles"))
        print()
        print(C.table_header(
            "  Name", "Type", "Semantic", "Null%", "Distinct",
            widths=[24, 10, 28, 8, 8],
        ))
        print(C.table_separator([24, 10, 28, 8, 8]))
        for col in p.columns:
            sem_color = C.purple if col.semantic_type == SemanticType.IDENTIFIER else C.cyan
            null_color = C.red if col.null_fraction > 0.5 else (C.yellow if col.null_fraction > 0.1 else C.green)
            print(C.table_row(
                f"  {col.name}",
                col.physical_dtype,
                sem_color(col.semantic_type.value),
                null_color(f"{col.null_fraction:.1%}"),
                str(col.distinct_estimate),
                widths=[24, 10, 28, 8, 8],
            ))
        if p.quality_findings:
            print()
            print(C.section("Quality Findings"))
            print()
            for f in p.quality_findings:
                badge = C.badge_warn() if f["severity"] == "warning" else C.badge_info()
                print(f"  {badge} {C.yellow(f['type'])}: {C.cyan(f['column'])}")
        print()
        print(C.footer())
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from .intake import auto_adapter
    adapter = auto_adapter(args.source, batch_rows=args.chunk_size)
    config = ProfileConfig(mode=ProfileMode(args.mode.upper()), max_rows=args.max_rows, scheduler=args.scheduler)
    profile = profile_source(adapter, config)

    # Problem-definition gates (steps 0021-0023): target exists, not constant.
    target_profile: Optional[Any] = None
    for col in profile.columns:
        if col.name == args.target:
            target_profile = col
            break
    if target_profile is None:
        raise FIAEError(
            code=ErrorCode.TARGET_MISSING,
            safe_message="Target column not found in source.",
            component="analyze",
            evidence={"target": args.target},
        )
    if target_profile.distinct_estimate <= 1:
        raise FIAEError(
            code=ErrorCode.TARGET_INVALID,
            safe_message="Target column is constant.",
            component="analyze",
            evidence={"target": args.target},
        )

    # Task inference + metric routing (doc 03, M2).
    from .problem import infer_task, route_metrics

    target_values: list[str] = []
    for batch in adapter.scan(projection=[args.target]):
        target_values.extend(batch.columns[args.target])
        if len(target_values) >= 5000:
            break
    inference = infer_task(
        target_profile,
        target_values=target_values if target_values else None,
        time_key=args.time_key,
        horizon=args.horizon,
    )
    templates = route_metrics(
        inference.task, is_imbalanced=inference.is_imbalanced
    )

    payload = {
        "source_id": adapter.source_id(),
        "target": args.target,
        "target_profile": _profile_to_dict(_Single(columns=[target_profile]))["columns"][0],
        "problem": {
            "task": inference.task.value,
            "confidence": inference.confidence,
            "reasons": inference.reasons,
            "ambiguities": inference.ambiguities,
            "positive_class": (
                None if inference.positive_class is None else str(inference.positive_class)
            ),
            "positive_rate": inference.positive_rate,
            "is_imbalanced": inference.is_imbalanced,
            "metrics": [
                {"name": t.name, "direction": t.direction.value, "note": t.note}
                for t in templates
            ],
        },
        "profile": _profile_to_dict(profile),
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    else:
        print(C.command_header("analyze", "Problem Definition & Task Inference"))
        print()
        print(C.key_value("target", f"{C.cyan(args.target)} ({C.purple(target_profile.semantic_type.value)})"))
        print(C.key_value("task", f"{C.bold(inference.task.value)} (confidence={C.green(f'{inference.confidence:.0%}')})"))
        if inference.positive_class is not None:
            print(C.key_value("positive", f"{inference.positive_class} rate={C.cyan(f'{inference.positive_rate:.3f}')}"))
        if inference.is_imbalanced:
            print(C.key_value("note", C.yellow("⚠ imbalanced target")))
        print(C.key_value("metrics", ", ".join(C.cyan(t.name) for t in templates)))
        if inference.ambiguities:
            print()
            print(C.section("Ambiguities"))
            for a in inference.ambiguities:
                print(f"  {C.yellow('⚠')} {a}")
        print()
        print(C.section("Dataset Summary"))
        print()
        print(C.key_value("rows", str(profile.rows_observed)))
        print(C.key_value("fingerprint", C.gray(profile.dataset_fingerprint)))
        print(C.key_value("quality findings", str(len(profile.quality_findings))))
        for f in profile.quality_findings:
            badge = C.badge_warn() if f["severity"] == "warning" else C.badge_info()
            print(f"    {badge} {f['type']}: {f['column']}")
        print()
        print(C.footer())
    return 0


def cmd_learn(args: argparse.Namespace) -> int:
    from .learn import learn, LearnConfig
    cfg = LearnConfig(max_rows=args.max_rows, sample_rows=args.sample_rows,
                      max_portfolio_features=args.max_features, seed=args.seed,
                      batch_rows=args.chunk_size, scheduler=args.scheduler)
    policy_path = getattr(args, "policy", None)
    if policy_path:
        import json as _json
        from .funnel import FunnelPolicy
        with open(policy_path, "r", encoding="utf-8") as _f:
            cfg.funnel_policy = FunnelPolicy(**_json.load(_f))
    track = getattr(args, "track", None)
    if track:
        from .experiment.tracking import TrackingConfig
        cfg.tracking = TrackingConfig(
            backend=track,
            experiment=getattr(args, "experiment", None) or "fiae",
            run_name=getattr(args, "run_name", None),
        )
    report = learn(args.source, args.target, config=cfg)
    payload = report.to_dict()
    if getattr(args, 'json', False):
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    else:
        r = report
        print(C.command_header("learn", "Feature Intelligence Pipeline"))
        print()
        print(C.key_value("source", r.source_id))
        print(C.key_value("target", f"{C.cyan(r.target)} ({C.purple(r.task)}, confidence={C.green(f'{r.task_confidence:.0%}')})"))
        if r.positive_class is not None:
            print(C.key_value("positive", str(r.positive_class)))
        print(C.key_value("rows", str(r.rows_in_source)))
        print(C.key_value("columns", str(r.columns_in_source)))
        print(C.key_value("metrics", ", ".join(C.cyan(m) for m in r.metrics)))
        print()
        print(C.section("Pipeline Summary"))
        print()
        print(C.status_line("Proposals generated", str(r.proposals_generated)))
        print(C.status_line("After deduplication", str(r.proposals_after_dedup)))
        print(C.status_line("Passed F2 funnel", str(r.funnel_passed_f2)))
        print(C.status_line("Portfolio size", str(r.portfolio_size)))
        if r.portfolio:
            print()
            print(C.section("Selected Features"))
            print()
            for i, m in enumerate(r.portfolio, 1):
                status = C.badge_pass() if m.f4_passed and m.f6_passed else C.badge_fail()
                gain_str = C.green(f"{m.incremental_gain:.6f}") if m.incremental_gain and m.incremental_gain > 0 else C.red(f"{m.incremental_gain:.6f}")
                stab = f"{m.fold_stability:.4f}" if m.fold_stability is not None else "N/A"
                print(f"  {C.purple(f'{i:>2}.')} {C.bold(m.operator):<30} gain={gain_str}  stability={C.cyan(stab)}  {status}")
        print()
        print(C.divider())
        print(f"  {C.purple('Total time:')} {C.cyan(f'{r.total_time_s:.2f}s')}")
        print(C.footer())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fiae",
        description=f"{C.purple('fiae')} — Feature Intelligence & Architecture Engine",
    )
    parser.add_argument(
        "--version", action="store_true", help="print version and exit"
    )
    sub = parser.add_subparsers(dest="command")

    p_banner = sub.add_parser(
        "banner", help="Print the FIAE CLI wordmark"
    )
    p_banner.add_argument(
        "--no-color", action="store_true", help="plain text wordmark"
    )
    p_banner.add_argument(
        "--color", action="store_true", help="force color even if not a TTY"
    )
    p_banner.add_argument(
        "--no-tagline", action="store_true", help="omit tagline + version"
    )
    p_banner.set_defaults(func=cmd_banner)

    p_inspect = sub.add_parser("inspect", help="Profile a data source")
    p_inspect.add_argument("source")
    p_inspect.add_argument("--mode", default="fast", choices=["fast", "standard", "exact"])
    p_inspect.add_argument("--max-rows", type=int, default=100_000, help="Maximum rows to process (default: 100,000)")
    p_inspect.add_argument("--chunk-size", type=int, default=2048, help="Rows per processing batch (memory control, default: 2048)")
    p_inspect.add_argument("--scheduler", default="serial", choices=["serial", "threads"], help="Profiling scheduler (default: serial)")
    p_inspect.add_argument("--json", action="store_true")
    p_inspect.set_defaults(func=cmd_inspect)

    p_analyze = sub.add_parser("analyze", help="Profile source and verify target")
    p_analyze.add_argument("source")
    p_analyze.add_argument("--target", required=True)
    p_analyze.add_argument("--mode", default="fast", choices=["fast", "standard", "exact"])
    p_analyze.add_argument("--time-key", default=None)
    p_analyze.add_argument("--horizon", default=None)
    p_analyze.add_argument("--max-rows", type=int, default=100_000, help="Maximum rows to process (default: 100,000)")
    p_analyze.add_argument("--chunk-size", type=int, default=2048, help="Rows per processing batch (memory control, default: 2048)")
    p_analyze.add_argument("--scheduler", default="serial", choices=["serial", "threads"], help="Profiling scheduler (default: serial)")
    p_analyze.add_argument("--json", action="store_true")
    p_analyze.set_defaults(func=cmd_analyze)

    p_learn = sub.add_parser("learn", help="Run full feature intelligence pipeline")
    p_learn.add_argument("source")
    p_learn.add_argument("--target", required=True)
    p_learn.add_argument("--mode", default="fast", choices=["fast", "standard", "exact"])
    p_learn.add_argument("--max-rows", type=int, default=100_000, help="Maximum rows to process (default: 100,000)")
    p_learn.add_argument("--chunk-size", type=int, default=2048, help="Rows per processing batch (memory control, default: 2048)")
    p_learn.add_argument("--scheduler", default="serial", choices=["serial", "threads"], help="Profiling scheduler (default: serial)")
    p_learn.add_argument("--max-features", type=int, default=8)
    p_learn.add_argument("--sample-rows", type=int, default=5000)
    p_learn.add_argument("--seed", type=int, default=0)
    p_learn.add_argument("--track", default=None, choices=["local", "mlflow", "wandb"],
                         help="Experiment tracking backend (default: none)")
    p_learn.add_argument("--experiment", default="fiae",
                         help="Experiment/project name for tracking (default: fiae)")
    p_learn.add_argument("--run-name", default=None,
                         help="Custom run name for tracking")
    p_learn.add_argument("--policy", default=None,
                         help="Path to a FunnelPolicy JSON file (see: fiae tune)")
    p_learn.add_argument("--json", action="store_true")
    p_learn.set_defaults(func=cmd_learn)

    # ── Advanced commands ────────────────────────────────────────────
    from .cli_advanced import (cmd_validate, cmd_leakage, cmd_experience,
                               cmd_codegen, cmd_tune, cmd_connect)

    p_connect = sub.add_parser(
        "connect", help="Connect to any data source (auto-detect + verify)")
    p_connect.add_argument("source", nargs="?", default=None,
                           help="Path, URI, or connection string")
    p_connect.add_argument("--list", action="store_true",
                           help="List all supported data source types")
    p_connect.add_argument("--table", default=None,
                           help="Table name (databases/warehouses)")
    p_connect.add_argument("--query", default=None,
                           help="Custom SQL query (databases/warehouses)")
    p_connect.add_argument("--header", action="append", default=None,
                           help="HTTP header 'Name: Value' (APIs; repeatable)")
    p_connect.add_argument("--json", action="store_true")
    p_connect.set_defaults(func=cmd_connect)

    p_validate = sub.add_parser("validate", help="Validate data source schema")
    p_validate.add_argument("source")
    p_validate.add_argument("--json", action="store_true")
    p_validate.set_defaults(func=cmd_validate)

    p_leakage = sub.add_parser("leakage", help="Detect feature leakage")
    p_leakage.add_argument("source")
    p_leakage.add_argument("--target", required=True)
    p_leakage.add_argument("--json", action="store_true")
    p_leakage.set_defaults(func=cmd_leakage)

    p_experience = sub.add_parser("experience", help="Experience store management")
    exp_sub = p_experience.add_subparsers(dest="exp_command")
    p_exp_show = exp_sub.add_parser("show", help="Display experience store contents")
    p_exp_show.add_argument("--db-path", default=None)
    p_exp_show.add_argument("--json", action="store_true")
    p_exp_show.set_defaults(func=cmd_experience)

    p_tune = sub.add_parser("tune", help="Tune funnel policy from experience store evidence")
    p_tune.add_argument("--db-path", default=None, help="Experience store path (default: experience.db)")
    p_tune.add_argument("--min-cases", type=int, default=5, help="Minimum cases before adjusting (default: 5)")
    p_tune.add_argument("-o", "--output", default=None, help="Write tuned policy JSON to this file")
    p_tune.add_argument("--json", action="store_true")
    p_tune.set_defaults(func=cmd_tune)

    p_codegen = sub.add_parser("codegen", help="Code generation & operator tests")
    p_codegen.add_argument("--operator", default=None, help="Generate test for specific operator")
    p_codegen.add_argument("--verify", action="store_true", help="Verify all operator contracts")
    p_codegen.add_argument("--json", action="store_true")
    p_codegen.set_defaults(func=cmd_codegen)

    # ── Dashboard commands ───────────────────────────────────────────
    from .cli_dashboard import cmd_status, cmd_portfolio, cmd_report, cmd_benchmarks

    p_status = sub.add_parser("status", help="Show run status")
    p_status.add_argument("run_path")
    p_status.add_argument("--json", action="store_true")
    p_status.set_defaults(func=cmd_status)

    p_portfolio = sub.add_parser("portfolio", help="Display feature portfolio")
    p_portfolio.add_argument("report")
    p_portfolio.add_argument("--json", action="store_true")
    p_portfolio.set_defaults(func=cmd_portfolio)

    p_report = sub.add_parser("report", help="Generate pipeline report")
    p_report.add_argument("report")
    p_report.add_argument("--json", action="store_true")
    p_report.set_defaults(func=cmd_report)

    p_benchmarks = sub.add_parser("benchmarks", help="Run benchmark suite")
    p_benchmarks.set_defaults(func=cmd_benchmarks)

    # ── Pipeline commands ────────────────────────────────────────────
    from .cli_pipeline import cmd_pipeline, cmd_export, cmd_optimize

    p_pipeline = sub.add_parser("pipeline", help="Run full pipeline with IR export")
    p_pipeline.add_argument("source")
    p_pipeline.add_argument("--target", required=True)
    p_pipeline.add_argument("--output", default=None)
    p_pipeline.add_argument("--json", action="store_true")
    p_pipeline.set_defaults(func=cmd_pipeline)

    p_export = sub.add_parser("export", help="Export Pipeline IR to sklearn project")
    p_export.add_argument("ir_path")
    p_export.add_argument("--project", default=None)
    p_export.add_argument("--json", action="store_true")
    p_export.set_defaults(func=cmd_export)

    p_optimize = sub.add_parser("optimize", help="Run HPO + ensemble selection")
    p_optimize.add_argument("source")
    p_optimize.add_argument("--target", required=True)
    p_optimize.add_argument("--json", action="store_true")
    p_optimize.set_defaults(func=cmd_optimize)

    p_serve = sub.add_parser(
        "serve", help="Start REST API + web dashboard (localhost)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8420)
    p_serve.add_argument("--runs-dir", default="runs")
    p_serve.add_argument("--max-workers", type=int, default=4,
                         help="Learn-job worker threads (default: 4)")
    p_serve.add_argument("--max-queue", type=int, default=100,
                         help="Pending job queue depth; full queue returns 429 (default: 100)")
    p_serve.add_argument("--rate-limit", type=float, default=50.0,
                         help="Max requests/second per client IP (default: 50)")
    p_serve.set_defaults(func=cmd_serve)

    p_ui = sub.add_parser(
        "ui", help="Start the FIAE web GUI (browser opens automatically)")
    p_ui.add_argument("--host", default="127.0.0.1")
    p_ui.add_argument("--port", type=int, default=8420)
    p_ui.add_argument("--runs-dir", default="runs")
    p_ui.add_argument("--max-workers", type=int, default=4,
                      help="Learn-job worker threads (default: 4)")
    p_ui.add_argument("--max-queue", type=int, default=100,
                      help="Pending job queue depth; full queue returns 429 (default: 100)")
    p_ui.add_argument("--rate-limit", type=float, default=50.0,
                      help="Max requests/second per client IP (default: 50)")
    p_ui.set_defaults(func=cmd_ui)

    return parser


def cmd_serve(args) -> int:
    """Start the FIAE REST API + web dashboard server."""
    from .server import serve
    serve(host=args.host, port=args.port, runs_dir=args.runs_dir,
          max_workers=args.max_workers, max_queue_size=args.max_queue,
          rate_limit=args.rate_limit)
    return 0


def cmd_ui(args) -> int:
    """Start the FIAE web GUI and open it in the default browser."""
    import threading
    import webbrowser

    from .server import serve
    url = f"http://{args.host}:{args.port}"
    if args.host == "127.0.0.1":
        # Open the browser once the server is accepting connections.
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    serve(host=args.host, port=args.port, runs_dir=args.runs_dir,
          max_workers=args.max_workers, max_queue_size=args.max_queue,
          rate_limit=args.rate_limit)
    return 0


def cmd_banner(args: argparse.Namespace) -> int:
    """Print the FIAE CLI wordmark / splash."""
    # --color wins over --no-color; otherwise let the stream decide.
    if args.no_color and not args.color:
        force = False
    elif args.color:
        force = True
    else:
        force = None
    if args.no_tagline:
        print(banner(force_color=force))
    else:
        print(splash(force_color=force))
    return 0


def _enable_windows_ansi() -> None:
    """Enable ANSI/VT processing on Windows 10+ consoles."""
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


def main(argv: Optional[list[str]] = None) -> int:
    _enable_windows_ansi()
    try:
        import colorama
        colorama.init(strip=False)
    except Exception:
                pass
    parser = build_parser()
    if argv is None:
        argv_ = sys.argv[1:]
    else:
        argv_ = list(argv)
    if "--version" in argv_:
        print(version_line())
        return 0
    # Show branded banner + help when run with no arguments
    if not argv_:
        print()
        print(splash())
        print()
        # Subtle donation footer
        import os
        if os.environ.get("FIAE_NO_DONATE_REMINDER", "").lower() not in ("1", "true", "yes"):
            print(C.gray("  FIAE is free to use. If it helps you, please consider supporting"))
            print(C.gray("  its development: https://github.com/sponsors/fiae  |  https://ko-fi.com/fiae"))
            print()
        parser.print_help()
        return 0
    args = parser.parse_args(argv_)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except FIAEError as err:
        if getattr(args, "json", False):
            print(json.dumps({"error": err.to_dict()}, ensure_ascii=False))
        else:
            print(f"\n  {C.error_header()} [{C.red(err.code.value)}] {err.safe_message}\n",
                  file=sys.stderr)
        return 2
