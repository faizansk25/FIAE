"""Advanced CLI commands (doc 10).

Additional subcommands for pipeline validation, leakage detection,
experience management, and code generation — all with FIAE purple identity.
"""

from __future__ import annotations

import json
import sys

from . import cli_colors as C


def cmd_validate(args) -> int:
    """Validate a data source against expected schema."""
    from .intake import ProfileConfig, ProfileMode, profile_source, auto_adapter

    try:
        adapter = auto_adapter(args.source)
        config = ProfileConfig(mode=ProfileMode.STANDARD)
        profile = profile_source(adapter, config)

        findings = profile.quality_findings
        if args.json:
            print(json.dumps({
                "fingerprint": profile.dataset_fingerprint,
                "rows": profile.rows_observed,
                "columns": len(profile.columns),
                "findings": findings,
            }, indent=2))
        else:
            print(C.command_header("validate", "Data Source Validation"))
            print()
            print(C.key_value("fingerprint", C.gray(profile.dataset_fingerprint)))
            print(C.key_value("rows", str(profile.rows_observed)))
            print(C.key_value("columns", str(len(profile.columns))))
            if findings:
                print()
                print(C.section(f"Quality Findings ({len(findings)})"))
                print()
                for f in findings:
                    severity = f.get('severity', 'info')
                    badge = C.badge_warn() if severity == "warning" else C.badge_info()
                    col = f.get('column', '?')
                    msg = f.get('message', f.get('detail', f.get('reason', '')))
                    print(f"  {badge} {C.yellow(f.get('type', '?'))}: {col} — {msg}")
            else:
                print()
                print(f"  {C.badge_pass()} No quality issues found.")
            print()
            print(C.footer())

        return 0 if not findings else 1
    except Exception as e:
        print(f"\n  {C.error_header()} {C.red(str(e))}\n", file=sys.stderr)
        return 1


def cmd_leakage(args) -> int:
    """Run leakage detection on a data source."""
    from .intake import ProfileConfig, ProfileMode, profile_source, auto_adapter

    try:
        adapter = auto_adapter(args.source)
        config = ProfileConfig(mode=ProfileMode.STANDARD)
        profile = profile_source(adapter, config)

        all_col_data = {}
        for batch in adapter.scan():
            for name, values in batch.columns.items():
                all_col_data.setdefault(name, []).extend(values)

        target = args.target
        findings = []

        for col in profile.columns:
            if col.name == target:
                continue
            col_data = all_col_data.get(col.name, [])
            unique = {v for v in col_data[:500] if v}
            if len(unique) <= 1:
                findings.append({
                    "column": col.name,
                    "type": "constant",
                    "severity": "informational",
                    "detail": f"only {len(unique)} unique values",
                })

            if col.distinct_ratio > 0.95 and len(col_data) > 100:
                findings.append({
                    "column": col.name,
                    "type": "potential_identifier",
                    "severity": "review_required",
                    "detail": f"distinct_ratio={col.distinct_ratio:.2f}",
                })

        if args.json:
            print(json.dumps({"findings": findings, "target": target}, indent=2))
        else:
            print(C.command_header("leakage", f"Leakage Detection — target: {C.cyan(target)}"))
            print()
            print(C.key_value("columns checked", str(len(profile.columns) - 1)))
            if findings:
                print()
                print(C.section(f"Findings ({len(findings)})"))
                print()
                for f in findings:
                    severity = f['severity']
                    if severity == "review_required":
                        badge = C.badge_warn()
                    else:
                        badge = C.badge_info()
                    print(f"  {badge} {C.purple(f['column'])}: {C.yellow(f['type'])} — {f['detail']}")
            else:
                print()
                print(f"  {C.badge_pass()} No leakage indicators found.")
            print()
            print(C.footer())

        return 0
    except Exception as e:
        print(f"\n  {C.error_header()} {C.red(str(e))}\n", file=sys.stderr)
        return 1


def cmd_tune(args) -> int:
    """Tune the FunnelPolicy from experience store history."""
    import os
    from .experience.store import ExperienceStore, StoreConfig
    from .tuning import tune_policy_from_experience

    db_path = args.db_path or "experience.db"
    if not os.path.exists(db_path):
        print(f"\n  {C.error_header()} Experience store not found: {C.cyan(db_path)}")
        print(f"  {C.gray('Run fiae learn to populate the store first.')}\n")
        return 1

    store = ExperienceStore(StoreConfig(db_path=db_path))
    report = tune_policy_from_experience(store, min_cases=args.min_cases)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
        return 0

    print(C.command_header("tune", "Policy Auto-Tuning"))
    print()
    print(C.key_value("cases analyzed", str(report.cases_analyzed)))
    print(C.key_value("success rate", f"{report.success_rate:.0%}"))
    if report.tag_counts:
        print(C.key_value("failure tags",
                          ", ".join(f"{k}({v})" for k, v in
                                    sorted(report.tag_counts.items()))))
    print()
    if not report.adjustments:
        print(C.section("No adjustments — policy unchanged"))
        print(C.gray("  (insufficient evidence or all gates healthy)"))
    else:
        print(C.section(f"Adjustments ({len(report.adjustments)})"))
        for a in report.adjustments:
            print(f"  {C.purple('-')} {C.bold(a.field)}: "
                  f"{C.red(str(a.old))} -> {C.green(str(a.new))}")
            print(f"      {C.gray(a.reason)}")
    print()

    if args.output:
        policy_dict = {
            f: getattr(report.policy, f)
            for f in report.policy.__dataclass_fields__
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(policy_dict, f, indent=2)
        print(C.key_value("policy saved", C.cyan(args.output)))
        print()
        print(C.gray(f"  Use it with: fiae learn <data> --target <y> "
                     f"--policy {args.output}"))
        print()
    print(C.footer())
    return 0


def cmd_connect(args) -> int:
    """Connect to any data source: auto-detect, verify, and profile.

    One command for all 25+ supported sources — files, URLs, cloud
    storage, databases, warehouses, DataFrames, and APIs.
    """
    from .intake import (
        auto_adapter, list_supported_sources,
        ProfileConfig, ProfileMode, profile_source,
    )

    if args.list:
        print(C.command_header("connect", "Supported Data Sources"))
        print()
        for category, items in list_supported_sources().items():
            print(C.section(category))
            for item in items:
                print(f"  {C.purple('•')} {item}")
            print()
        print(C.footer())
        return 0

    source = args.source
    print(C.command_header("connect", f"Connecting to {C.cyan(source)}"))
    print()
    try:
        t0 = __import__("time").monotonic()
        adapter = auto_adapter(source, **_connect_kwargs(args))
        sid = adapter.source_id()
        kind = sid.split("|", 1)[0]
        print(C.key_value("adapter", C.green(type(adapter).__name__)))
        print(C.key_value("type", kind))
        print(C.key_value("source_id", C.gray(sid)))

        rows = adapter.estimate_rows()
        if rows is not None:
            print(C.key_value("rows (est)", f"{rows:,}"))
        nbytes = adapter.estimate_bytes()
        if nbytes is not None:
            print(C.key_value("size (est)", f"{nbytes / 1e6:.1f} MB"))
        if adapter.supports_pushdown():
            print(C.key_value("pushdown", C.green("supported")))
        print()

        # Quick profile to prove the connection is real
        print(C.section("Verification Profile"))
        print()
        config = ProfileConfig(mode=ProfileMode.FAST)
        profile = profile_source(adapter, config)
        elapsed = __import__("time").monotonic() - t0
        print(C.key_value("status", f"{C.badge_pass()} connected and readable"))
        print(C.key_value("rows observed", f"{profile.rows_observed:,}"))
        print(C.key_value("columns", str(len(profile.columns))))
        print(C.key_value("fingerprint", C.gray(profile.dataset_fingerprint[:24] + "...")))
        if profile.quality_findings:
            print(C.key_value("quality findings", C.yellow(str(len(profile.quality_findings)))))
        print()
        sem_types = sorted({c.semantic_type.value for c in profile.columns})
        print(C.key_value("semantic types", ", ".join(C.cyan(t) for t in sem_types)))
        print()
        print(C.key_value("time", f"{elapsed:.2f}s"))
        print()
        print(C.key_value("next", C.gray(f"fiae inspect '{source}'")))
        print(C.key_value("", C.gray(f"fiae learn '{source}' --target <column>")))
        print()
        print(C.footer())
        return 0
    except Exception as e:
        print(f"\n  {C.error_header()} {C.red(str(e))}\n", file=sys.stderr)
        print(C.gray("  See all supported sources with: fiae connect --list"), file=sys.stderr)
        return 1


def _connect_kwargs(args) -> dict:
    """Build adapter kwargs from connect CLI flags."""
    kwargs = {}
    if getattr(args, "table", None):
        kwargs["table"] = args.table
    if getattr(args, "query", None):
        kwargs["query"] = args.query
    if getattr(args, "header", None):
        kwargs["headers"] = dict(h.split(":", 1) for h in args.header)
    return kwargs


def cmd_experience(args) -> int:
    """Display experience store contents."""

    import os
    from .experience.store import ExperienceStore, StoreConfig

    db_path = args.db_path or "experience.db"
    if not os.path.exists(db_path):
        print(f"\n  {C.badge_info()} Experience store not found: {C.cyan(db_path)}")
        print(f"  {C.gray('Run fiae learn to populate the store.')}\n")
        return 1

    try:
        store = ExperienceStore(StoreConfig(db_path=db_path))
        try:
            cases = store.all_cases()
            recent = cases[-10:]
            if args.json:
                from dataclasses import asdict
                print(json.dumps([asdict(c) for c in recent], indent=2, default=str))
            else:
                print(C.command_header("experience", "Experience Store"))
                print()
                print(C.key_value("total cases", str(len(cases))))
                print(C.key_value("showing", f"last {len(recent)}"))
                print()
                if recent:
                    print(C.table_header("  ID", "Task", "Status", widths=[30, 20, 12]))
                    print(C.table_separator([30, 20, 12]))
                    for c in recent:
                        task_val = c.context.task.value if hasattr(c.context.task, 'value') else str(c.context.task)
                        status = C.green("success") if c.is_success() else C.red("failed")
                        print(C.table_row(
                            f"  {c.case_id[:28]}",
                            task_val,
                            status,
                            widths=[30, 20, 12],
                        ))
                else:
                    print(f"  {C.gray('No cases recorded yet.')}")
                print()
                print(C.footer())
        except Exception as e:
            print(f"  {C.yellow('Warning:')} Could not list cases: {e}")
        return 0
    except Exception as e:
        print(f"\n  {C.error_header()} {C.red(str(e))}\n", file=sys.stderr)
        return 1


def cmd_codegen(args) -> int:
    """Generate test scaffold for an operator."""
    from .codegen.auto_tests import (
        generate_operator_test, generate_all_scaffolds,
        generate_compliance_report,
    )

    if args.operator:
        from .features.registry import get_operator
        try:
            op = get_operator(args.operator)
            scaffold = generate_operator_test(op)
            if args.json:
                print(json.dumps({
                    "operator": scaffold.operator_name,
                    "test_code": scaffold.test_code,
                    "description": scaffold.description,
                }, indent=2))
            else:
                print(C.command_header("codegen", f"Operator: {C.cyan(scaffold.operator_name)}"))
                print()
                print(f"  {C.purple('Description:')} {scaffold.description}")
                print()
                print(C.section("Generated Test Code"))
                print()
                for line in scaffold.test_code.split("\n"):
                    print(f"  {line}")
                print()
                print(C.footer())
            return 0
        except Exception as e:
            print(f"\n  {C.error_header()} {C.red(str(e))}\n", file=sys.stderr)
            return 1

    if args.verify:
        report = generate_compliance_report()
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(C.command_header("codegen", "Operator Contract Compliance"))
            print()
            print(C.key_value("total operators", str(report['total_operators'])))
            print(C.key_value("compliant", C.green(str(report['compliant']))))
            print(C.key_value("non-compliant", C.red(str(report['non_compliant']))))
            print(C.key_value("compliance rate", f"{report['compliance_rate']:.1%}"))
            if report['violations']:
                print()
                print(C.section("Violations"))
                for name, viols in report['violations'].items():
                    print(f"  {C.red('[FAIL]')} {C.purple(name)}: {', '.join(viols)}")
            else:
                print()
                print(f"  {C.badge_pass()} All operators compliant.")
            print()
            print(C.footer())
        return 0

    scaffolds = generate_all_scaffolds()
    if args.json:
        print(json.dumps([{
            "operator": s.operator_name,
            "description": s.description,
        } for s in scaffolds], indent=2))
    else:
        print(C.command_header("codegen", "Operator Test Scaffolds"))
        print()
        print(C.key_value("generated", f"{C.cyan(str(len(scaffolds)))} test scaffolds"))
        print()
        for s in scaffolds:
            print(f"  {C.purple('•')} {C.bold(s.operator_name)}: {s.description}")
        print()
        print(C.footer())

    return 0
