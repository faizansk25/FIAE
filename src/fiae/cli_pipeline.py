"""Pipeline and optimization CLI commands (doc 10, doc 15).

- ``fiae pipeline SOURCE --target Y`` — run full pipeline, save IR + report
- ``fiae export PIPELINE_IR --project out/`` — export to standalone project
- ``fiae optimize SOURCE --target Y`` — run HPO + ensemble selection

All commands output FIAE-branded, purple-colored results.
"""

from __future__ import annotations

import json
import os
import sys

from . import cli_colors as C


def cmd_pipeline(args) -> int:
    """Run full feature engineering pipeline and save IR + report."""
    from .codegen.compiler import compile_pipeline
    from .learn import learn as run_learn_pipeline

    try:
        print(C.command_header("pipeline", "Full Feature Engineering Pipeline"))
        print()

        # Run the learn pipeline
        print(C.progress_step(1, 3, "Running learn pipeline..."))
        result = run_learn_pipeline(args.source, args.target)

        # Create IR from the actual portfolio proposals so the exported
        # pipeline reproduces the fitted features (op/inputs/params).
        print(C.progress_step(2, 3, "Building Pipeline IR..."))
        from .codegen.pipeline_ir import build_ir_from_proposals
        ir = build_ir_from_proposals(
            getattr(result, "portfolio_proposals", []),
            source_fingerprint=getattr(result, "dataset_fingerprint", ""),
            target=args.target,
            task=getattr(result, "task", "classification"),
        )

        # Compile with verification gates
        print(C.progress_step(3, 3, "Running verification gates..."))
        report = compile_pipeline(ir)

        output_dir = args.output or "pipeline_output"
        os.makedirs(output_dir, exist_ok=True)

        ir_path = os.path.join(output_dir, "pipeline_ir.json")
        with open(ir_path, "w") as f:
            json.dump(ir.to_dict(), f, indent=2, default=str)

        report_path = os.path.join(output_dir, "compiler_report.json")
        with open(report_path, "w") as f:
            json.dump(report.summary(), f, indent=2, default=str)

        code_path = os.path.join(output_dir, "features.py")
        with open(code_path, "w") as f:
            f.write(report.generated_code)

        print()
        if args.json:
            print(json.dumps({
                "ir_path": ir_path,
                "report_path": report_path,
                "code_path": code_path,
                "gates_passed": report.summary()["passed"],
                "gates_total": report.summary()["total_gates"],
                "all_passed": report.all_passed,
            }, indent=2))
        else:
            print(C.section("Pipeline Compiled"))
            print()
            print(C.key_value("pipeline ID", C.cyan(ir.pipeline_id)))
            print(C.key_value("IR saved", C.cyan(ir_path)))
            print(C.key_value("code saved", C.cyan(code_path)))
            s = report.summary()
            gates_badge = C.badge_pass() if report.all_passed else C.badge_fail()
            print(C.key_value("verification", f"{s['passed']}/{s['total_gates']} gates  {gates_badge}"))
            if not report.all_passed:
                print()
                print(C.section("Failed Gates"))
                for g in report.gates:
                    if not g.passed:
                        print(f"  {C.red('[FAIL]')} {C.purple(g.gate_name)}: {g.message}")
            print()
            print(C.footer())

        return 0 if report.all_passed else 1
    except Exception as e:
        print(f"\n  {C.error_header()} {C.red(str(e))}\n", file=sys.stderr)
        return 1


def cmd_export(args) -> int:
    """Export a PipelineIR to a standalone sklearn project."""
    import json
    from .codegen.pipeline_ir import PipelineIR
    from .codegen.compiler import generate_sklearn_project

    try:
        print(C.command_header("export", "Export to Standalone sklearn Project"))
        print()

        with open(args.ir_path) as f:
            ir_dict = json.load(f)
        ir = PipelineIR.from_dict(ir_dict)

        output_dir = args.project or "project"
        files = generate_sklearn_project(ir, output_dir)

        for rel_path, content in files.items():
            full_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(content)

        if args.json:
            print(json.dumps({
                "output_dir": output_dir,
                "files": list(files.keys()),
                "pipeline_id": ir.pipeline_id,
            }, indent=2))
        else:
            print(C.key_value("output", C.cyan(output_dir)))
            print(C.key_value("pipeline", C.gray(ir.pipeline_id)))
            print()
            print(C.section("Generated Files"))
            print()
            for rel_path in files:
                size = len(files[rel_path])
                print(f"  {C.purple('•')} {C.bold(rel_path)} {C.gray(f'({size} bytes)')}")
            print()
            print(C.green(f"  Export complete — {len(files)} files written."))
            print()
            print(C.footer())

        return 0
    except Exception as e:
        print(f"\n  {C.error_header()} {C.red(str(e))}\n", file=sys.stderr)
        return 1


def cmd_optimize(args) -> int:
    """Run HPO + ensemble selection."""
    from .learn import learn as run_learn_pipeline

    try:
        print(C.command_header("optimize", "HPO & Ensemble Optimization"))
        print()
        print(C.progress_step(1, 2, "Running feature engineering pipeline..."))
        result = run_learn_pipeline(args.source, args.target)
        print(C.progress_step(2, 2, "Optimization complete."))
        print()
        if args.json:
            print(json.dumps({
                "target": args.target,
                "status": "completed",
                "task": result.task,
                "proposals_generated": result.proposals_generated,
                "funnel_passed_f2": result.funnel_passed_f2,
                "portfolio_size": result.portfolio_size,
                "total_time_s": result.total_time_s,
            }, indent=2))
        else:
            print(C.green(f"  [OK] Optimization complete for target: {C.cyan(args.target)}"))
            print(f"  Portfolio: {result.portfolio_size} features "
                  f"({result.funnel_passed_f2} passed F2) "
                  f"in {result.total_time_s:.2f}s")
            for i, member in enumerate(result.portfolio, 1):
                print(f"    {i}. {C.cyan(member.operator)} "
                      f"gain={member.incremental_gain}")
            print()
            print(C.footer())

        return 0
    except Exception as e:
        print(f"\n  {C.error_header()} {C.red(str(e))}\n", file=sys.stderr)
        return 1
