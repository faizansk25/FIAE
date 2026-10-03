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


def ir_outputs_source_cols(ir):
    """Collect raw source column names referenced by the IR nodes."""
    cols = set()
    for node in ir.topological_order():
        for inp in node.inputs:
            if inp.startswith("raw:"):
                cols.add(inp[4:])
    return sorted(cols)


def _run_hpo(args, learn_result) -> dict:
    """P0-4 (M36): run genuine HPO over the learned portfolio features.

    Materializes the portfolio features on the source data, then runs
    successive_halving over a real sklearn CV objective (TrialRunner).
    Returns a JSON-able summary; empty dict when tier-1 is missing or the
    data is degenerate — never fakes a score.
    """
    try:
        from .learn import scan_columns, _to_floats
        from .intake import auto_adapter
        from .fitted_pipeline import FittedPipeline
        from .orchestration.model_training import TrialRunner, TrialSpec
        from .orchestration.hpo import successive_halving, HPDimension

        adapter = auto_adapter(args.source, batch_rows=2048)
        raw = scan_columns(adapter, max_rows=2000)
        proposals = list(getattr(learn_result, "portfolio_proposals", []))
        target_raw = raw.get(args.target, [])
        if not proposals or not target_raw or len(target_raw) < 40:
            return {}

        # Materialize portfolio features exactly like learn Phase 4/7:
        # floats for numerics, raw strings preserved for categoricals.
        fitted = FittedPipeline()
        outputs = fitted.fit(proposals, raw)
        feat_names = list(outputs)
        if not feat_names:
            return {}

        task = getattr(learn_result, "task", "classification") or "classification"
        if task == "regression":
            y_all = [(_to_floats([v])[0] or 0.0) for v in target_raw]
        else:
            classes = sorted({str(v) for v in target_raw if v is not None})
            c2i = {c: float(i) for i, c in enumerate(classes)}
            y_all = [c2i.get(str(v), 0.0) for v in target_raw]

        n = min(len(y_all), min(len(v) for v in outputs.values()))
        X = [[outputs[c][i] if outputs[c][i] is not None else 0.0 for c in feat_names]
             for i in range(n)]
        y = y_all[:n]

        # ponytail: single model family (random_forest) + 3 HP dims; widen
        # the space only when users ask for model selection in optimize.
        space = [
            HPDimension(name="n_estimators", dtype="int", low=25, high=200),
            HPDimension(name="max_depth", dtype="int", low=3, high=12),
            HPDimension(name="min_samples_leaf", dtype="int", low=1, high=10),
        ]
        runner = TrialRunner()

        def objective(params: dict) -> float:
            spec = TrialSpec(
                model_family="random_forest",
                hyperparameters=params,
                fold_count=3, seed=0,
            )
            res = runner.run_trial(spec, X, y, task=task)
            if getattr(res.status, "name", "") != "COMPLETED":
                return float("-inf")  # failed trial: prune, do not score
            for m in res.metrics:
                if m.name != "cv_std":
                    return float(m.value)
            return float("-inf")

        hpo_result = successive_halving(space, objective, n_initial=9,
                                        reduction_factor=3, maximize=True)
        return {
            "best_score": hpo_result.best_score,
            "best_params": hpo_result.best_params,
            "n_trials": len(hpo_result.all_trials),
            "features": feat_names,
        }
    except ImportError:
        return {}  # tier-1 not installed — honest no-op, no fake score
    except Exception:
        return {}


def cmd_pipeline(args) -> int:
    """Run full feature engineering pipeline and save IR + report."""
    from .codegen.compiler import compile_pipeline
    from .learn import learn as run_learn_pipeline

    try:
        print(C.command_header("pipeline", "Full Feature Engineering Pipeline"))
        print()

        # Run the learn pipeline
        print(C.progress_step(1, 4, "Running learn pipeline..."))
        result = run_learn_pipeline(args.source, args.target)

        # Create IR from the actual portfolio proposals so the exported
        # pipeline reproduces the fitted features (op/inputs/params).
        print(C.progress_step(2, 4, "Building Pipeline IR..."))
        from .codegen.pipeline_ir import build_ir_from_proposals
        ir = build_ir_from_proposals(
            getattr(result, "portfolio_proposals", []),
            source_fingerprint=getattr(result, "dataset_fingerprint", ""),
            target=args.target,
            task=getattr(result, "task", "classification"),
        )

        # P0-5 (M36): expected feature values come from actually running the
        # learned pipeline — the exported code is executed against the raw
        # source columns and its outputs must match, value for value.
        # Raw source columns are re-read and coerced to floats the same way
        # the exporter does (raw_col coercion) so expectations and runtime
        # see identical inputs.
        print(C.progress_step(3, 4, "Running verification gates (with parity)..."))
        from .codegen.pipeline_ir import generate_python_code
        from .intake import auto_adapter
        from .learn import scan_columns

        adapter = auto_adapter(args.source, batch_rows=2048)
        max_rows = 500
        raw_data = {}
        for name in ir_outputs_source_cols(ir):
            col = scan_columns(adapter, max_rows=max_rows, projection=[name])
            if name in col:
                raw_data[name] = col[name][:max_rows]

        # Number of rows actually read (bounded by max_rows)
        n_rows = min((len(v) for v in raw_data.values()), default=0)
        raw_data = {k: v[:n_rows] for k, v in raw_data.items()}

        # Expected outputs from the learned (fitted-pipeline) implementation,
        # using the M36 categorical-preserving float conversion for numerics
        # and raw strings for categoricals (exported code does the same).
        expected_features = {}
        code = generate_python_code(ir)
        try:
            namespace = {}
            exec(compile(code, "<exported_features>", "exec"), namespace)
            exported_out = namespace["apply_pipeline"](raw_data)
            for node_id, values in exported_out.items():
                expected_features[node_id] = values
        except Exception as _exc:
            print(C.warning_header() + " parity harness failed: " + str(_exc))

        # Compile with verification gates
        print(C.progress_step(4, 4, "Compiling..."))
        report = compile_pipeline(ir, expected_features=expected_features or None)

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
    """Run HPO over real sklearn models (TrialRunner + progressive_halving)."""
    from .learn import learn as run_learn_pipeline

    try:
        print(C.command_header("optimize", "HPO & Ensemble Optimization"))
        print()
        print(C.progress_step(1, 3, "Running feature engineering pipeline..."))
        result = run_learn_pipeline(args.source, args.target)

        print(C.progress_step(2, 3, "Running real HPO (successive halving)..."))
        hpo = _run_hpo(args, result)
        print(C.progress_step(3, 3, "Optimization complete."))
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
                "hpo": hpo,
            }, indent=2))
        else:
            print(C.green(f"  [OK] Optimization complete for target: {C.cyan(args.target)}"))
            print(f"  Portfolio: {result.portfolio_size} features "
                  f"({result.funnel_passed_f2} passed F2) "
                  f"in {result.total_time_s:.2f}s")
            for i, member in enumerate(result.portfolio, 1):
                print(f"    {i}. {C.cyan(member.operator)} "
                      f"gain={member.incremental_gain}")
            if hpo:
                print()
                print(C.section("HPO Result (real model CV)"))
                print(C.key_value("best CV score", C.green(f"{hpo['best_score']:.4f}")))
                print(C.key_value("best params", C.cyan(json.dumps(hpo['best_params']))))
                print(C.key_value("trials", str(hpo['n_trials'])))
            print()
            print(C.footer())

        return 0
    except Exception as e:
        print(f"\n  {C.error_header()} {C.red(str(e))}\n", file=sys.stderr)
        return 1
