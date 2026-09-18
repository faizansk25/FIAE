"""GUI job functions for CLI-parity features.

Each function runs on a JobWorkerPool worker thread and returns a
JSON-serializable dict — the GUI equivalent of the corresponding CLI
command:

- leakage_job      <-> fiae leakage SOURCE --target Y
- experience_job   <-> fiae experience show
- tune_job         <-> fiae tune
- codegen_job      <-> fiae codegen --verify | --operator X
- benchmarks_job   <-> fiae benchmarks
- pipeline_job     <-> fiae pipeline SOURCE --target Y (learn + IR + gates)
"""

from __future__ import annotations

import os
from typing import Any, Optional


def leakage_job(source: str, target: str) -> dict:
    """Profile + scan a source and report leakage indicators (doc 03)."""
    from .intake import auto_adapter, ProfileConfig, ProfileMode, profile_source

    adapter = auto_adapter(source)
    profile = profile_source(adapter, ProfileConfig(mode=ProfileMode.FAST))

    findings: list[dict] = []
    col_data: dict[str, list] = {}
    try:
        for batch in adapter.scan():
            for name, values in batch.columns.items():
                col_data.setdefault(name, []).extend(values)
    except Exception:
        pass

    for col in profile.columns:
        if col.name == target:
            continue
        values = col_data.get(col.name, [])
        unique = {v for v in values[:500] if v is not None}
        if len(unique) <= 1:
            findings.append({
                "column": col.name, "type": "constant",
                "severity": "informational",
                "detail": f"only {len(unique)} unique value(s)",
            })
        if col.distinct_ratio > 0.95 and len(unique) >= 10:
            findings.append({
                "column": col.name, "type": "potential_identifier",
                "severity": "review_required",
                "detail": f"distinct_ratio={col.distinct_ratio:.2f} "
                          f"({len(unique)} distinct of {len(values)} scanned)",
            })
        if getattr(col, "null_fraction", 0.0) > 0.6:
            findings.append({
                "column": col.name, "type": "high_null_rate",
                "severity": "warning",
                "detail": f"null_fraction={col.null_fraction:.0%}",
            })

    return {
        "target": target,
        "columns_checked": len(profile.columns) - 1,
        "findings": findings,
        "source_id": adapter.source_id(),
        "fingerprint": profile.dataset_fingerprint,
    }


def experience_job(db_path: Optional[str] = None) -> dict:
    """Summarize the experience store (fiae experience show)."""
    from .experience.store import ExperienceStore, StoreConfig

    db_path = db_path or "experience.db"
    if not os.path.exists(db_path):
        return {"available": False, "db_path": db_path,
                "message": "Run learn jobs to populate the experience store."}
    store = ExperienceStore(StoreConfig(db_path=db_path))
    try:
        cases = store.all_cases()
        recent = []
        for c in cases[-20:]:
            task = ""
            try:
                task = c.context.task.value
            except Exception:
                try:
                    task = str(c.context.task)
                except Exception:
                    task = "?"
            recent.append({
                "case_id": c.case_id,
                "dataset": c.dataset_fingerprint[:16],
                "task": task,
                "success": bool(c.is_success()),
                "failure_tags": [str(t) for t in
                                 getattr(c.diagnosis, "failure_tags", []) or []],
                "wall_time_s": getattr(getattr(c, "cost", None),
                                       "wall_time_s", None),
            })
        successes = sum(1 for c in cases if c.is_success())
        return {
            "available": True,
            "db_path": db_path,
            "total_cases": len(cases),
            "success_rate": (successes / len(cases)) if cases else 0.0,
            "recent": recent,
        }
    finally:
        store.close()


def tune_job(db_path: Optional[str] = None, min_cases: int = 5) -> dict:
    """Funnel-policy tuning evidence (fiae tune)."""
    from .experience.store import ExperienceStore, StoreConfig
    from .tuning import tune_policy_from_experience

    db_path = db_path or "experience.db"
    if not os.path.exists(db_path):
        return {"available": False, "db_path": db_path,
                "message": "Run learn jobs to populate the experience store."}
    store = ExperienceStore(StoreConfig(db_path=db_path))
    try:
        report = tune_policy_from_experience(store, min_cases=min_cases)
        d = report.to_dict()
        d["available"] = True
        d["db_path"] = db_path
        return d
    finally:
        store.close()


def codegen_job(operator: Optional[str] = None) -> dict:
    """Operator contract verification (fiae codegen --verify)."""
    from .codegen.auto_tests import generate_compliance_report

    report = generate_compliance_report()
    report["requested_operator"] = operator
    return report


def benchmarks_job() -> dict:
    """Inline benchmark suite (fiae benchmarks)."""
    import random
    import sys
    import time

    from .features import all_operators

    out: dict[str, Any] = {}
    t0 = time.monotonic()
    ops = all_operators()
    out["catalog_ms"] = round((time.monotonic() - t0) * 1000, 1)
    out["total_operators"] = len(ops)

    rng = random.Random(42)
    data = [rng.gauss(0, 1) for _ in range(10_000)]
    t0 = time.monotonic()
    ran = 0
    for op in ops:
        if op.arity == "unary" and "continuous_numeric" in op.input_types:
            try:
                op.transform(data[:100])
                ran += 1
            except Exception:
                pass
    out["transform_ops"] = ran
    out["transform_ms"] = round((time.monotonic() - t0) * 1000, 1)
    out["python"] = sys.version.split()[0]
    return out


def pipeline_job(source: str, target: str, output_dir: Optional[str] = None) -> dict:
    """Learn + IR + verification gates (fiae pipeline)."""
    from .codegen.pipeline_ir import PipelineIR
    from .codegen.compiler import compile_pipeline
    from .learn import learn as run_learn

    result = run_learn(source, target)
    ir = PipelineIR(
        target=target,
        task=getattr(result, "task", "classification"),
    )
    report = compile_pipeline(ir)

    out_dir = output_dir or "pipeline_output"
    os.makedirs(out_dir, exist_ok=True)
    ir_path = os.path.join(out_dir, "pipeline_ir.json")
    import json as _json
    with open(ir_path, "w", encoding="utf-8") as f:
        _json.dump(ir.to_dict(), f, indent=2, default=str)
    code_path = os.path.join(out_dir, "features.py")
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(report.generated_code)

    summary = report.summary()
    return {
        "pipeline_id": ir.pipeline_id,
        "ir_path": ir_path,
        "code_path": code_path,
        "gates_passed": summary["passed"],
        "gates_total": summary["total_gates"],
        "all_passed": report.all_passed,
        "gates": [
            {"gate": g.gate_name, "passed": g.passed, "message": g.message}
            for g in report.gates
        ],
        "portfolio_size": getattr(result, "portfolio_size", 0),
        "failed_gates": [g.gate_name for g in report.gates if not g.passed],
    }
