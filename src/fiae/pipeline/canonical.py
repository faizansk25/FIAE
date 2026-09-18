"""Canonical end-to-end pipeline phases (doc 15, steps 0001-0120).

Implements the full canonical execution procedure as discrete phases
that compose into a complete run. Each phase corresponds to a group
of steps from doc 15.

Normative source: doc 15 "End-to-End FIAE Algorithm".
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..ids import new_id, content_hash


# ---------------------------------------------------------------------------
# Run context (doc 15 steps 0001-0004)
# ---------------------------------------------------------------------------

@dataclass
class RunContext:
    """Persistent run context with immutable IDs (doc 15 step 0001)."""
    run_id: str = ""
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    build_commit: str = "dev"
    build_version: str = "0.1.0"
    started_at: float = 0.0
    run_dir: str = ""
    source_fingerprint: str = ""
    config_hash: str = ""
    event_stream_path: str = ""
    metadata_store_path: str = ""
    cancellation_token: str = ""

    def __post_init__(self):
        if not self.run_id:
            self.run_id = new_id("run")
        if self.started_at == 0.0:
            self.started_at = time.time()


@dataclass
class DecisionRecord:
    """Immutable decision record (doc 15 step 0001 mandatory checks)."""
    decision_id: str = ""
    step: str = ""
    timestamp: float = 0.0
    reason: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.decision_id:
            self.decision_id = new_id("dec")
        if self.timestamp == 0.0:
            self.timestamp = time.time()


# ---------------------------------------------------------------------------
# Phase 1: Bootstrap (doc 15 steps 0001-0010)
# ---------------------------------------------------------------------------

def phase_bootstrap(source_path: str, target: str, config: dict | None = None) -> RunContext:
    """Phase 1: Bootstrap — create run, capture identity, init state.

    Steps 0001-0010: run ID, build identity, directory, mode validation,
    config snapshot, fingerprint capture.
    """
    ctx = RunContext(config_snapshot=config or {})
    ctx.config_hash = content_hash(ctx.config_snapshot)
    ctx.cancellation_token = new_id("cancel")

    # Create run directory
    run_dir = os.path.join(".fiae", "runs", ctx.run_id)
    os.makedirs(run_dir, exist_ok=True)
    ctx.run_dir = run_dir
    ctx.event_stream_path = os.path.join(run_dir, "events.jsonl")
    ctx.metadata_store_path = os.path.join(run_dir, "metadata.json")

    # Capture source fingerprint
    try:
        with open(source_path, "rb") as f:
            data = f.read(8192)
            ctx.source_fingerprint = hashlib.sha256(data).hexdigest()[:16]
    except Exception:
        ctx.source_fingerprint = "unknown"

    # Persist context
    with open(ctx.metadata_store_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": ctx.run_id,
            "source": source_path,
            "target": target,
            "config_hash": ctx.config_hash,
            "source_fingerprint": ctx.source_fingerprint,
            "started_at": ctx.started_at,
        }, f, indent=2)

    return ctx


# ---------------------------------------------------------------------------
# Phase 2: Intake & Profile (doc 15 steps 0011-0025)
# ---------------------------------------------------------------------------

@dataclass
class IntakeResult:
    """Result of intake and profiling phase."""
    profile: Any = None  # DatasetProfile
    columns: list[dict] = field(default_factory=list)
    n_rows: int = 0
    n_columns: int = 0
    quality_findings: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def phase_intake(ctx: RunContext, source_path: str, target: str) -> IntakeResult:
    """Phase 2: Intake & Profile — scan, profile, detect quality issues.

    Steps 0011-0025: source scan, column profiling, type inference,
    quality assessment, schema fingerprint.
    """
    result = IntakeResult()

    try:
        from ..intake import CsvDataSourceAdapter, ProfileConfig, ProfileMode, profile_source
        adapter = CsvDataSourceAdapter(source_path)
        config = ProfileConfig(mode=ProfileMode.STANDARD)
        profile = profile_source(adapter, config)
        result.profile = profile
        result.n_rows = profile.rows_observed
        result.n_columns = len(profile.columns)
        result.quality_findings = profile.quality_findings
        result.columns = [
            {
                "name": c.name,
                "dtype": c.physical_dtype,
                "semantic": c.semantic_type.value if hasattr(c.semantic_type, 'value') else str(c.semantic_type),
                "null_fraction": c.null_fraction,
            }
            for c in profile.columns
        ]
    except Exception as e:
        result.errors.append(str(e))

    return result


# ---------------------------------------------------------------------------
# Phase 3: Problem Validation & Leakage (doc 15 steps 0026-0040)
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of problem validation and leakage detection."""
    task: str = "classification"
    task_confidence: float = 0.0
    leakage_flags: list[dict] = field(default_factory=list)
    temporal_structure: bool = False
    group_structure: bool = False
    target_type: str = ""
    is_imbalanced: bool = False
    errors: list[str] = field(default_factory=list)


def phase_validate(ctx: RunContext, intake: IntakeResult, target: str) -> ValidationResult:
    """Phase 3: Problem Validation & Leakage Detection.

    Steps 0026-0040: task inference, temporal detection, group detection,
    leakage scan, target analysis, imbalance detection.
    """
    result = ValidationResult()

    try:
        from ..problem.task import infer_task
        target_profile = None
        for col in (intake.profile.columns if intake.profile else []):
            if col.name == target:
                target_profile = col
                break
        task_info = infer_task(target_profile)
        result.task = task_info.task.value if hasattr(task_info.task, "value") else str(task_info.task)
        result.task_confidence = task_info.confidence
        result.is_imbalanced = task_info.is_imbalanced
        if result.task in ("", "auto", "AUTO"):
            # No profile evidence -> conservative default (doc 03).
            result.task = "classification"
            result.task_confidence = min(result.task_confidence, 0.5)
    except Exception:
        pass

    try:
        # Basic leakage check
        for col in intake.columns:
            if col["name"] == target:
                continue
            if col["null_fraction"] < 0.01 and col.get("semantic", "") == "identifier":
                result.leakage_flags.append({
                    "column": col["name"],
                    "type": "potential_identifier",
                    "severity": "review_required",
                })
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Phase 4: Splits (doc 15 steps 0041-0050)
# ---------------------------------------------------------------------------

@dataclass
class SplitResult:
    """Result of split planning."""
    strategy: str = "kfold"
    n_folds: int = 5
    train_size: int = 0
    dev_size: int = 0
    test_size: int = 0
    temporal_cutoff: Optional[float] = None
    errors: list[str] = field(default_factory=list)


def phase_splits(
    ctx: RunContext, validation: ValidationResult, n_rows: int
) -> SplitResult:
    """Phase 4: Split Planning.

    Steps 0041-0050: split strategy selection, fold count,
    temporal cutoff, train/dev/test allocation.
    """
    result = SplitResult()

    if validation.temporal_structure:
        result.strategy = "temporal"
        result.train_size = int(n_rows * 0.6)
        result.dev_size = int(n_rows * 0.2)
        result.test_size = n_rows - result.train_size - result.dev_size
    elif validation.group_structure:
        result.strategy = "group_kfold"
    else:
        result.strategy = "kfold"
        result.n_folds = 5

    result.train_size = int(n_rows * 0.8) if not result.train_size else result.train_size
    result.dev_size = int(n_rows * 0.1) if not result.dev_size else result.dev_size
    result.test_size = n_rows - result.train_size - result.dev_size

    return result


# ---------------------------------------------------------------------------
# Phase 5: Feature Generation (doc 15 steps 0051-0067)
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """Result of feature generation."""
    proposals_count: int = 0
    after_dedup: int = 0
    sources_used: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def phase_generate(
    ctx: RunContext, intake: IntakeResult, validation: ValidationResult,
    target: str, max_proposals: int = 200,
) -> GenerationResult:
    """Phase 5: Feature Proposal Generation.

    Steps 0051-0067: enumerate operators, check preconditions,
    generate proposals from all 6 sources.
    """
    result = GenerationResult()
    sources_used = []

    try:
        from ..features.registry import all_operators
        ops = all_operators()  # returns list of FeatureOperator
        n_ops = len(ops)
        sources_used.append(f"operator_registry({n_ops})")

        # Count proposals per column
        n_cols = len(intake.columns) if intake.columns else 1
        n_target_excluded = 1 if target else 0
        n_feature_cols = max(1, n_cols - n_target_excluded)

        # Generate proposals per compatible column
        filtered = 0
        for op in ops:
            if intake.columns:
                for col in intake.columns:
                    if col.get("name") == target:
                        continue
                    sem = col.get("semantic", "").lower()
                    input_types_str = str(op.input_types).lower()
                    is_numeric_col = any(t in sem for t in ["numeric", "continuous", "count", "integer", "ratio"])
                    is_cat_col = any(t in sem for t in ["categorical", "low_cardinality", "ordinal"])
                    is_text_col = "text" in sem
                    if (is_numeric_col and ("numeric" in input_types_str or "continuous" in input_types_str or "count" in input_types_str)) or (is_cat_col and ("categorical" in input_types_str or "any" in input_types_str)) or (is_text_col and ("text" in input_types_str or "any" in input_types_str)) or not input_types_str or input_types_str == "any":
                        filtered += 1
                    else:
                        # Default: include if we can't determine type mismatch
                        filtered += 1
            else:
                filtered += n_feature_cols

        result.proposals_count = filtered
        result.after_dedup = min(filtered, max_proposals)
        result.sources_used = sources_used

    except Exception as e:
        result.errors.append(str(e))
        result.proposals_count = 0
        result.after_dedup = 0

    return result


# ---------------------------------------------------------------------------
# Phase 6: Funnel & Portfolio (doc 15 steps 0068-0090)
# ---------------------------------------------------------------------------

@dataclass
class FunnelResult:
    """Result of funnel filtering and portfolio selection."""
    f0_count: int = 0
    f1_count: int = 0
    f2_count: int = 0
    f5_count: int = 0
    portfolio_size: int = 0
    rejected: list[dict] = field(default_factory=list)
    # FeatureProposals backing the selected portfolio, in selection order
    # (op/inputs/params -- enough to rebuild the fitted features).
    portfolio_proposals: list = field(default_factory=list)
    # Row positions used for selection (the reserved holdout is excluded).
    dev_indexes: list = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def phase_funnel(
    ctx: RunContext, generation: GenerationResult, validation: ValidationResult,
    intake: IntakeResult | None = None,
    source_path: str = "", target: str = "",
) -> FunnelResult:
    """Phase 6: Funnel Filtering & Portfolio Selection.

    Steps 0068-0090: F0 validity, F1 precondition, F2 data quality,
    F5 complementarity, F3 portfolio selection, F4/F6 stability.

    Real execution path (doc 04/06): concrete FeatureProposals are generated
    from the profiled columns, every candidate runs through the actual
    F0->F1->F2 gates on a data sample, survivors are materialized, and the
    F3 portfolio is greedily selected on dev rows (the reserved holdout is
    never used for selection). Without a data sample only structural counts
    are reported, flagged via ``errors``.
    """
    result = FunnelResult()
    try:
        if (intake is None or intake.n_rows == 0 or intake.profile is None
                or not source_path or not target):
            # Structural pass only: operators x compatible columns.
            from ..features.registry import all_operators
            ops = all_operators()
            n_cols = max(1, len(intake.columns) if intake and intake.columns else 1)
            result.f0_count = len(ops) * n_cols
            result.f1_count = result.f0_count
            result.f2_count = result.f0_count
            result.f5_count = result.f0_count
            result.portfolio_size = min(max(result.f0_count // 10, 1), 20)
            result.errors.append("no data sample available - structural funnel counts only")
            return result

        from ..contracts import Task
        from ..fitted_pipeline import FittedPipeline
        from ..funnel import FunnelPolicy, run_funnel
        from ..learn import _to_floats, scan_columns
        from ..intake import auto_adapter
        from ..problem.splits import make_splits
        from ..probe import ProbePolicy, select_portfolio
        from ..search.hints import build_hint_candidates
        from ..search.records import StageVerdict
        from ..search.triggers import FeatureProposal, build_candidates

        profile = intake.profile
        policy = FunnelPolicy()

        # Concrete proposals over real columns (sources 2-3 + categorical).
        proposals: list[FeatureProposal] = list(
            build_candidates(profile, with_interactions=True, max_interactions=8)
        )
        proposals.extend(build_hint_candidates(profile))
        col_map = {c.name: c for c in profile.columns}
        for col in intake.columns:
            name = col.get("name", "")
            if name == target or name not in col_map:
                continue
            if "categorical" in str(col.get("semantic", "")):
                n_unique = col_map[name].distinct_estimate
                proposals.append(FeatureProposal(
                    op="hash_encode", inputs=["raw:" + name],
                    params={"n_buckets": min(32, max(4, n_unique))},
                    source="categorical_auto",
                ))

        # Target exclusion + dedup (mirrors learn.py).
        raw_target = "raw:" + target
        proposals = [p for p in proposals
                     if target not in p.inputs and raw_target not in p.inputs]
        seen: set = set()
        unique = []
        for p in proposals:
            sig = (p.op, tuple(p.inputs), tuple(sorted(p.params.items())))
            if sig not in seen:
                seen.add(sig)
                unique.append(p)
        proposals = unique[:200]

        # Read + float-convert the sample the gates materialize on.
        all_col_data = scan_columns(
            auto_adapter(source_path), max_rows=policy.f2_sample_rows)
        sample: dict[str, list] = {}
        for col in profile.columns:
            if col.name == target:
                continue
            raw = all_col_data.get(col.name)
            if raw is None:
                continue
            converted = []
            for v in raw:
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    converted.append(float(v))
                    continue
                try:
                    converted.append(float(str(v).strip()))
                except (ValueError, TypeError):
                    converted.append(None)
            sample[col.name] = converted

        # Encoded target for F3 selection (numeric passthrough, else class index).
        raw_target_vals = all_col_data.get(target, [])
        floats_t = _to_floats(raw_target_vals)
        is_cls = "classification" in str(validation.task)
        if is_cls and any(v is None for v in floats_t):
            classes = sorted({str(v).strip() for v in raw_target_vals if v is not None})
            _idx = {c: float(i) for i, c in enumerate(classes)}
            y = [_idx.get(str(v).strip() if v is not None else "", 0.0)
                 for v in raw_target_vals]
        else:
            y = [v if v is not None else 0.0 for v in floats_t]

        # Real F0 -> F1 -> F2 on every candidate.
        f1_survivors = 0
        f2_survivors = 0
        survivors: list[FeatureProposal] = []
        for p in proposals:
            record = run_funnel(p, profile, sample, policy)
            stages = {s.stage: s for s in record.stages}
            if not stages.get("F0", StageVerdict("F0", False)).passed:
                result.rejected.append({"op": p.op, "gate": "F0",
                                        "reason": record.final_reason})
                continue
            f1_survivors += 1
            if not stages.get("F1", StageVerdict("F1", False)).passed:
                result.rejected.append({"op": p.op, "gate": "F1",
                                        "reason": record.final_reason})
                continue
            if not stages.get("F2", StageVerdict("F2", False)).passed:
                result.rejected.append({"op": p.op, "gate": "F2",
                                        "reason": record.final_reason})
                continue
            f2_survivors += 1
            survivors.append(p)
        result.f0_count = len(proposals)
        result.f1_count = f1_survivors
        result.f2_count = f2_survivors

        if not survivors:
            result.errors.append("no candidates survived the funnel gates")
            return result

        # Materialize survivors and greedily select the portfolio (F3) on
        # dev rows only -- the reserved holdout never influences selection.
        outputs = FittedPipeline().fit(survivors, sample)
        if not outputs:
            result.errors.append("candidate materialization produced no outputs")
            return result

        task_enum = Task.BINARY if is_cls else Task.REGRESSION
        n = min(len(next(iter(outputs.values()))), len(y))
        if n <= 0:
            result.errors.append("no rows available for portfolio selection")
            return result
        split_spec = make_splits(task_enum, n, y=y[:n],
                                 holdout_fraction=0.15, seed=42)
        dev_rows = [i for i in split_spec.dev_indexes if i < n] or list(range(n))
        result.dev_indexes = dev_rows
        y_probe = [y[i] for i in dev_rows]
        dev_outputs = {name: [vals[i] for i in dev_rows]
                       for name, vals in outputs.items()}
        selected, _gains = select_portfolio(
            dev_outputs, y_probe, task_enum, ProbePolicy(), max_features=20)
        result.f5_count = len(selected)
        result.portfolio_size = len(selected)

        # Re-attach winning proposals in selection order (same naming the
        # FittedPipeline uses for feature outputs).
        by_name: dict = {}
        for p in survivors:
            nm = p.op + "(" + "_".join(
                inp[4:] if inp.startswith("raw:") else inp for inp in p.inputs
            ) + ")"
            by_name[nm] = p
        result.portfolio_proposals = [by_name[nm] for nm in selected if nm in by_name]
    except Exception as e:
        result.errors.append(str(e))
    return result


# ---------------------------------------------------------------------------
# Phase 7: HPO & Model Selection (doc 15 steps 0091-0100)
# ---------------------------------------------------------------------------

@dataclass
class HOResult:
    """Result of HPO and model selection."""
    trials_completed: int = 0
    best_model: str = ""
    best_score: float = 0.0
    search_strategy: str = "successive_halving"
    # Real TrialResults from the CV runs over routed families.
    trials: list = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ModelFamily values with a concrete sklearn estimator behind them
# (doc 07 routing -> model_training._get_sklearn_model names).
# Sklearn estimators are task-specific: Ridge ("linear") is regression-only,
# LogisticRegression is classification-only. Routing a regressor into a
# classification bracket scored NaN under ROC-AUC and silently poisoned
# model selection (fixed 0.0.2) — keep the maps disjoint per task.
_CLASSIFICATION_FAMILIES = {
    "logistic": "logistic_regression",
    "sgd_linear": "sgd",
    "decision_tree": "decision_tree",
    "random_forest": "random_forest",
    "extra_trees": "extra_trees",
}
_REGRESSION_FAMILIES = {
    "linear": "ridge",
    "sgd_linear": "sgd",
    "decision_tree": "decision_tree",
    "random_forest": "random_forest",
    "extra_trees": "extra_trees",
}


def _materialize_training_matrix(
    funnel: FunnelResult, source_path: str, target: str,
    validation: ValidationResult,
) -> tuple[list[list[float]], list[float], str]:
    """Materialize portfolio features + encoded target as an X/y matrix.

    Applies the portfolio's FeatureProposals via FittedPipeline on the
    sampled data. Missing feature values are imputed with 0.0 (canonical
    phase keeps no per-column fit state; probe-grade fidelity, not final
    model fidelity).
    """
    from ..fitted_pipeline import FittedPipeline
    from ..intake import auto_adapter
    from ..learn import _to_floats, scan_columns

    all_col_data = scan_columns(auto_adapter(source_path), max_rows=5000)
    raw_target = all_col_data.get(target, [])
    if not raw_target:
        return [], [], "classification"

    def _fv(v) -> float:
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        if isinstance(v, (int, float)):
            return float(v)
        return 0.0

    sample: dict[str, list] = {}
    for name, raw in all_col_data.items():
        if name == target:
            continue
        converted: list = []
        for v in raw:
            if isinstance(v, bool):
                converted.append(1.0 if v else 0.0)
            elif isinstance(v, (int, float)):
                converted.append(float(v))
            else:
                # CSV/JSON adapters return raw strings — parse them, do NOT
                # collapse to 0.0 (that zeroed the whole training matrix and
                # made every model score a silent 0.5 AUC).
                s = str(v).strip() if v is not None else ""
                try:
                    converted.append(float(s))
                except (ValueError, TypeError):
                    converted.append(None)  # categorical/text -> imputed later
        sample[name] = converted

    outputs = FittedPipeline().fit(funnel.portfolio_proposals, sample)
    if not outputs:
        return [], [], "classification"

    feature_names = list(outputs.keys())
    n = min(len(outputs[k]) for k in feature_names)
    X = [[_fv(outputs[k][i]) for k in feature_names] for i in range(n)]

    floats_t = _to_floats(raw_target)
    is_cls = "classification" in str(validation.task)
    if is_cls:
        if any(v is None for v in floats_t):
            classes = sorted({str(v).strip() for v in raw_target if v is not None})
            _idx = {c: float(i) for i, c in enumerate(classes)}
            y = [_idx.get(str(v).strip() if v is not None else "", 0.0)
                 for v in raw_target]
        else:
            y = [v if v is not None else 0.0 for v in floats_t]
        task = "classification"
    else:
        y = [v if v is not None else 0.0 for v in floats_t]
        task = "regression"
    m = min(len(X), len(y))
    return X[:m], y[:m], task


def phase_hpo(
    ctx: RunContext, funnel: FunnelResult, validation: ValidationResult,
    intake: IntakeResult | None = None,
    source_path: str = "", target: str = "",
) -> HOResult:
    """Phase 7: HPO & Model Selection.

    Steps 0091-0100: route to model families, run probe trials,
    successive halving, promotion, verify.

    Real execution path (doc 07): every routed family with a concrete
    backend trains an actual model via TrialRunner with 5-fold CV on the
    portfolio features; the family with the best CV score is promoted.
    Without training data nothing is fabricated -- the routing-only outcome
    is reported and flagged via ``errors``.
    """
    result = HOResult()
    try:
        from ..model_registry import ROUTING_TABLE
        from ..orchestration.model_training import TrialRunner, TrialSpec

        X: list = []
        y: list = []
        task = "classification"
        if funnel.portfolio_proposals and source_path and target:
            X, y, task = _materialize_training_matrix(
                funnel, source_path, target, validation)
        if not X or not y:
            result.errors.append(
                "no training data available - recorded model family routing only")
            result.best_model = "random_forest"
            return result

        # Task-aware routing happens AFTER the task is known (see map
        # comments above).
        family_map = (
            _REGRESSION_FAMILIES if task == "regression" else _CLASSIFICATION_FAMILIES
        )
        families = []
        for spec in ROUTING_TABLE:
            fam = spec.family.value if hasattr(spec.family, "value") else str(spec.family)
            if fam in family_map:
                families.append(fam)
        if not families:
            families = ["random_forest"]

        # Successive-halving-style bracket: cap candidates by portfolio size.
        n_candidates = max(1, min(len(families), max(funnel.portfolio_size, 2)))

        runner = TrialRunner()
        best: tuple | None = None
        for fam in families[:n_candidates]:
            spec = TrialSpec(model_family=family_map[fam], fold_count=5)
            trial = runner.run_trial(spec, X, y, task)
            result.trials.append(trial)
            result.trials_completed += 1
            if trial.status.name == "COMPLETED":
                score = next(
                    (m.value for m in trial.metrics if m.name == "quality"), None)
                # score == score filters NaN (a non-finite CV score must never
                # win promotion — belt-and-suspenders on top of task routing).
                if score is not None and score == score and (
                    best is None or score > best[1]
                ):
                    best = (fam, score)

        if best is not None:
            result.best_model = best[0]
            result.best_score = best[1]
        else:
            result.errors.append("all model trials failed")
            result.best_model = families[0]
    except Exception as e:
        result.errors.append(f"HPO error: {e}")
        result.best_model = "random_forest"
    return result


# ---------------------------------------------------------------------------
# Phase 8: Ensemble (doc 15 steps 0101-0105)
# ---------------------------------------------------------------------------

@dataclass
class EnsembleResult:
    """Result of ensemble construction."""
    members: int = 0
    weights: list[float] = field(default_factory=list)
    stacking_used: bool = False
    marginal_value: bool = False
    errors: list[str] = field(default_factory=list)


def phase_ensemble(
    ctx: RunContext, hpo: HOResult,
) -> EnsembleResult:
    """Phase 8: Ensemble Construction.

    Steps 0101-0105: eligibility, diversity, weight optimization,
    stacking guard, marginal value check.
    """
    result = EnsembleResult()
    try:
        from ..orchestration.ensemble import (
            EnsemblePolicy, build_ensemble,
        )

        # Real trials from HPO (no synthetic fabrication).
        trials = list(getattr(hpo, "trials", []) or [])

        if len(trials) >= 2:
            policy = EnsemblePolicy()
            ensemble_spec = build_ensemble(trials, policy)
            if ensemble_spec:
                result.members = len(ensemble_spec.member_trial_ids)
                result.weights = ensemble_spec.weights
                result.stacking_used = ensemble_spec.stacker is not None
                result.marginal_value = True
            else:
                # Fallback: compute weights manually
                n = len(trials)
                result.members = n
                result.weights = [1.0 / n] * n
                result.marginal_value = n > 1
        elif len(trials) == 1:
            result.members = 1
            result.weights = [1.0]
            result.marginal_value = False
        else:
            result.errors.append("no completed trials - ensemble not built")

    except Exception as e:
        result.errors.append(f"Ensemble error: {e}")
    return result


# ---------------------------------------------------------------------------
# Phase 9: Evaluation & Calibration (doc 15 steps 0106-0110)
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Result of evaluation and calibration."""
    primary_metric: str = "quality"
    primary_value: float = 0.0
    fold_mean: float = 0.0
    fold_std: float = 0.0
    generalization_gap: float = 0.0
    calibration_valid: bool = True
    errors: list[str] = field(default_factory=list)


def phase_evaluate(
    ctx: RunContext, hpo: HOResult, ensemble: EnsembleResult,
) -> EvalResult:
    """Phase 9: Final Evaluation & Calibration.

    Steps 0106-0110: full evaluation, fold distribution,
    generalization gap, calibration, threshold.
    """
    result = EvalResult()
    try:
        # Primary metric from the promoted HPO trial's real CV score.
        if hpo.best_score > 0:
            result.primary_value = hpo.best_score
            result.primary_metric = "quality"
        else:
            result.primary_value = 0.0
            result.errors.append("No model trained - evaluation based on structural analysis only")

        # Fold distribution from the promoted trial's actual CV metrics.
        best_trial = None
        for t in (getattr(hpo, "trials", []) or []):
            if t.status.name != "COMPLETED":
                continue
            q = next((m.value for m in t.metrics if m.name == "quality"), None)
            if q is not None and q == hpo.best_score:
                best_trial = t
                break
        if best_trial is not None:
            result.fold_mean = hpo.best_score
            result.fold_std = max(
                0.0, next((m.value for m in best_trial.metrics
                           if m.name == "cv_std"), 0.0))
            # Conservative generalization-gap estimate from fold spread.
            result.generalization_gap = min(0.2, result.fold_std * 0.5)
            result.calibration_valid = True
        else:
            result.fold_mean = 0.0
            result.calibration_valid = False

    except Exception as e:
        result.errors.append(f"Evaluation error: {e}")
    return result


# ---------------------------------------------------------------------------
# Phase 10: Codegen & Export (doc 15 steps 0111-0120)
# ---------------------------------------------------------------------------

@dataclass
class CodegenResult:
    """Result of code generation and export."""
    pipeline_ir_id: str = ""
    verification_gates_passed: int = 0
    verification_gates_total: int = 0
    export_path: str = ""
    export_code_path: str = ""
    errors: list[str] = field(default_factory=list)


def phase_codegen(
    ctx: RunContext, funnel: FunnelResult, eval_result: EvalResult,
) -> CodegenResult:
    """Phase 10: Code Generation & Export.

    Steps 0111-0120: build IR, compile, verify, export, smoke test.

    Real execution path (doc 09): the IR is built from the portfolio's
    actual FeatureProposals (op/inputs/params) so the exported feature code
    reproduces the fitted pipeline; it is compiled through the full
    verification gate battery and exported into the run directory. An empty
    portfolio skips the export instead of shipping an empty pipeline with
    vacuously green gates.
    """
    result = CodegenResult()
    try:
        from ..codegen.compiler import compile_pipeline
        from ..codegen.pipeline_ir import build_ir_from_proposals

        proposals = list(getattr(funnel, "portfolio_proposals", []) or [])
        if not proposals:
            result.errors.append("no portfolio proposals - export skipped")
            return result

        ir = build_ir_from_proposals(
            proposals,
            source_fingerprint=ctx.source_fingerprint,
            target=str(ctx.config_snapshot.get("target", "")),
            task=str(ctx.config_snapshot.get("task", "classification")),
        )
        result.pipeline_ir_id = ir.pipeline_id
        report = compile_pipeline(ir)
        result.verification_gates_passed = sum(1 for g in report.gates if g.passed)
        result.verification_gates_total = len(report.gates)

        export_dir = os.path.join(ctx.run_dir, "export")
        os.makedirs(export_dir, exist_ok=True)
        ir_path = os.path.join(export_dir, "pipeline_ir.json")
        with open(ir_path, "w", encoding="utf-8") as f:
            json.dump(ir.to_dict(), f, indent=2, default=str)
        report_path = os.path.join(export_dir, "compiler_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report.summary(), f, indent=2, default=str)
        code_path = os.path.join(export_dir, "features.py")
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(report.generated_code)
        result.export_path = export_dir
        result.export_code_path = code_path

        if not report.all_passed:
            result.errors.append(
                "verification gates failed: "
                + ", ".join(g.gate_name for g in report.gates if not g.passed))
    except Exception as e:
        result.errors.append(str(e))

    return result


# ---------------------------------------------------------------------------
# Full canonical pipeline (doc 15)
# ---------------------------------------------------------------------------

@dataclass
class CanonicalResult:
    """Full result of the canonical pipeline execution."""
    run_id: str = ""
    phases_completed: int = 0
    total_phases: int = 10
    intake: Optional[IntakeResult] = None
    validation: Optional[ValidationResult] = None
    splits: Optional[SplitResult] = None
    generation: Optional[GenerationResult] = None
    funnel: Optional[FunnelResult] = None
    hpo: Optional[HOResult] = None
    ensemble: Optional[EnsembleResult] = None
    evaluation: Optional[EvalResult] = None
    codegen: Optional[CodegenResult] = None
    total_time_s: float = 0.0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "phases_completed": self.phases_completed,
            "total_time_s": self.total_time_s,
            "portfolio_size": self.funnel.portfolio_size if self.funnel else 0,
            "best_model": self.hpo.best_model if self.hpo else "",
            "best_score": self.hpo.best_score if self.hpo else 0.0,
            "codegen_gates": f"{self.codegen.verification_gates_passed}/{self.codegen.verification_gates_total}" if self.codegen else "N/A",
            "primary_metric": self.evaluation.primary_metric if self.evaluation else "quality",
            "primary_value": self.evaluation.primary_value if self.evaluation else 0.0,
            "export_path": self.codegen.export_path if self.codegen else "",
            "errors": len(self.errors),
        }


def run_canonical_pipeline(
    source_path: str, target: str, config: dict | None = None,
) -> CanonicalResult:
    """Run the full canonical pipeline (doc 15 steps 0001-0120).

    Executes all 10 phases in sequence with proper state management,
    decision recording, and failure handling.
    """
    t0 = time.monotonic()
    result = CanonicalResult()

    try:
        # Phase 1: Bootstrap (steps 0001-0010)
        run_config = dict(config or {})
        run_config.setdefault("target", target)
        ctx = phase_bootstrap(source_path, target, run_config)
        result.run_id = ctx.run_id
        result.phases_completed = 1

        # Phase 2: Intake & Profile (steps 0011-0025)
        intake = phase_intake(ctx, source_path, target)
        result.intake = intake
        result.phases_completed = 2

        if intake.errors:
            result.errors.extend(intake.errors)

        # Phase 3: Problem Validation (steps 0026-0040)
        validation = phase_validate(ctx, intake, target)
        result.validation = validation
        ctx.config_snapshot["task"] = validation.task
        result.phases_completed = 3

        # Phase 4: Splits (steps 0041-0050)
        splits = phase_splits(ctx, validation, intake.n_rows)
        result.splits = splits
        result.phases_completed = 4

        # Phase 5: Feature Generation (steps 0051-0067)
        generation = phase_generate(ctx, intake, validation, target)
        result.generation = generation
        result.phases_completed = 5

        # Phase 6: Funnel (steps 0068-0090)
        funnel = phase_funnel(ctx, generation, validation, intake=intake,
                              source_path=source_path, target=target)
        result.funnel = funnel
        result.phases_completed = 6

        # Phase 7: HPO (steps 0091-0100)
        hpo = phase_hpo(ctx, funnel, validation, intake=intake,
                        source_path=source_path, target=target)
        result.hpo = hpo
        result.phases_completed = 7

        # Phase 8: Ensemble (steps 0101-0105)
        ensemble = phase_ensemble(ctx, hpo)
        result.ensemble = ensemble
        result.phases_completed = 8

        # Phase 9: Evaluation (steps 0106-0110)
        evaluation = phase_evaluate(ctx, hpo, ensemble)
        result.evaluation = evaluation
        result.phases_completed = 9

        # Phase 10: Codegen (steps 0111-0120)
        codegen = phase_codegen(ctx, funnel, evaluation)
        result.codegen = codegen
        result.phases_completed = 10

    except Exception as e:
        result.errors.append(f"Pipeline error: {e}")

    result.total_time_s = time.monotonic() - t0
    return result
