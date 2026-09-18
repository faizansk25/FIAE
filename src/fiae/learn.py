"""End-to-end feature intelligence pipeline (doc 15, doc 04)."""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .contracts import SemanticType, Task
from .errors import ErrorCode, FIAEError
from .funnel import FunnelPolicy, run_funnel
from .intake import ProfileConfig, ProfileMode, profile_source
from .probe import ProbePolicy, select_portfolio
from .problem.splits import make_splits
from .problem.task import infer_task, route_metrics
from .search.records import FeatureAcceptanceRecord, StageVerdict
from .search.triggers import FeatureProposal, build_candidates
from .search.hints import build_hint_candidates

if TYPE_CHECKING:
    from .experiment.tracking import TrackingConfig


@dataclass
class LearnConfig:
    max_proposals: int = 64
    max_portfolio_features: int = 8
    funnel_policy: FunnelPolicy = field(default_factory=FunnelPolicy)
    probe_policy: ProbePolicy = field(default_factory=ProbePolicy)
    holdout_fraction: float = 0.2
    seed: int = 0
    sample_rows: int = 5000
    max_rows: int = 100_000
    batch_rows: int = 2048
    scheduler: str = "serial"
    mode: ProfileMode = ProfileMode.FAST
    tracking: Optional[TrackingConfig] = None
    experience_db: Optional[str] = None  # opt-in: path to experience store


def _write_experience_case(report, cfg: "LearnConfig", success: bool) -> None:
    """Persist a learn run to the experience store (opt-in, never crashes)."""
    if not cfg.experience_db:
        return
    try:
        from .experience.store import ExperienceStore, StoreConfig
        from .experience.case import (
            CaseRecord, CaseAction, CaseResult, CaseCost, CaseDiagnosis)
        from .contracts import TrialStatus

        store = ExperienceStore(StoreConfig(db_path=cfg.experience_db))
        try:
            failure_tags = [] if success else ["NO_INCREMENTAL_GAIN"]
            case = CaseRecord(
                case_id="case_" + report.dataset_fingerprint,
                dataset_fingerprint=report.dataset_fingerprint,
                schema_fingerprint=report.dataset_fingerprint,
                action=CaseAction(
                    model_family="fiae-learn",
                    feature_portfolio=[m.operator for m in report.portfolio],
                ),
                result=CaseResult(
                    primary_metric_name="portfolio_size",
                    primary_metric_value=float(report.portfolio_size),
                ),
                cost=CaseCost(wall_time_s=report.total_time_s),
                diagnosis=CaseDiagnosis(
                    status=TrialStatus.COMPLETED,
                    failure_tags=failure_tags,
                ),
                seed=cfg.seed,
            )
            store.write_case(case)
        finally:
            store.close()
    except Exception:
        # Experience write-back must never break the pipeline (doc 10).
        pass


def scan_columns(adapter, max_rows=5000, projection=None):
    result = {}
    for batch in adapter.scan(projection=projection):
        for name, values in batch.columns.items():
            result.setdefault(name, []).extend(values)
        if all(len(v) >= max_rows for v in result.values()):
            break
    return {k: v[:max_rows] for k, v in result.items()}


def _to_floats(values):
    out = []
    for v in values:
        # Native numeric values (SQL/DataFrame adapters) pass through.
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(None if isinstance(v, float) and math.isnan(v) else float(v))
            continue
        if v is None:
            out.append(None)
            continue
        s = str(v).strip()
        if s == '' or s in ('None','none','NULL','null','NA','N/A','NaN'):
            out.append(None)
            continue
        try:
            f = float(s)
            out.append(None if math.isnan(f) else f)
        except ValueError:
            out.append(None)
    return out


def _encode_target(raw: list, floats: list, inference) -> list[float]:
    """Encode the target as floats the probe can use.

    - numeric targets: pass through (None -> 0.0)
    - string/categorical targets for classification tasks: deterministic
      class-index encoding (sorted class order), so the probe sees real
      signal instead of an all-zero vector.
    """
    has_numeric = any(v is not None for v in floats)
    if has_numeric:
        return [v if v is not None else 0.0 for v in floats]
    # Categorical target: encode distinct values as class indices.
    classes = sorted({str(v).strip() for v in raw if v is not None})
    if len(classes) < 2:
        return [0.0] * len(raw)
    index = {c: float(i) for i, c in enumerate(classes)}
    return [index.get(str(v).strip() if v is not None else "", 0.0)
            for v in raw]


@dataclass
class FunnelStageResult:
    proposal_op: str
    proposal_inputs: list
    proposal_source: str
    f0_passed: bool
    f0_reason: str
    f1_passed: bool
    f1_reason: str
    f2_passed: bool
    f2_reason: str
    final_passed: bool
    final_reason: str

    @classmethod
    def from_record(cls, record):
        stages = {s.stage: s for s in record.stages}
        f0 = stages.get("F0", StageVerdict("F0", False, "not reached"))
        f1 = stages.get("F1", StageVerdict("F1", False, "not reached"))
        f2 = stages.get("F2", StageVerdict("F2", False, "not reached"))
        return cls(
            proposal_op=record.operator,
            proposal_inputs=list(record.inputs),
            proposal_source=", ".join(record.source) if record.source else "",
            f0_passed=f0.passed, f0_reason=f0.reason or "",
            f1_passed=f1.passed, f1_reason=f1.reason or "",
            f2_passed=f2.passed, f2_reason=f2.reason or "",
            final_passed=record.passed, final_reason=record.final_reason,
        )


@dataclass
class PortfolioMember:
    feature_id: str
    operator: str
    inputs: list
    incremental_gain: float
    f4_passed: bool
    f4_retention: Optional[float]
    f6_passed: bool
    f6_cv: Optional[float]
    fold_stability: Optional[float]

    @classmethod
    def from_record(cls, record):
        stages = {s.stage: s for s in record.stages}
        f4 = stages.get("F4")
        f6 = stages.get("F6")
        f4_ret = None
        if f4 and f4.passed and "gain_retention" in f4.metrics:
            f4_ret = f4.metrics["gain_retention"]
        f6_cv = None
        if f6 and "gain_cv" in f6.metrics:
            f6_cv = f6.metrics["gain_cv"]
        return cls(
            feature_id=record.feature_id, operator=record.operator,
            inputs=list(record.inputs), incremental_gain=record.incremental_gain or 0.0,
            f4_passed=f4.passed if f4 else False, f4_retention=f4_ret,
            f6_passed=f6.passed if f6 else False, f6_cv=f6_cv,
            fold_stability=record.fold_stability,
        )


@dataclass
class LearnReport:
    source_id: str = ""
    dataset_fingerprint: str = ""
    columns_in_source: int = 0
    rows_in_source: int = 0
    target: str = ""
    task: str = ""
    task_confidence: float = 0.0
    positive_class: Optional[str] = None
    is_imbalanced: bool = False
    metrics: list = field(default_factory=list)
    proposals_generated: int = 0
    proposals_after_dedup: int = 0
    funnel_passed_f2: int = 0
    funnel_results: list = field(default_factory=list)
    portfolio_size: int = 0
    portfolio: list = field(default_factory=list)
    total_time_s: float = 0.0
    # FeatureProposals backing the selected portfolio, in selection order.
    # Carries op/inputs/params so consumers (PipelineIR export) can rebuild
    # exactly what was fitted instead of guessing from names.
    portfolio_proposals: list = field(default_factory=list)

    def to_dict(self):
        return {
            "source_id": self.source_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "columns": self.columns_in_source,
            "rows": self.rows_in_source,
            "target": self.target,
            "task": self.task,
            "task_confidence": self.task_confidence,
            "positive_class": self.positive_class,
            "is_imbalanced": self.is_imbalanced,
            "metrics": self.metrics,
            "proposals_generated": self.proposals_generated,
            "proposals_after_dedup": self.proposals_after_dedup,
            "funnel_passed_f2": self.funnel_passed_f2,
            "funnel_results": [dataclasses.asdict(r) for r in self.funnel_results],
            "portfolio_size": self.portfolio_size,
            "portfolio": [dataclasses.asdict(p) for p in self.portfolio],
            "total_time_s": self.total_time_s,
        }


def learn(source, target, *, config=None):
    cfg = config or LearnConfig()
    t0 = time.monotonic()
    report = LearnReport(target=target)

    # Optional experiment tracking + audit trail (Phase 4)
    tracker = None
    if cfg.tracking is not None:
        from .experiment.tracking import ExperimentTracker
        tracker = ExperimentTracker(cfg.tracking)
        tracker.start({
            "target": target,
            "mode": getattr(cfg.mode, "value", str(cfg.mode)),
            "max_rows": cfg.max_rows,
            "batch_rows": cfg.batch_rows,
            "scheduler": cfg.scheduler,
            "sample_rows": cfg.sample_rows,
            "seed": cfg.seed,
        })


    # Phase 1: Intake and profiling
    if isinstance(source, str):
        from .intake import auto_adapter
        adapter = auto_adapter(source, batch_rows=cfg.batch_rows)
    else:
        adapter = source
    report.source_id = adapter.source_id()
    profile_config = ProfileConfig(mode=cfg.mode, max_rows=cfg.max_rows, scheduler=cfg.scheduler)
    profile = profile_source(adapter, profile_config)
    report.dataset_fingerprint = profile.dataset_fingerprint
    report.columns_in_source = len(profile.columns)
    report.rows_in_source = profile.rows_observed

    # Phase 2: Task inference
    col_data = scan_columns(adapter, max_rows=cfg.sample_rows, projection=[target])
    target_raw = col_data.get(target, [])
    if not target_raw:
        raise FIAEError(code=ErrorCode.TARGET_MISSING,
                        safe_message="Target column '" + target + "' not found or empty.",
                        component="learn")
    target_profile = None
    for col in profile.columns:
        if col.name == target:
            target_profile = col
            break
    if target_profile is None:
        raise FIAEError(code=ErrorCode.TARGET_MISSING,
                        safe_message="Target column '" + target + "' not found in source.",
                        component="learn")

    inference = infer_task(target_profile, target_values=target_raw if target_raw else None)
    templates = route_metrics(inference.task, is_imbalanced=inference.is_imbalanced)
    report.task = inference.task.value
    report.task_confidence = inference.confidence
    report.is_imbalanced = inference.is_imbalanced
    report.metrics = [t.name for t in templates]
    if inference.positive_class is not None:
        report.positive_class = str(inference.positive_class)

    # Phase 3: Splits
    # Use the actual number of target values (may be limited by sample_rows)
    n_rows = len(target_raw)
    if n_rows == 0:
        raise FIAEError(code=ErrorCode.DATA_FORMAT_ERROR,
                        safe_message="Source contains zero readable rows.", component="learn")
    target_floats = _to_floats(target_raw)
    y_all = _encode_target(target_raw, target_floats, inference.task)
    split_spec = make_splits(inference.task, n_rows,
                             y=y_all if inference.task != Task.UNSUPERVISED else None,
                             holdout_fraction=cfg.holdout_fraction, seed=cfg.seed)

    # Phase 4: Scan columns and detect types
    all_col_data = scan_columns(adapter, max_rows=cfg.sample_rows)
    numeric_cols = {}
    categorical_cols = {}
    for name, raw_vals in all_col_data.items():
        if name == target:
            continue
        # Detect semantic type from profile
        col_profile = None
        for cp in profile.columns:
            if cp.name == name:
                col_profile = cp
                break
        is_cat = (col_profile is not None and
                  col_profile.semantic_type in (
                      SemanticType.LOW_CARDINALITY_CATEGORICAL,
                      SemanticType.HIGH_CARDINALITY_CATEGORICAL))
        if is_cat:
            categorical_cols[name] = raw_vals
        else:
            num = _to_floats(raw_vals)
            if any(v is not None for v in num):
                numeric_cols[name] = num

    # Phase 5: Generate proposals (sources 2-3 + categorical)
    proposals = []
    proposals.extend(build_candidates(profile, with_interactions=True, max_interactions=8))
    proposals.extend(build_hint_candidates(profile))

    # Generate categorical operator proposals (L0 only — no fit state needed)
    for cat_name in categorical_cols:
        raw_inp = "raw:" + cat_name
        n_unique = len({v for v in categorical_cols[cat_name] if v})
        # hash_encode works for all cardinalities (L0, no fit needed)
        bucket_count = min(32, max(4, n_unique))
        proposals.append(FeatureProposal(
            op="hash_encode", inputs=[raw_inp],
            params={"n_buckets": bucket_count},
            source="categorical_auto",
        ))
        # ordinal_true_scale if user provides order (skip for now — needs params)

    if len(proposals) > cfg.max_proposals:
        proposals = proposals[:cfg.max_proposals]
    # Exclude proposals that reference the target column (raw: prefix)
    raw_target = "raw:" + target
    proposals = [p for p in proposals if target not in p.inputs and raw_target not in p.inputs]
    report.proposals_generated = len(proposals)

    seen_sigs = set()
    unique = []
    for p in proposals:
        sig = (p.op, tuple(p.inputs), tuple(sorted(p.params.items())))
        if sig not in seen_sigs:
            seen_sigs.add(sig)
            unique.append(p)
    proposals = unique
    report.proposals_after_dedup = len(proposals)

    sample = {}
    for col in profile.columns:
        if col.name in all_col_data:
            raw = all_col_data[col.name][:cfg.sample_rows]
            # Convert to floats for transforms that expect numerics.
            # Native numerics (SQL/DataFrame adapters) pass through.
            converted = []
            for v in raw:
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    converted.append(float(v))
                    continue
                v_stripped = str(v).strip() if v is not None else ''
                try:
                    converted.append(float(v_stripped))
                except (ValueError, TypeError):
                    converted.append(None)
            sample[col.name] = converted

    # Phase 6: Funnel F0-F2
    funnel_passed = []
    for proposal in proposals:
        record = run_funnel(proposal, profile, sample, cfg.funnel_policy)
        report.funnel_results.append(FunnelStageResult.from_record(record))
        if record.passed:
            funnel_passed.append((proposal, record))
    report.funnel_passed_f2 = len(funnel_passed)

    # Phase 7: Materialize survivors using FittedPipeline
    from .fitted_pipeline import FittedPipeline
    fitted_pipe = FittedPipeline()
    surviving_proposals = [p for p, r in funnel_passed]
    # Use float-converted sample for materialization
    candidate_outputs = fitted_pipe.fit(surviving_proposals, sample)

    # Phase 8: Portfolio selection
    if not candidate_outputs:
        report.total_time_s = time.monotonic() - t0
        _write_experience_case(report, cfg, success=False)
        if tracker is not None:
            tracker.log_metrics({"rows": report.rows_in_source,
                                 "portfolio_size": 0,
                                 "total_time_s": report.total_time_s})
            tracker.finish()
        return report
    # Holdout discipline (doc 03): the final holdout reserved in Phase 3 is
    # never used for feature selection or stability evaluation. Restrict the
    # probe/F4/F6 evaluation rows to dev (non-holdout) positions.
    sample_n = len(next(iter(candidate_outputs.values())))
    dev_rows = [i for i in split_spec.dev_indexes if i < sample_n]
    if not dev_rows:
        # Degenerate guard: profiled sample smaller than the holdout.
        dev_rows = list(range(sample_n))
    y_probe = [y_all[i] for i in dev_rows]
    dev_outputs = {
        name: [vals[i] for i in dev_rows]
        for name, vals in candidate_outputs.items()
    }
    selected_names, gains = select_portfolio(
        dev_outputs, y_probe, inference.task,
        cfg.probe_policy, max_features=cfg.max_portfolio_features)
    report.portfolio_size = len(selected_names)
    # Re-attach the winning FeatureProposals (op/inputs/params) so downstream
    # consumers — e.g. PipelineIR export — can rebuild exactly what was fitted.
    _name_to_proposal = {}
    for p in surviving_proposals:
        _nm = p.op + "(" + "_".join(
            inp[4:] if inp.startswith("raw:") else inp for inp in p.inputs
        ) + ")"
        _name_to_proposal[_nm] = p
    report.portfolio_proposals = [
        _name_to_proposal[n] for n in selected_names if n in _name_to_proposal
    ]

    # Phase 9: F5 complementarity + F4/F6 stability
    from .evaluate import f4_progressive_eval, f6_final_stability
    from .funnel import f5_complementarity, FunnelPolicy
    f5_policy = FunnelPolicy()
    for feat_name in selected_names:
        if feat_name not in dev_outputs:
            continue
        idx = selected_names.index(feat_name)
        base_cols = [dev_outputs[n] for n in selected_names[:idx]]
        cand = dev_outputs[feat_name]
        y_eval = y_probe
        rec = FeatureAcceptanceRecord(
            feature_id="port_" + feat_name, operator=feat_name,
            inputs=[], source=["portfolio"])
        # F5: complementarity check
        from .search.triggers import FeatureProposal as _FP
        dummy_proposal = _FP(op=feat_name, inputs=[], source="portfolio")
        f5_verdict = f5_complementarity(
            dummy_proposal, cand, base_cols, f5_policy)
        rec.add_stage(f5_verdict)
        if not f5_verdict.passed:
            rec.final_reason = f5_verdict.reason or "F5 rejected"
            report.portfolio.append(PortfolioMember.from_record(rec))
            continue
        # Record the probe's measured gain (Phase 8) — F4/F6 append verdicts
        # but deliberately re-measure at other row budgets/seeds.
        rec.incremental_gain = gains.get(feat_name)
        f4_progressive_eval(rec, base_cols, cand, y_eval, inference.task)
        f6_final_stability(rec, base_cols, cand, y_eval, inference.task)
        report.portfolio.append(PortfolioMember.from_record(rec))

    report.total_time_s = time.monotonic() - t0
    _write_experience_case(report, cfg, success=bool(report.portfolio))
    if tracker is not None:
        metrics = {
            "rows": report.rows_in_source,
            "columns": report.columns_in_source,
            "proposals_generated": report.proposals_generated,
            "proposals_after_dedup": report.proposals_after_dedup,
            "funnel_passed_f2": report.funnel_passed_f2,
            "portfolio_size": report.portfolio_size,
            "task_confidence": report.task_confidence,
            "total_time_s": report.total_time_s,
        }
        for member in report.portfolio:
            for stage in getattr(member, "stages", []) or []:
                gain = getattr(stage, "metric_gain", None)
                if isinstance(gain, (int, float)):
                    metrics.setdefault("best_feature_gain", gain)
                    metrics["best_feature_gain"] = max(
                        metrics["best_feature_gain"], gain)
        tracker.log_metrics(metrics)
        tracker.finish()
    return report
