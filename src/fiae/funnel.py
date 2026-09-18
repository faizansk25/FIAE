"""Funnel gates F0-F2 (doc 04).

F0 preconditions: operator exists, inputs available, semantic compatibility,
depth policy.
F1 static resource gate: bounded CPU/RAM/output estimates before any data is
touched.
F2 cheap materialization: run the exact transform on a small development
sample and measure invalid rate, constantness, and duplicate ratio.

Every gate appends a StageVerdict to the candidate's FeatureAcceptanceRecord;
a candidate rejected at any gate is never evaluated further. Gates return
findings only -- they never mutate data or accept features (doc 04 hypothesis
principle: experiments decide, this layer only filters and records).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .contracts import DatasetProfile, SemanticType
from .features.registry import FeatureOperator, get_operator
from .ids import content_hash
from .search.records import FeatureAcceptanceRecord, StageVerdict
from .search.triggers import FeatureProposal

# Numeric-like semantic types acceptable wherever "continuous_numeric" is
# requested (counts, percentages, currencies are numeric in behavior).
_NUMERIC_TYPES = frozenset(
    {
        SemanticType.CONTINUOUS_NUMERIC,
        SemanticType.COUNT,
        SemanticType.ORDINAL,
        SemanticType.PERCENTAGE,
        SemanticType.CURRENCY,
    }
)


@dataclass
class FunnelPolicy:
    """Policy thresholds for the F0-F2+F5 gates (policy, not universal constants)."""

    max_depth: int = 1
    min_input_null_ok: float = 0.95       # above this, input is effectively gone
    max_estimated_cpu_seconds: float = 60.0
    max_estimated_peak_ram_mb: float = 512.0
    max_estimated_output_mb: float = 512.0
    f2_sample_rows: int = 1000
    f2_max_invalid_rate: float = 0.5      # explicit-missing fraction tolerated
    f2_min_variance: float = 1e-12        # constant outputs carry no signal
    # F5 complementarity gate
    f5_max_correlation: float = 0.95      # above this, candidate is redundant
    f5_min_mutual_info_ratio: float = 0.1 # candidate must add >= 10% new info


def _raw_name(feature_id: str) -> Optional[str]:
    return feature_id[4:] if feature_id.startswith("raw:") else None


def _compatible(sem: SemanticType, want: str) -> bool:
    if want == "continuous_numeric":
        return sem in _NUMERIC_TYPES
    if want == "datetime":
        return sem is SemanticType.DATETIME
    if want == "boolean":
        return sem is SemanticType.BOOLEAN
    if want == "categorical":
        return sem in (
            SemanticType.LOW_CARDINALITY_CATEGORICAL,
            SemanticType.HIGH_CARDINALITY_CATEGORICAL,
        )
    return True  # unknown want -> do not over-reject at F0


def _column(profile: DatasetProfile, name: str):
    for c in profile.columns:
        if c.name == name:
            return c
    return None


def f0_preconditions(
    proposal: FeatureProposal,
    profile: DatasetProfile,
    policy: FunnelPolicy,
) -> StageVerdict:
    """F0: operator existence, input availability, semantic fit, depth."""
    metrics: dict[str, Any] = {"op": proposal.op, "depth": proposal.depth}

    try:
        op = get_operator(proposal.op)
    except Exception:
        return StageVerdict("F0", False, f"unknown operator: {proposal.op}", metrics)
    metrics["operator"] = op.name

    if proposal.depth > policy.max_depth:
        return StageVerdict("F0", False, f"depth {proposal.depth} > policy {policy.max_depth}", metrics)

    wants = op.input_types
    if len(proposal.inputs) != len(wants):
        return StageVerdict(
            "F0", False,
            f"arity mismatch: {len(proposal.inputs)} inputs, operator wants {len(wants)}",
            metrics,
        )

    for fid, want in zip(proposal.inputs, wants):
        name = _raw_name(fid)
        col = _column(profile, name) if name else None
        if col is None:
            return StageVerdict("F0", False, f"input not available: {fid}", metrics)
        if col.null_fraction >= policy.min_input_null_ok:
            return StageVerdict(
                "F0", False, f"input {name} null_fraction={col.null_fraction}", metrics
            )
        if not _compatible(col.semantic_type, want):
            return StageVerdict(
                "F0", False,
                f"input {name} semantic {col.semantic_type.value} incompatible with {want}",
                metrics,
            )
    return StageVerdict("F0", True, "preconditions satisfied", metrics)


def _estimate_resources(op: FeatureOperator, n_rows: int) -> dict[str, float]:
    """Rough static cost shape -> resource estimates (F1, bounded and cheap)."""
    if op.cost_shape == "O(n)":
        factor = 1.0
    elif op.cost_shape == "O(n * window)":
        factor = 8.0
    elif op.cost_shape.startswith("O(n log n)"):
        factor = 4.0
    else:
        factor = 2.0
    row_bytes = 8.0 * len(op.input_types)
    return {
        "estimated_cpu_seconds": factor * n_rows / 5_000_000.0,
        "estimated_peak_ram_mb": factor * n_rows * row_bytes / (1024.0 * 1024.0),
        "estimated_output_mb": n_rows * 8.0 / (1024.0 * 1024.0),
    }


def f1_static_resources(
    proposal: FeatureProposal,
    op: FeatureOperator,
    n_rows: int,
    policy: FunnelPolicy,
) -> StageVerdict:
    """F1: static resource gate; rejects before any materialization."""
    est = _estimate_resources(op, n_rows)
    for key, limit in (
        ("estimated_cpu_seconds", policy.max_estimated_cpu_seconds),
        ("estimated_peak_ram_mb", policy.max_estimated_peak_ram_mb),
        ("estimated_output_mb", policy.max_estimated_output_mb),
    ):
        if est[key] > limit:
            return StageVerdict("F1", False, f"{key}={est[key]:.3f} > {limit}", est)
    return StageVerdict("F1", True, "within static resource budget", est)


def _variance(values: list[Optional[float]]) -> float:
    xs = [v for v in values if v is not None]
    if len(xs) < 2:
        return 0.0
    mean = sum(xs) / len(xs)
    return sum((x - mean) ** 2 for x in xs) / len(xs)


def f2_materialize(
    proposal: FeatureProposal,
    op: FeatureOperator,
    sample: dict[str, list],
    policy: FunnelPolicy,
) -> StageVerdict:
    """F2: materialize on the dev sample; measure invalid rate/constantness.

    ``sample`` maps raw column names to value lists of equal length. The
    output values are *not* stored here; the caller re-runs the exact transform
    on the full data if the candidate advances (lazy DAG, doc 04).
    """
    n = min((len(v) for v in sample.values()), default=0)
    if n == 0:
        return StageVerdict("F2", False, "empty development sample", {})

    columns = [sample.get(_raw_name(fid) or "", [])[:n] for fid in proposal.inputs]
    values = op.transform(*columns, **proposal.params)

    invalid = sum(1 for v in values if v is None)
    invalid_rate = invalid / n
    metrics: dict[str, Any] = {"sample_rows": n, "invalid_rate": round(invalid_rate, 4)}

    if invalid_rate > policy.f2_max_invalid_rate:
        return StageVerdict(
            "F2", False, f"invalid_rate={invalid_rate:.3f} > {policy.f2_max_invalid_rate}", metrics
        )

    numeric: list[Optional[float]] = []
    for v in values:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            numeric.append(float(v))
        else:
            numeric.append(None)
    if numeric and numeric.count(None) < n:
        var = _variance(numeric)
        metrics["variance"] = var
        if var < policy.f2_min_variance:
            return StageVerdict("F2", False, f"constant output (variance={var:.3e})", metrics)

    dup = n - len({repr(v) for v in values if v is not None})
    metrics["duplicate_rows"] = dup
    return StageVerdict("F2", True, "materialization healthy", metrics)


# ---------------------------------------------------------------------------
# F5: Portfolio interaction / complementarity gate (doc 04)
# ---------------------------------------------------------------------------
def _pearson(a: list, b: list) -> float:
    """Compute Pearson correlation between two float lists."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 10:
        return 0.0
    ax = [p[0] for p in pairs]
    by = [p[1] for p in pairs]
    n = len(ax)
    mean_a = sum(ax) / n
    mean_b = sum(by) / n
    var_a = sum((x - mean_a) ** 2 for x in ax) / n
    var_b = sum((x - mean_b) ** 2 for x in by) / n
    if var_a == 0 or var_b == 0:
        return 0.0
    cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(ax, by)) / n
    return cov / (var_a ** 0.5 * var_b ** 0.5)


def _to_floats_safe(values: list) -> list:
    """Coerce values to floats, treating None/bool/str as None."""
    out = []
    for v in values:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(float(v))
        elif isinstance(v, bool):
            out.append(1.0 if v else 0.0)
        else:
            out.append(None)
    return out


def f5_complementarity(
    proposal: FeatureProposal,
    candidate_values: list,
    portfolio_columns: list[list],
    policy: FunnelPolicy,
) -> StageVerdict:
    """F5: check the candidate is not redundant with existing portfolio.

    Uses Pearson correlation as a cheap proxy for redundancy.
    ``candidate_values`` is the materialized output of the candidate.
    ``portfolio_columns`` is a list of already-selected feature vectors.
    """
    n = len(candidate_values)
    metrics: dict[str, Any] = {"candidate_op": proposal.op}

    if not portfolio_columns:
        v = StageVerdict("F5", True, "empty portfolio — no redundancy check", metrics)
        return v

    cand_floats = _to_floats_safe(candidate_values)
    max_corr = 0.0
    corr_details = []

    for i, port_col in enumerate(portfolio_columns):
        port_floats = _to_floats_safe(port_col[:n])
        corr = _pearson(cand_floats, port_floats)
        abs_corr = abs(corr)
        corr_details.append({"portfolio_idx": i, "correlation": round(corr, 4)})
        if abs_corr > max_corr:
            max_corr = abs_corr

    metrics["max_correlation"] = round(max_corr, 4)
    metrics["correlations"] = corr_details

    if max_corr > policy.f5_max_correlation:
        v = StageVerdict(
            "F5", False,
            f"redundant: max correlation {max_corr:.3f} > {policy.f5_max_correlation}",
            metrics,
        )
        return v

    v = StageVerdict("F5", True, f"complementary (max corr={max_corr:.3f})", metrics)
    return v


def run_funnel(
    proposal: FeatureProposal,
    profile: DatasetProfile,
    sample: dict[str, list],
    policy: Optional[FunnelPolicy] = None,
) -> FeatureAcceptanceRecord:
    """Run one candidate through F0 -> F1 -> F2, recording every verdict."""
    policy = policy or FunnelPolicy()
    cid = content_hash(
        {"op": proposal.op, "inputs": proposal.inputs, "params": proposal.params}
    )
    record = FeatureAcceptanceRecord(
        feature_id=f"cand_{cid}",
        operator=proposal.op,
        inputs=list(proposal.inputs),
        params=dict(proposal.params),
        source=[proposal.source],
        trigger_reason=proposal.trigger_reason,
        lineage=[_raw_name(fid) or fid for fid in proposal.inputs],
    )

    v0 = f0_preconditions(proposal, profile, policy)
    record.add_stage(v0)
    if not v0.passed:
        record.final_reason = v0.reason or "F0 rejected"
        return record

    op = get_operator(proposal.op)
    n_rows = len(next(iter(sample.values()))) if sample else 0
    v1 = f1_static_resources(proposal, op, n_rows, policy)
    record.add_stage(v1)
    if not v1.passed:
        record.final_reason = v1.reason or "F1 rejected"
        return record

    v2 = f2_materialize(proposal, op, sample, policy)
    record.add_stage(v2)
    if not v2.passed:
        record.final_reason = v2.reason or "F2 rejected"
    return record
