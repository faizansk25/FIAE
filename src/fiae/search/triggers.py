"""Profile/statistical feature proposal triggers (doc 04, proposal sources 2).

The operator registry matches semantic types; this module turns *observed
profile statistics* into concrete candidate operations ("profile triggers"):
positive heavy skew -> log1p, mixed-sign heavy tail -> signed log, missingness ->
missing indicator, excess zeros -> zero indicator, low-cardinality categorical ->
(none; passed through as-is), date -> date parts/cyclical, high cardinality ->
frequency/cross-fit (deferred to later milestones/plugins).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..contracts import ColumnProfile, DatasetProfile, SemanticType


@dataclass(frozen=True)
class FeatureProposal:
    """A single candidate feature as (operation, inputs, params, provenance)."""

    op: str
    inputs: list[str]  # feature ids (raw:<name> leaves or derived ids)
    params: dict[str, Any] = field(default_factory=dict)
    source: str = "profile"          # proposal source label
    trigger_reason: str = ""
    depth: int = 1


def _raw_id(name: str) -> str:
    return f"raw:{name}"


def _num(col: ColumnProfile) -> Optional[dict[str, Any]]:
    if not col.statistics:
        return None
    return col.statistics.get("numeric")


def propose_for_column(col: ColumnProfile) -> list[FeatureProposal]:
    """Emit candidate transforms for a single profiled column (source 2 triggers)."""
    out: list[FeatureProposal] = []
    rid = _raw_id(col.name)

    # -- raw passthrough --------------------------------------------------------
    # The raw column itself competes for the portfolio. Without this,
    # transformed-only portfolios can silently lose the strongest signal
    # (e.g. churn AUC 0.69 on raw tenure vs 0.5 on transforms alone).
    if col.null_fraction < 0.5:
        out.append(
            FeatureProposal(
                op="identity", inputs=[rid], source="profile:raw",
                trigger_reason="raw column as baseline candidate",
            )
        )

    # -- missingness indicator (any column with observed nulls) -----------------
    if col.null_fraction > 0.0:
        out.append(
            FeatureProposal(
                op="missing_indicator", inputs=[rid],
                source="profile:missingness",
                trigger_reason=f"null_fraction={col.null_fraction:.4f}>0",
            )
        )

    # -- numeric distribution triggers ---------------------------------------
    num = _num(col)
    n_numeric = int((num or {}).get("count") or 0)
    if num is not None and n_numeric > 0:
        std = num.get("stddev")
        mn = num.get("min")
        mx = num.get("max")
        mean = num.get("mean")

        # constant/near-constant -> skip numeric transforms (no useful variation)
        if std is None or std == 0.0 or std < 1e-12:
            return out

        zero_count = int(num.get("zero_count") or 0)
        if zero_count > 0:
            out.append(
                FeatureProposal(
                    op="zero_indicator", inputs=[rid],
                    source="profile:excess_zeros",
                    trigger_reason=f"zero_count={zero_count}",
                )
            )

        if mn is not None and mn >= 0.0:
            # non-negative domain -> positive-skew compression candidates
            mean_gt_zero = mean is not None and mean > 0.0
            if mean_gt_zero or (mx or 0.0) > 1.0:
                out.append(
                    FeatureProposal(
                        op="log1p", inputs=[rid], source="profile:skew",
                        trigger_reason="non-negative domain; compress right skew",
                    )
                )
            out.append(
                FeatureProposal(
                    op="sqrt", inputs=[rid], source="profile:skew",
                    trigger_reason="non-negative domain",
                )
            )
        elif mx is not None and mx < 0.0:
            # all-negative -> make non-negative/central
            out.append(
                FeatureProposal(
                    op="square", inputs=[rid], source="profile:skew",
                    trigger_reason="all-negative; expose quadratic signal",
                )
            )
            out.append(
                FeatureProposal(
                    op="abs", inputs=[rid], source="profile:skew",
                    trigger_reason="all-negative; flip sign",
                )
            )
        else:
            # mixed sign -> signed/log/cube tails
            out.append(
                FeatureProposal(
                    op="signed_log1p", inputs=[rid], source="profile:signed_skew",
                    trigger_reason="mixed-sign heavy tail",
                )
            )
            out.append(
                FeatureProposal(
                    op="cbrt", inputs=[rid], source="profile:signed_skew",
                    trigger_reason="mixed-sign tail compression",
                )
            )

    # -- datetime part triggers ----------------------------------------------
    if col.semantic_type == SemanticType.DATETIME:
        for op in ("year", "month", "quarter", "day_of_week", "is_weekend", "hour"):
            out.append(
                FeatureProposal(
                    op=op, inputs=[rid], source="profile:datetime",
                    trigger_reason="datetime -> calendar part",
                )
            )

    return out
def _numeric_columns(profile: DatasetProfile) -> list[ColumnProfile]:
    return [
        c for c in profile.columns
        if c.semantic_type in (
            SemanticType.CONTINUOUS_NUMERIC,
            SemanticType.COUNT,
            SemanticType.PERCENTAGE,
            SemanticType.CURRENCY,
        )
    ]


def propose_interactions(
    profile: DatasetProfile,
    max_pairs: int = 8,
) -> list[FeatureProposal]:
    """Propose depth-1 interactions between high-signal numeric columns.

    Only well-behaved (low null, non-constant) columns are candidates, capped
    at ``max_pairs`` to bound the search space before the cheap gates.
    """
    cols = [c for c in _numeric_columns(profile) if c.null_fraction < 0.5]
    scored = []
    for c in cols:
        num = _num(c)
        if not num:
            continue
        std = num.get("stddev")
        if std is None or std == 0.0:
            continue
        scored.append(((1.0 - c.null_fraction) * (1.0 + min(abs(std), 10.0)), c))
    scored.sort(key=lambda t: t[0], reverse=True)
    top = [c.name for _, c in scored[: int(max_pairs ** 0.5) + 2]]

    proposals: list[FeatureProposal] = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            if len(proposals) >= max_pairs:
                return proposals
            a, b = _raw_id(top[i]), _raw_id(top[j])
            proposals.append(
                FeatureProposal(
                    op="safe_ratio", inputs=[a, b],
                    source="profile:interaction",
                    trigger_reason="numeric ratio interaction", depth=1,
                )
            )
            proposals.append(
                FeatureProposal(
                    op="difference", inputs=[a, b],
                    source="profile:interaction",
                    trigger_reason="numeric difference interaction", depth=1,
                )
            )
    return proposals


def build_candidates(
    profile: DatasetProfile,
    with_interactions: bool = True,
    max_interactions: int = 8,
) -> list[FeatureProposal]:
    """Folded proposal list over every column (source 2), plus interactions."""
    proposals: list[FeatureProposal] = []
    for col in profile.columns:
        proposals.extend(propose_for_column(col))
    if with_interactions:
        proposals.extend(propose_interactions(profile, max_pairs=max_interactions))
    return proposals
