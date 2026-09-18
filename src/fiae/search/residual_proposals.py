"""Proposal sources 5 and 6 (doc 04).

Source 5: OOF residual-based feature proposals — analyze out-of-fold
residuals from a baseline model to suggest interaction features where
baseline errors remain structured.

Source 6: Domain plugin proposals — user-defined domain-specific feature
templates that apply to detected data patterns.

Normative source: doc 04 section "Proposal sources".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..contracts import DatasetProfile, SemanticType
from .triggers import FeatureProposal


# ---------------------------------------------------------------------------
# Source 5: OOF Residual Proposals
# ---------------------------------------------------------------------------
def _pearson_simple(a: list[float], b: list[float]) -> float:
    """Simple Pearson correlation."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 10:
        return 0.0
    n = len(pairs)
    ax = [p[0] for p in pairs]
    by = [p[1] for p in pairs]
    mean_a = sum(ax) / n
    mean_b = sum(by) / n
    var_a = sum((x - mean_a) ** 2 for x in ax) / n
    var_b = sum((x - mean_b) ** 2 for x in by) / n
    if var_a == 0 or var_b == 0:
        return 0.0
    cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(ax, by)) / n
    return cov / (var_a ** 0.5 * var_b ** 0.5)


def propose_from_residuals(
    columns: dict[str, list],
    residuals: list[float],
    target: str = "",
    max_proposals: int = 10,
    min_correlation: float = 0.1,
) -> list[FeatureProposal]:
    """Source 5: generate proposals from OOF residuals (doc 04).

    Analyzes which source features correlate with the absolute residuals
    (structured errors) of a baseline model, suggesting interactions
    that could reduce those errors.
    """
    if not residuals or not columns:
        return []

    n = min(len(residuals), max(len(v) for v in columns.values()))
    abs_residuals = [abs(r) for r in residuals[:n]]

    candidates = []
    for name, values in columns.items():
        if name == target:
            continue
        # Convert to floats
        float_vals = []
        for v in values[:n]:
            try:
                float_vals.append(float(v) if v is not None else None)
            except (ValueError, TypeError):
                float_vals.append(None)

        valid = sum(1 for v in float_vals if v is not None)
        if valid < 20:
            continue

        corr = _pearson_simple(float_vals, abs_residuals)
        if abs(corr) >= min_correlation:
            # Propose log1p if residual correlation is strong and values are positive
            has_positive = any(v is not None and v > 0 for v in float_vals)
            if has_positive and abs(corr) > 0.2:
                candidates.append(FeatureProposal(
                    op="log1p", inputs=[f"raw:{name}"],
                    params={}, source="residual_analysis",
                    trigger_reason=f"residual correlation {corr:.3f}",
                ))
            # Propose sqrt for non-negative values
            has_nonneg = any(v is not None and v >= 0 for v in float_vals)
            if has_nonneg and abs(corr) > 0.15:
                candidates.append(FeatureProposal(
                    op="sqrt", inputs=[f"raw:{name}"],
                    params={}, source="residual_analysis",
                    trigger_reason=f"residual correlation {corr:.3f}",
                ))
            # Propose zero_indicator if many zeros correlate with high residuals
            zero_count = sum(1 for v in float_vals if v is not None and v == 0.0)
            if zero_count > 5 and zero_count < n * 0.5:
                candidates.append(FeatureProposal(
                    op="zero_indicator", inputs=[f"raw:{name}"],
                    params={}, source="residual_analysis",
                    trigger_reason="zero residual correlation",
                ))

    # Sort by number of proposals (columns with more residual signal first)
    candidates.sort(key=lambda p: p.trigger_reason, reverse=True)
    return candidates[:max_proposals]


# ---------------------------------------------------------------------------
# Source 6: Domain Plugin Proposals
# ---------------------------------------------------------------------------
@dataclass
class DomainPlugin:
    """A user-defined domain-specific feature template."""

    name: str
    description: str
    applies_to: tuple[str, ...]  # semantic types this applies to
    generator: Callable[[str, DatasetProfile], list[FeatureProposal]]
    enabled: bool = True


# Built-in domain plugins
_PLUGINS: list[DomainPlugin] = []


def register_plugin(plugin: DomainPlugin) -> None:
    """Register a domain plugin."""
    _PLUGINS.append(plugin)


def get_plugins() -> list[DomainPlugin]:
    """Get all registered domain plugins."""
    return [p for p in _PLUGINS if p.enabled]


def propose_from_plugins(
    profile: DatasetProfile,
    max_proposals: int = 10,
) -> list[FeatureProposal]:
    """Source 6: generate proposals from domain plugins (doc 04)."""
    proposals = []
    for plugin in get_plugins():
        try:
            plugin_proposals = plugin.generator("raw", profile)
            proposals.extend(plugin_proposals)
        except Exception:
            continue
    return proposals[:max_proposals]


# Register built-in plugins
def _financial_plugin(prefix: str, profile: DatasetProfile) -> list[FeatureProposal]:
    """Financial domain: log of monetary columns, price/quantity ratios."""
    proposals = []
    money_names = {"price", "amount", "cost", "revenue", "salary", "payment", "total", "value"}
    count_names = {"quantity", "qty", "count", "num", "number", "items"}
    money_cols = [c.name for c in profile.columns if c.name.lower() in money_names
                  and c.semantic_type in (SemanticType.CURRENCY, SemanticType.CONTINUOUS_NUMERIC)]
    count_cols = [c.name for c in profile.columns if c.name.lower() in count_names
                  and c.semantic_type in (SemanticType.COUNT, SemanticType.CONTINUOUS_NUMERIC)]

    for mc in money_cols:
        proposals.append(FeatureProposal(
            op="log1p", inputs=[f"raw:{mc}"], source="domain_financial",
            trigger_reason="financial domain: log monetary value",
        ))
    for mc in money_cols:
        for cc in count_cols:
            proposals.append(FeatureProposal(
                op="safe_ratio", inputs=[f"raw:{mc}", f"raw:{cc}"],
                source="domain_financial",
                trigger_reason="financial domain: price per unit",
            ))
    return proposals


register_plugin(DomainPlugin(
    name="financial", description="Financial domain features",
    applies_to=("currency", "count"), generator=_financial_plugin,
))


def _temporal_plugin(prefix: str, profile: DatasetProfile) -> list[FeatureProposal]:
    """Temporal domain: time-based features for datetime columns."""
    proposals = []
    for col in profile.columns:
        if col.semantic_type == SemanticType.DATETIME:
            for op in ["year", "month", "day_of_week", "hour"]:
                proposals.append(FeatureProposal(
                    op=op, inputs=[f"raw:{col.name}"], source="domain_temporal",
                    trigger_reason=f"temporal domain: {op}",
                ))
    return proposals


register_plugin(DomainPlugin(
    name="temporal", description="Temporal domain features",
    applies_to=("datetime",), generator=_temporal_plugin,
))
