"""Semantic hints from names and metadata (doc 04, proposal source 3).

Name/metadata-based proposals are *hints only* -- the weakest proposal source.
They generate hypotheses; the funnel gates and later experiments decide. No
hint ever accepts a feature by itself.
"""

from __future__ import annotations

import re

from ..contracts import ColumnProfile, SemanticType
from ..search.triggers import FeatureProposal, _raw_id

# ordered name patterns: (compiled regex, [(op, trigger_reason), ...])
_PATTERNS: tuple[tuple[re.Pattern, tuple[tuple[str, str], ...]], ...] = (
    (
        re.compile(r"(^|_)(date|dt|timestamp|ts|time)(_|$)", re.IGNORECASE),
        (("year", "name hints datetime"), ("month", "name hints datetime"),
         ("day_of_week", "name hints datetime")),
    ),
    (
        re.compile(r"(^|_)(price|amount|cost|revenue|salary|income|fee)(_|$)", re.IGNORECASE),
        (("log1p", "name hints money amount; skew expected"),
         ("safe_ratio", "name hints money amount; ratios meaningful")),
    ),
    (
        re.compile(r"(^|_)(count|qty|quantity|num|number|n_|cnt)(_|$)", re.IGNORECASE),
        (("log1p", "name hints count; right-skew expected"),),
    ),
    (
        re.compile(r"(^|_)(rate|ratio|pct|percent|share|frac)(_|$)", re.IGNORECASE),
        (("sqrt", "name hints rate/ratio; variance stabilization"),),
    ),
    (
        re.compile(r"^(is|has|was|ever)_", re.IGNORECASE),
        (("identity", "name hints boolean flag"),),
    ),
)


def propose_hints(col: ColumnProfile) -> list[FeatureProposal]:
    """Emit name-hint candidates for one column (source 3).

    Guards: identifiers and sensitive strings are never transformed; hints on
    datetime-typed columns are skipped when the type engine already emits
    calendar parts via the profile triggers (source 2 owns those).
    """
    if col.semantic_type in (
        SemanticType.IDENTIFIER,
        SemanticType.SENSITIVE_STRING,
        SemanticType.MIXED_UNKNOWN,
    ):
        return []

    out: list[FeatureProposal] = []
    for pattern, ops in _PATTERNS:
        if pattern.search(col.name):
            for op, reason in ops:
                out.append(
                    FeatureProposal(
                        op=op, inputs=[_raw_id(col.name)],
                        source="hints:name", trigger_reason=reason,
                    )
                )
            break  # first matching pattern wins; keep hint volume bounded
    return out


def build_hint_candidates(profile) -> list[FeatureProposal]:
    """Folded hint proposals over every column (source 3)."""
    proposals: list[FeatureProposal] = []
    for col in profile.columns:
        proposals.extend(propose_hints(col))
    return proposals
