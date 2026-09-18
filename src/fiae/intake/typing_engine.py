"""Physical and semantic type inference (doc 02, FR-003).

Physical dtype and semantic role are separate outputs. Each observed non-null
value eliminates incompatible candidates; ambiguous mixed columns remain
string/mixed rather than unsafe coercion. Name hints are never hard evidence
alone for identifiers.
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import Any, Optional

from ..contracts import SemanticType
from .stats import Missingness, NumericAccumulator

_PHYSICAL_CANDIDATES = ("boolean", "integer", "float", "date", "timestamp", "string")

_INT_RE = re.compile(r"^[+-]?\d+$")
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_BOOL_TOKENS = frozenset({"true", "false"})


def parse_scalar_candidates(value: str) -> set[str]:
    """Return the set of physical candidates a single raw value is compatible with."""
    v = value.strip()
    out: set[str] = set()
    if v.lower() in _BOOL_TOKENS:
        out.add("boolean")
    if _INT_RE.match(v):
        out.add("integer")
        out.add("float")
    else:
        try:
            float(v)
            out.add("float")
        except ValueError:
            pass
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            _dt.datetime.strptime(v, fmt)
            out.add("date")
            out.add("timestamp")
            break
        except ValueError:
            continue
    try:
        parsed = _dt.datetime.fromisoformat(v)
        if (parsed.hour, parsed.minute, parsed.second) != (0, 0, 0) or "T" in v or " " in v:
            out.add("timestamp")
    except ValueError:
        pass
    if not out:
        out.add("string")
    return out


def infer_physical_type(sampled_values: list[str]) -> dict[str, Any]:
    """Candidate elimination over observed non-null values (doc 02).

    Ambiguous mixtures remain string/mixed rather than unsafe coercion.
    Numeric and datetime chains (integer⊂float, date⊂timestamp) do not count
    as ambiguity when the more specific type is consistent everywhere.
    """
    candidates = set(_PHYSICAL_CANDIDATES)
    for value in sampled_values:
        if value is None:
            continue
        ok = parse_scalar_candidates(value)
        if ok == {"string"}:
            # Untypeable value eliminates every typed candidate.
            candidates &= {"string"}
        else:
            candidates &= ok
        if not candidates:
            break

    typed = candidates - {"string"}
    survivors = sorted(candidates)
    if not typed:
        # nothing typed anywhere (or mixed typed/untypeable handled above):
        # untypeable columns stay strings, mixed candidates keep low confidence.
        dtype = "string"
        confidence = 1.0 if candidates == {"string"} else 0.5
    elif candidates <= _NUMERIC_CHAIN:
        dtype = "integer" if "integer" in candidates else "float"
        confidence = 1.0
    elif candidates <= _DT_CHAIN:
        dtype = "date" if "date" in candidates else "timestamp"
        confidence = 1.0
    elif len(typed) == 1:
        dtype, confidence = next(iter(typed)), 1.0
    else:
        dtype = "string"  # contradictory typed candidates: unsafe to coerce
        confidence = 0.5
    return {"physical_dtype": dtype, "confidence": confidence, "survivors": survivors}


_NUMERIC_CHAIN = frozenset({"integer", "float"})
_DT_CHAIN = frozenset({"date", "timestamp"})

_ID_NAME_HINTS = ("_id", "id", "uuid", "guid", "key", "record")
_CURRENCY_HINTS = ("price", "cost", "amount", "revenue", "salary", "fee", "payment")
_PCT_HINTS = ("pct", "percent", "ratio", "rate", "share")


def infer_semantic_type(
    name: str,
    physical_dtype: str,
    *,
    cardinality: tuple[int, bool],
    distinct_ratio: float,
    missingness: Missingness,
    numeric: Optional[NumericAccumulator],
    length_stats: dict[str, float],
) -> tuple[SemanticType, dict[str, Any]]:
    """Semantic role inference with recorded evidence (doc 02, FR-003)."""
    distinct, exact = cardinality
    evidence: dict[str, Any] = {
        "distinct": distinct,
        "exact": exact,
        "distinct_ratio": round(distinct_ratio, 4),
    }

    if physical_dtype in ("date", "timestamp"):
        return SemanticType.DATETIME, evidence
    if physical_dtype == "boolean":
        return SemanticType.BOOLEAN, evidence

    lname = name.lower()
    if physical_dtype in ("integer", "float"):
        if numeric is not None and distinct > 1:
            non_neg = numeric.minimum is not None and numeric.minimum >= 0
            if any(h in lname for h in _PCT_HINTS) and non_neg and (
                numeric.maximum is not None and numeric.maximum <= 100
            ):
                evidence["hint"] = "percentage_name_range"
                return SemanticType.PERCENTAGE, evidence
            if any(h in lname for h in _CURRENCY_HINTS):
                evidence["hint"] = "currency_name"
                return SemanticType.CURRENCY, evidence
            if non_neg and physical_dtype == "integer":
                return SemanticType.COUNT, evidence
        return SemanticType.CONTINUOUS_NUMERIC, evidence

    # string-valued columns -------------------------------------------------
    # Identifier evidence: distinct ratio near one AND (pattern or name hint).
    if distinct_ratio > 0.95 and distinct > 16 and length_stats["mean_length"] <= 64:
        uuid_like = length_stats.get("uuid_fraction", 0.0) > 0.8
        fixed_token = (
            exact
            and length_stats.get("length_std", 1.0) < 0.01
            and length_stats["mean_length"] >= 8
        )
        name_hint = any(h == lname or lname.endswith(h) for h in _ID_NAME_HINTS)
        evidence.update(
            {"uuid_like": uuid_like, "fixed_token": fixed_token, "name_hint": name_hint}
        )
        if uuid_like or fixed_token:
            return SemanticType.IDENTIFIER, evidence
        if name_hint:
            # Name hints are never hard evidence alone (doc 02) -> medium conf.
            evidence["confidence"] = "medium"
            return SemanticType.IDENTIFIER, evidence

    # Text vs categorical evidence (doc 02).
    if length_stats["mean_length"] > 40 or length_stats.get("token_mean", 0) > 6:
        evidence["reason"] = "length_or_tokens"
        return SemanticType.FREE_TEXT, evidence
    if distinct <= 50 or distinct_ratio <= 0.05:
        return SemanticType.LOW_CARDINALITY_CATEGORICAL, evidence
    if distinct_ratio >= 0.9 or distinct > 10_000:
        evidence["reason"] = "high_distinct_ratio"
        return SemanticType.HIGH_CARDINALITY_CATEGORICAL, evidence
    return SemanticType.LOW_CARDINALITY_CATEGORICAL, evidence
