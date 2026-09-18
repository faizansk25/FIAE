"""Dataset profiler: bounded modes, quality checks, fingerprints (doc 02).

Profiling respects time/row/byte budgets and records whether statistics are
exact or approximate (FR-002). Acceptance failures abort or clarify: zero rows,
parser confidence too low, malformed rows beyond tolerance, source changed
mid-run.
"""

from __future__ import annotations

import enum
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from ..contracts import ColumnProfile, DatasetProfile, SemanticType
from ..errors import ErrorCode, FIAEError
from ..ids import content_hash, dataset_fingerprint
from .base import DataSourceAdapter, SamplePlan
from .stats import (
    CardinalitySketch,
    Missingness,
    NumericAccumulator,
    stable_hash64,
)
from .typing_engine import infer_physical_type, infer_semantic_type

import re

_RAW_SAMPLE_PER_COLUMN = 2000
_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$")


class ProfileMode(str, enum.Enum):
    FAST = "FAST"          # strict bounds; approximation allowed
    STANDARD = "STANDARD"  # larger representative scan
    EXACT = "EXACT"        # full scan where needed and affordable


@dataclass
class ProfileConfig:
    mode: ProfileMode = ProfileMode.FAST
    max_rows: int = 100_000
    max_bytes: int = 33_554_432
    time_budget_s: float = 30.0
    missing_tokens: frozenset = frozenset({"NA", "N/A", "null", "NULL", "None", "NaN"})
    cardinality_exact_cap: int = 4096
    malformed_tolerance: float = 0.05
    parser_confidence_floor: float = 0.5
    scheduler: str = "serial"  # "serial" | "threads" (parallel per-column accumulation)


@dataclass
class _ColumnAccumulator:
    missingness: Missingness
    cardinality: CardinalitySketch
    numeric: Optional[NumericAccumulator] = None
    raw_sample: list[str] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    uuid_count: int = 0
    length_hist: Counter = field(default_factory=Counter)
    # frequency keyed by stable hash: counts without retaining raw values
    _freq: dict[int, int] = field(default_factory=dict)
    max_freq: int = 0

    def add(self, value: Optional[str]) -> None:
        self.missingness.update(value)
        if value is None or value == "":
            return
        v = value.strip()
        if not v:
            return
        self.cardinality.add(v)
        if len(self.raw_sample) < _RAW_SAMPLE_PER_COLUMN:
            self.raw_sample.append(v)
        self.lengths.append(len(v))
        self.length_hist[len(v)] += 1
        if v.count("-") == 4:
            self.uuid_count += 1
        h = stable_hash64(v)
        c = self._freq.get(h, 0) + 1
        self._freq[h] = c
        if c > self.max_freq:
            self.max_freq = c

    def merge(self, other: "_ColumnAccumulator") -> None:
        """Merge a parallel partial accumulator into this one."""
        self.missingness.merge(other.missingness)
        self.cardinality.merge(other.cardinality)
        if other.numeric is not None:
            if self.numeric is None:
                self.numeric = other.numeric
            else:
                self.numeric.merge(other.numeric)
        if len(self.raw_sample) < _RAW_SAMPLE_PER_COLUMN:
            self.raw_sample.extend(
                other.raw_sample[: _RAW_SAMPLE_PER_COLUMN - len(self.raw_sample)]
            )
        self.lengths.extend(other.lengths)
        self.length_hist.update(other.length_hist)
        self.uuid_count += other.uuid_count
        for h, c in other._freq.items():
            new_c = self._freq.get(h, 0) + c
            self._freq[h] = new_c
            if new_c > self.max_freq:
                self.max_freq = new_c


def _length_stats(acc: _ColumnAccumulator) -> dict[str, float]:
    if not acc.lengths:
        return {"mean_length": 0.0, "length_std": 0.0, "token_mean": 0.0, "uuid_fraction": 0.0}
    n = len(acc.lengths)
    mean = sum(acc.lengths) / n
    var = sum((x - mean) ** 2 for x in acc.lengths) / n
    tokens = sum(x // 4 for x in acc.lengths) / n  # crude token proxy
    return {
        "mean_length": mean,
        "length_std": math.sqrt(var),
        "token_mean": tokens,
        "uuid_fraction": acc.uuid_count / n,
    }

def _quality_checks(
    name: str,
    acc: _ColumnAccumulator,
    semantic: SemanticType,
    distinct: int,
) -> list[dict[str, Any]]:
    """Data-quality checks from doc 02; findings, never silent coercion."""
    findings: list[dict[str, Any]] = []
    m = acc.missingness
    n = max(m.n_observed, 1)
    missing_frac = m.missing_total / n
    top_count = acc.max_freq

    if missing_frac >= 0.95:
        findings.append(
            {"type": "excessive_missingness", "column": name,
             "severity": "warning", "missing_fraction": round(missing_frac, 4)}
        )
    if distinct <= 1:
        findings.append({"type": "constant_column", "column": name, "severity": "warning"})
    elif distinct > 1 and n and top_count / n >= 0.995:
        findings.append(
            {"type": "near_constant_column", "column": name, "severity": "info",
             "dominant_fraction": round(top_count / n, 4)}
        )
    if m.sentinel / n >= 0.01:
        findings.append(
            {"type": "disguised_missing_values", "column": name,
             "severity": "warning", "sentinel_fraction": round(m.sentinel / n, 4),
             "note": "suspicious sentinels are warnings until confirmed"}
        )
    if acc.numeric is not None and acc.numeric.n_numeric:
        num = acc.numeric
        if num.zero_count / num.n_numeric >= 0.9:
            findings.append({"type": "excess_zeros", "column": name, "severity": "info"})
    if semantic is SemanticType.IDENTIFIER:
        findings.append({"type": "likely_identifier", "column": name, "severity": "info"})
    return findings


def _meta_features(
    rows_observed: int,
    n_columns: int,
    profiles: list[ColumnProfile],
) -> dict[str, Any]:
    """Dataset meta-features (doc 02; consumed by experience retrieval later)."""
    import math as _m

    missing_fractions = [p.null_fraction for p in profiles] or [0.0]
    cardinalities = [p.distinct_estimate for p in profiles] or [0]
    sem_fractions: Counter = Counter()
    for p in profiles:
        sem_fractions[p.semantic_type.value] += 1
    n = max(len(profiles), 1)
    return {
        "log_rows": round(_m.log(max(rows_observed, 1)), 4),
        "log_columns": round(_m.log(max(n_columns, 1)), 4),
        "row_to_column_ratio": round(rows_observed / max(n_columns, 1), 4),
        "missingness_mean": round(sum(missing_fractions) / n, 4),
        "missingness_max": round(max(missing_fractions), 4),
        "cardinality_mean": round(sum(cardinalities) / n, 2),
        "cardinality_max": max(cardinalities),
        "semantic_type_fractions": {
            k: round(v / n, 4) for k, v in sorted(sem_fractions.items())
        },
    }

def _feed(acc: "_ColumnAccumulator", values: Any, missing_tokens: frozenset) -> "_ColumnAccumulator":
    """Accumulate one column's values into an accumulator (parallel-safe)."""
    for v in values:
        s = None if v is None else str(v)
        stripped = s.strip() if s is not None else ""
        missing = s is None or stripped == "" or s in missing_tokens
        if not missing and _FLOAT_RE.match(stripped):
            if acc.numeric is None:
                acc.numeric = NumericAccumulator()
            acc.numeric.update(float(stripped))
        acc.add(s)
    return acc


def profile_source(
    adapter: DataSourceAdapter,
    config: ProfileConfig,
) -> DatasetProfile:
    """Profile a source under bounded budgets (doc 02, steps 0008-0020)."""
    start = time.monotonic()

    if config.mode is ProfileMode.EXACT:
        batches = adapter.scan()
    else:
        plan = SamplePlan(
            block_bytes=min(config.max_bytes, 262_144),
            max_total_bytes=config.max_bytes,
            random_seed=0,
        )
        batches = adapter.sample(plan)

    # Parser confidence gate runs after the zero-row abort (an empty source is
    # unambiguous about having no data; see acceptance failures, doc 02).
    report = getattr(adapter, "dialect_report", lambda: {})()

    rows_observed = 0
    accs: Optional[dict[str, _ColumnAccumulator]] = None
    columns: Optional[list[str]] = None
    parallel = config.scheduler == "threads"
    executor = None
    if parallel:
        import os
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4))

    try:
        for batch in batches:
            if time.monotonic() - start > config.time_budget_s:
                break  # bounded: keep what we have, statistics marked approximate
            if accs is None:
                columns = list(batch.columns.keys())
                accs = {
                    name: _ColumnAccumulator(
                        missingness=Missingness(config.missing_tokens),
                        cardinality=CardinalitySketch(config.cardinality_exact_cap),
                    )
                    for name in columns
                }
            if executor is not None:
                # Parallel per-column accumulation, merged serially (deterministic).
                futures = {
                    name: executor.submit(
                        _feed,
                        _ColumnAccumulator(
                            missingness=Missingness(config.missing_tokens),
                            cardinality=CardinalitySketch(config.cardinality_exact_cap),
                        ),
                        values,
                        config.missing_tokens,
                    )
                    for name, values in batch.columns.items()
                }
                for name, future in futures.items():
                    partial = future.result()
                    accs[name].merge(partial)
            else:
                for name, values in batch.columns.items():
                    _feed(accs[name], values, config.missing_tokens)
            rows_observed += batch.n_rows
            if rows_observed >= config.max_rows:
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=False)

    if accs is None or columns is None or rows_observed == 0:
        raise FIAEError(
            code=ErrorCode.DATA_FORMAT_ERROR,
            safe_message="Source contains zero readable rows.",
            component="profiler",
            evidence={"source": adapter.source_id()},
        )

    malformed = getattr(adapter, "malformed_fraction", 0.0)
    if malformed > config.malformed_tolerance:
        raise FIAEError(
            code=ErrorCode.DATA_FORMAT_ERROR,
            safe_message="Malformed row fraction exceeds tolerance.",
            component="profiler",
            evidence={"malformed_fraction": round(malformed, 4)},
        )

    # Parser confidence gate (step 0011, rule 10: lower confidence / abort).
    if report and report.get("dialect_confidence", 1.0) < config.parser_confidence_floor:
        raise FIAEError(
            code=ErrorCode.SCHEMA_AMBIGUITY,
            safe_message="Parser confidence below policy floor.",
            component="profiler",
            evidence={"dialect": report},
        )

    # Source stability (acceptance failure: source changes mid-run).
    verify = getattr(adapter, "verify_unchanged", None)
    if verify is not None:
        verify()

    exact_stats = config.mode is ProfileMode.EXACT
    column_profiles: list[ColumnProfile] = []
    content_signatures: dict[str, str] = {}
    for name in columns:
        acc = accs[name]
        phys = infer_physical_type(acc.raw_sample)
        distinct, exact_card = acc.cardinality.estimate()
        n_obs = max(acc.missingness.n_observed, 1)
        distinct_ratio = distinct / max(n_obs - acc.missingness.missing_total, 1)
        semantic, evidence = infer_semantic_type(
            name,
            phys["physical_dtype"],
            cardinality=(distinct, exact_card),
            distinct_ratio=distinct_ratio,
            missingness=acc.missingness,
            numeric=acc.numeric,
            length_stats=_length_stats(acc),
        )
        statistics: dict[str, Any] = {
            "exact_stats": exact_stats,
            "mode": config.mode.value,
            "missingness": acc.missingness.to_dict(),
            "physical_inference": phys,
            "semantic_evidence": evidence,
        }
        if acc.numeric is not None:
            statistics["numeric"] = acc.numeric.to_dict()
        content_signatures[name] = content_hash(
            {
                "length_hist": dict(acc.length_hist),
                "distinct": distinct,
                # stable hashes of sampled values (privacy-safe, no raw cells)
                "value_hashes": sorted(stable_hash64(v) for v in acc.raw_sample[:1000]),
            }
        )
        column_profiles.append(
            ColumnProfile(
                name=name,
                physical_dtype=phys["physical_dtype"],
                semantic_type=semantic,
                null_fraction=round(acc.missingness.missing_total / n_obs, 4),
                distinct_estimate=distinct,
                distinct_ratio=round(distinct_ratio, 4),
                statistics=statistics,
                warnings=[],
            )
        )

    # Duplicate columns by sampled content signature.
    seen: dict[str, str] = {}
    for p in column_profiles:
        sig = content_signatures[p.name]
        if sig in seen:
            p.warnings.append("duplicate_of:" + seen[sig])
        else:
            seen[sig] = p.name

    all_findings: list[dict[str, Any]] = []
    for p in column_profiles:
        for f in _quality_checks(p.name, accs[p.name], p.semantic_type, p.distinct_estimate):
            f["finding_id"] = f"{f['type']}|{p.name}"
            all_findings.append(f)
        if p.warnings:
            all_findings.append(
                {"type": "duplicate_columns", "column": p.name, "severity": "warning"}
            )

    schema_fingerprint = "schema_" + content_hash(
        [(p.name, p.physical_dtype, p.semantic_type.value) for p in column_profiles]
    )
    parser_config = {
        "format": "csv",
        "encoding": report.get("encoding"),
        "delimiter": report.get("delimiter"),
        "has_header": report.get("has_header"),
    }
    fp = dataset_fingerprint(
        adapter.fingerprint_material(), schema_fingerprint, parser_config
    )

    return DatasetProfile(
        dataset_fingerprint=fp,
        rows_observed=rows_observed,
        columns=column_profiles,
        rows_estimated=adapter.estimate_rows(),
        meta_features=_meta_features(rows_observed, len(columns), column_profiles),
        source_cost={
            "elapsed_s": round(time.monotonic() - start, 4),
            "estimated_bytes": adapter.estimate_bytes(),
            "mode": config.mode.value,
            "exact_stats": exact_stats,
        },
        quality_findings=all_findings,
    )


