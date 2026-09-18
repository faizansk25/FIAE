"""Benchmark suite (doc 12).

Standardized benchmarks for measuring:
- Operator throughput (rows/sec per operator family)
- Pipeline latency (end-to-end timing)
- Memory usage patterns
- Regression detection (compare against baseline)

Normative source: doc 12 section "Benchmarks".
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..features.registry import all_operators


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""

    name: str
    duration_ms: float
    ops_per_sec: float = 0.0
    rows: int = 0
    details: dict[str, Any] = field(default_factory=dict)
    passed: bool = True


@dataclass
class BenchmarkSuite:
    """Full benchmark suite results."""

    results: list[BenchmarkResult] = field(default_factory=list)
    total_duration_ms: float = 0.0
    summary: dict[str, Any] = field(default_factory=dict)


def benchmark_operator_throughput(
    family: Optional[str] = None, n_rows: int = 10000, n_runs: int = 3,
) -> list[BenchmarkResult]:
    """Benchmark operator throughput for a family or all operators."""
    rng = random.Random(42)
    data = [rng.gauss(0, 1) for _ in range(n_rows)]
    ops = all_operators()

    if family:
        ops = [o for o in ops if o.family == family]

    results = []
    for op in ops:
        if op.arity != "unary":
            continue
        if op.fit_scope.value != "none":
            continue

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            try:
                op.transform(data)
            except Exception:
                continue
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        if times:
            avg_ms = sum(times) / len(times)
            ops_per_sec = n_rows / (avg_ms / 1000) if avg_ms > 0 else 0
            results.append(BenchmarkResult(
                name=f"throughput_{op.name}",
                duration_ms=avg_ms,
                ops_per_sec=ops_per_sec,
                rows=n_rows,
                details={"family": op.family, "runs": len(times)},
            ))

    return results


def benchmark_pipeline_latency(n_rows: int = 1000, n_runs: int = 3) -> BenchmarkResult:
    """Benchmark the feature proposal + funnel pipeline."""
    from ..intake import CsvDataSourceAdapter, ProfileConfig, ProfileMode, profile_source
    from ..search.triggers import build_candidates
    from ..funnel import FunnelPolicy, run_funnel
    import tempfile
    import os

    # Create temp CSV
    rng = random.Random(42)
    lines = ["x,y,label"]
    for _ in range(n_rows):
        x = round(rng.gauss(0, 1), 4)
        y = round(rng.gauss(5, 2), 4)
        label = 1 if x + y > 5 else 0
        lines.append(f"{x},{y},{label}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("\n".join(lines))
        csv_path = f.name

    try:
        times = []
        for _ in range(n_runs):
            adapter = CsvDataSourceAdapter(csv_path)
            config = ProfileConfig(mode=ProfileMode.FAST)
            t0 = time.perf_counter()
            profile = profile_source(adapter, config)
            proposals = build_candidates(profile, with_interactions=True, max_interactions=4)
            # Materialize sample
            sample = {}
            for col in profile.columns:
                col_data = []
                for batch in adapter.scan(projection=[col.name]):
                    col_data.extend(batch.columns.get(col.name, []))
                converted = []
                for v in col_data[:100]:
                    try:
                        converted.append(float(v))
                    except (ValueError, TypeError):
                        converted.append(None)
                sample[col.name] = converted
            for p in proposals[:10]:
                run_funnel(p, profile, sample, FunnelPolicy())
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        avg_ms = sum(times) / len(times)
        return BenchmarkResult(
            name="pipeline_latency",
            duration_ms=avg_ms,
            rows=n_rows,
            details={"proposals": len(proposals), "columns": len(profile.columns)},
        )
    finally:
        os.unlink(csv_path)


def run_full_benchmark() -> BenchmarkSuite:
    """Run the complete benchmark suite (doc 12)."""
    suite = BenchmarkSuite()

    # Operator throughput
    for family in ["numeric", "numeric interaction", "temporal", "datetime"]:
        results = benchmark_operator_throughput(family=family, n_rows=5000)
        suite.results.extend(results)

    # Pipeline latency
    pipeline_result = benchmark_pipeline_latency(n_rows=500)
    suite.results.append(pipeline_result)

    suite.total_duration_ms = sum(r.duration_ms for r in suite.results)
    suite.summary = {
        "total_benchmarks": len(suite.results),
        "total_duration_ms": round(suite.total_duration_ms, 2),
        "avg_ops_per_sec": round(
            sum(r.ops_per_sec for r in suite.results if r.ops_per_sec > 0) /
            max(1, sum(1 for r in suite.results if r.ops_per_sec > 0)), 2
        ),
    }

    return suite
