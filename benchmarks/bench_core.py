"""FIAE core benchmarks (audit program M7).

Measures wall time and peak memory (tracemalloc) for:
- operator throughput at 1K / 100K / 1M rows
- table shapes: narrow (2 cols), wide (50 cols), high-cardinality,
  missing-heavy (40% nulls)
- end-to-end canonical pipeline on 1K rows

Run:  python benchmarks/bench_core.py [--quick]
      (--quick skips the 1M-row tier and the full pipeline)

Results print as a table; this script is not part of the pytest run
(benchmarks belong in CI as a scheduled job, not a per-PR gate).
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import tracemalloc
from dataclasses import dataclass

sys.path.insert(0, "src")

from fiae.features.registry import get_operator
from fiae.fitted_pipeline import FittedPipeline
from fiae.pipeline.canonical import run_canonical_pipeline
from fiae.search.triggers import FeatureProposal


@dataclass
class BenchResult:
    name: str
    n_rows: int
    seconds: float
    peak_mb: float

    def row(self) -> str:
        return (
            f"{self.name:<38} {self.n_rows:>9,} {self.seconds:>9.3f}s "
            f"{self.peak_mb:>9.1f}MB"
        )


def _timed(name: str, n_rows: int, fn) -> BenchResult:
    tracemalloc.start()
    t0 = time.perf_counter()
    fn()
    dt = time.perf_counter() - t0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return BenchResult(name, n_rows, dt, peak / 1024 / 1024)


def _numeric_data(n: int, cols: int, *, null_rate: float = 0.0, seed: int = 42):
    rng = random.Random(seed)
    data = {}
    for c in range(cols):
        vals = [rng.gauss(0, 1) for _ in range(n)]
        if null_rate:
            vals = [None if rng.random() < null_rate else v for v in vals]
        data[f"x{c}"] = vals
    return data


def bench_operator(name: str, op_name: str, n_rows: int, cols: int = 1,
                   null_rate: float = 0.0) -> BenchResult:
    op = get_operator(op_name)
    data = _numeric_data(n_rows, cols, null_rate=null_rate)
    needs_state = "state" in op.transform.__code__.co_varnames
    vals = next(iter(data.values())) if cols == 1 else None
    if cols > 1 or needs_state:
        props = [FeatureProposal(op=op_name, inputs=[f"raw:x{c}"]) for c in range(cols)]
        pipe = FittedPipeline()
        return _timed(name, n_rows, lambda: pipe.fit(props, data))

    def run():
        op.transform(list(vals))

    return _timed(name, n_rows, run)


def bench_canonical(n_rows: int, tmp_dir: str) -> BenchResult:
    import csv
    import os

    path = os.path.join(tmp_dir, f"bench_{n_rows}.csv")
    rng = random.Random(7)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num1", "num2", "cat", "y"])
        for i in range(n_rows):
            w.writerow([rng.gauss(0, 1), rng.gauss(5, 2), f"c{i % 12}", i % 2])
    return _timed(
        f"canonical pipeline ({n_rows:,} rows)", n_rows,
        lambda: run_canonical_pipeline(path, "y"),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="skip 1M tier and full pipeline")
    args = parser.parse_args()

    tiers = [1_000, 100_000] + ([] if args.quick else [1_000_000])
    results: list[BenchResult] = []

    print(f"FIAE core benchmarks ({'quick' if args.quick else 'full'})")
    print("-" * 78)
    print(f"{'benchmark':<38} {'rows':>9} {'time':>10} {'peak mem':>10}")
    print("-" * 78)

    # Operator throughput by size
    for n in tiers:
        results.append(bench_operator("log1p throughput", "log1p", n))
    # Shapes at 100K
    n = 100_000
    results.append(bench_operator("narrow (log1p, 1 col)", "log1p", n, cols=1))
    results.append(bench_operator("wide (log1p, 50 cols)", "log1p", n, cols=50))
    results.append(bench_operator("missing-heavy (40% null)", "log1p", n, null_rate=0.4))
    results.append(bench_operator("standardize (fitted)", "standardize", n))
    results.append(bench_operator("rolling_mean (windowed)", "rolling_mean", n))
    results.append(bench_operator("hash_encode (md5 buckets)", "hash_encode", n))

    if not args.quick:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            results.append(bench_canonical(1_000, td))

    for r in results:
        print(r.row())
    print("-" * 78)
    print(f"total: {len(results)} benchmarks, "
          f"{sum(r.seconds for r in results):.2f}s wall")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
