"""End-to-end FIAE example: the canonical 10-phase pipeline on churn data.

Generates a synthetic churn dataset locally (no downloads, no credentials,
CPU-only), then runs the full canonical pipeline — intake -> profile ->
validate -> splits -> generate -> funnel -> HPO -> ensemble -> evaluate ->
verified export — and prints a per-phase summary.

Run:  python examples/churn/run_api.py
"""

from __future__ import annotations

import csv
import math
import os
import random
import sys
import tempfile

# Allow running from a source checkout without installation.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from fiae.pipeline import run_canonical_pipeline


def generate_dataset(path: str, n_rows: int = 400, seed: int = 42) -> str:
    """Write a synthetic churn dataset with learnable structure."""
    rng = random.Random(seed)
    cities = ["nyc", "sf", "la", "chicago"]
    plans = ["basic", "plus", "premium"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["customer_id", "tenure_months", "monthly_fee",
                    "support_tickets", "avg_minutes", "city", "plan",
                    "is_returned"])
        for i in range(n_rows):
            tenure = rng.randint(1, 72)
            fee = round(20 + 60 * rng.random(), 2)
            tickets = rng.randint(0, 8)
            minutes = round(30 + 400 * rng.random(), 1)
            city = rng.choice(cities)
            plan = rng.choice(plans)
            # Learnable churn signal: short tenure, high fee, many tickets.
            logit = (
                2.2
                - 0.06 * tenure
                + 0.04 * fee
                + 0.45 * tickets
                + 0.002 * minutes
                + (0.5 if plan == "premium" else 0.0)
                + 0.3 * math.log1p(fee)
            )
            churn = 1 if rng.random() < 1 / (1 + math.exp(-logit)) else 0
            w.writerow([f"CUST-{i:05d}", tenure, fee, tickets, minutes,
                        city, plan, churn])
    return path


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="fiae_example_")
    csv_path = os.path.join(tmp, "churn.csv")
    generate_dataset(csv_path)
    print(f"dataset:    {csv_path} (400 rows, synthetic churn)")

    result = run_canonical_pipeline(csv_path, target="is_returned")
    summary = result.summary()

    print()
    print("canonical pipeline (doc 15) — 10 phases")
    print("=" * 52)
    print(f"run_id:          {summary['run_id']}")
    print(f"phases complete: {summary['phases_completed']}/10")
    print(f"portfolio size:  {summary['portfolio_size']} features")
    print(f"best model:      {summary['best_model']}")
    print(f"total time:      {summary['total_time_s']:.2f}s")
    print(f"errors:          {summary['errors']}")
    if summary["export_path"]:
        print(f"export:          {summary['export_path']}")

    if summary["errors"]:
        print("\ncompleted with errors — see run artifacts for details")
    else:
        print("\nall phases completed cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
