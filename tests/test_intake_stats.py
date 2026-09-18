import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fiae.intake.stats import (
    CardinalitySketch,
    Missingness,
    NumericAccumulator,
    Welford,
)
from fiae.intake.csv_source import detect_dialect


def test_welford_matches_reference():
    import random
    import statistics

    xs = [random.gauss(10, 3) for _ in range(5000)]
    w = Welford()
    for x in xs:
        w.update(x)
    assert abs(w.mean - statistics.fmean(xs)) < 1e-9
    assert abs(w.variance - statistics.variance(xs)) < 1e-6


def test_cardinality_exact_below_cap():
    sk = CardinalitySketch(exact_cap=4096, k=256)
    for i in range(1000):
        sk.add(f"v{i}")
    est, exact = sk.estimate()
    assert exact and est == 1000


def test_cardinality_approximate_above_cap_is_bounded():
    sk = CardinalitySketch(exact_cap=1000, k=512)
    for i in range(50_000):
        sk.add(f"v{i}")
    est, exact = sk.estimate()
    assert not exact
    # KMV: order-of-magnitude correctness
    assert 25_000 < est < 100_000


def test_missingness_taxonomy():
    m = Missingness(frozenset({"NA", "NULL"}))
    for v in [None, "", "  ", "NA", "NULL", "5", "-999", "5"]:
        m.update(v)
    d = m.to_dict()
    assert d["true_null"] == 1
    assert d["empty_string"] == 1
    assert d["whitespace"] == 1
    assert d["missing_token"] == 2
    assert d["suspicious_sentinel"] == 1
    assert m.missing_total == 5  # sentinels are warnings, not missing


def test_numeric_accumulator():
    n = NumericAccumulator()
    for x in [1.0, 2.0, 3.0, 0.0, -1.0, float("nan"), float("inf")]:
        n.update(x)
    d = n.to_dict()
    assert d["min"] == -1.0
    assert d["max"] == 3.0
    assert d["nan_count"] == 1
    assert d["inf_count"] == 1
    assert d["zero_count"] == 1
    assert d["negative_count"] == 1
    assert abs(d["mean"] - 1.0) < 1e-9  # finite values only: (1+2+3+0-1)/5


def test_low_confidence_dialect_raises():
    from fiae.errors import FIAEError

    # No stable width across candidates for any delimiter -> ambiguity
    with __import__("pytest").raises(FIAEError):
        detect_dialect("\n\n   \n")
