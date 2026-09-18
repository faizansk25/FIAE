import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fiae.contracts import SemanticType
from fiae.intake.stats import Missingness
from fiae.intake.typing_engine import (
    infer_physical_type,
    infer_semantic_type,
)
import contextlib


def mk_missing(n):
    m = Missingness()
    m.n_observed = n
    return m


def test_physical_integer():
    r = infer_physical_type(["1", "2", "3", "-4"])
    assert r["physical_dtype"] == "integer"
    assert r["confidence"] == 1.0


def test_physical_float_with_integers_stays_float():
    # 1.5 eliminates integer; remaining candidates: float
    r = infer_physical_type(["1", "2", "3.5"])
    assert r["physical_dtype"] == "float"


def test_physical_mixed_stays_string():
    r = infer_physical_type(["1", "abc", "3.5", "2024-01-02"])
    assert r["physical_dtype"] == "string"


def test_physical_boolean():
    r = infer_physical_type(["true", "false", "true"])
    assert r["physical_dtype"] == "boolean"


def test_physical_date_and_timestamp():
    assert infer_physical_type(["2024-01-02", "2024-03-04"])["physical_dtype"] == "date"
    assert infer_physical_type(["2024-01-02T05:06:07", "2024-03-04T01:02:03"])[
        "physical_dtype"
    ] == "timestamp"


def test_physical_null_values_skipped():
    r = infer_physical_type([None, "5", None])
    assert r["physical_dtype"] == "integer"


def _sem(name, dtype, values, exact=True):
    import math

    from fiae.intake.stats import NumericAccumulator

    n = len(values)
    distinct = len({v for v in values if v is not None})
    lengths = [len(v) for v in values if v is not None]
    mean = sum(lengths) / max(len(lengths), 1)
    var = sum((x - mean) ** 2 for x in lengths) / max(len(lengths), 1)
    uuid_frac = sum(1 for v in values if v is not None and v.count("-") == 4) / max(
        len(values), 1
    )
    num = NumericAccumulator()
    for v in values:
        if v is None:
            continue
        with contextlib.suppress(ValueError):
            num.update(float(v.strip()))
    return infer_semantic_type(
        name,
        dtype,
        cardinality=(distinct, exact),
        distinct_ratio=distinct / n,
        missingness=mk_missing(n),
        numeric=num if num.n_numeric else None,
        length_stats={
            "mean_length": mean,
            "length_std": math.sqrt(var),
            "token_mean": mean / 4,
            "uuid_fraction": uuid_frac,
        },
    )


def test_identifier_requires_more_than_name_hint():
    # 'user_id' hint but low distinct ratio: NOT an identifier (doc 02).
    values = [str(i % 10) for i in range(100)]
    sem, _ev = _sem("user_id", "string", values)
    assert sem is not SemanticType.IDENTIFIER
    # UUID pattern IS hard evidence.
    import uuid

    uuids = [str(uuid.uuid4()) for _ in range(50)]
    sem2, _ = _sem("ref", "string", uuids)
    assert sem2 is SemanticType.IDENTIFIER


def test_low_vs_high_cardinality():
    low = ["cat_a", "cat_b", "cat_c"] * 10
    sem, _ = _sem("animal", "string", low)
    assert sem is SemanticType.LOW_CARDINALITY_CATEGORICAL
    # Variable-length values: a fixed-token pattern must not win (that would
    # be identifier evidence); high distinct ratio drives HIGH_CARDINALITY.
    high = [f"category_level_{i}" for i in range(60)]
    sem2, _ = _sem("segment", "string", high)
    assert sem2 is SemanticType.HIGH_CARDINALITY_CATEGORICAL


def test_text_detection():
    text = [
        "This is a much longer free-form sentence with several words in it "
        "that keeps going for a while.",
        "Another long free text observation describing something in natural "
        "language form here.",
    ]
    sem, _ = _sem("comment", "string", text)
    assert sem is SemanticType.FREE_TEXT


def test_count_vs_continuous_and_hints():
    sem, _ = _sem("purchases", "integer", ["1", "2", "3"])
    assert sem is SemanticType.COUNT
    sem2, _ = _sem("temperature", "float", ["1.5", "2.5", "3.5"])
    assert sem2 is SemanticType.CONTINUOUS_NUMERIC
    sem3, _ = _sem("discount_pct", "integer", ["10", "20", "30"])
    assert sem3 is SemanticType.PERCENTAGE
    sem4, _ = _sem("total_cost", "float", ["10.5", "20.5", "30.5"])
    assert sem4 is SemanticType.CURRENCY


def test_datetime_semantic():
    sem, _ = _sem("when", "timestamp", ["2024-01-01T00:00:00"])
    assert sem is SemanticType.DATETIME
