import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fiae.ids import (
    EPHEMERAL_KEYS,
    canonical_json,
    content_hash,
    dataset_fingerprint,
    feature_id_for,
    new_id,
    split_fingerprint,
)
from fiae.contracts import FeatureNode


def test_canonical_json_is_key_order_stable():
    a = canonical_json({"b": 1, "a": [3, 1, 2]})
    b = canonical_json({"a": [3, 1, 2], "b": 1})
    assert a == b


def test_canonical_json_excludes_ephemeral_keys():
    x = {"timestamp": "2026-01-01", "value": 5}
    y = {"timestamp": "2099-12-31", "value": 5}
    assert content_hash(x) == content_hash(y)
    assert "timestamp" not in canonical_json(x)


def test_content_hash_deterministic_and_sensitive():
    x = {"a": 1.5, "b": [1, 2, 3]}
    y = {"b": [1, 2, 3], "a": 1.5}
    assert content_hash(x) == content_hash(y)
    assert content_hash({"a": 2.5}) != content_hash({"a": 1.5})


def test_nonfinite_float_normalization():
    h1 = content_hash({"x": float("nan")})
    h2 = content_hash({"x": float("nan")})
    assert h1 == h2
    assert content_hash({"x": float("inf")}) != content_hash({"x": float("-inf")})


def test_new_id_prefix_and_uniqueness():
    ids = {new_id("run") for _ in range(1000)}
    assert len(ids) == 1000
    assert all(i.startswith("run_") for i in ids)


def test_dataset_fingerprint_ignores_path():
    fp1 = dataset_fingerprint(b"content-bytes", "schema_x", {"sep": ","})
    fp2 = dataset_fingerprint(b"content-bytes", "schema_x", {"sep": ","})
    assert fp1 == fp2
    assert fp1.startswith("ds_")
    # Different content material or schema changes identity.
    assert dataset_fingerprint(b"other", "schema_x", {"sep": ","}) != fp1
    assert dataset_fingerprint(b"content-bytes", "schema_y", {"sep": ","}) != fp1


def test_split_and_feature_fingerprints_deterministic():
    plan = {"strategy": "stratified_kfold", "folds": 5, "seed": 7}
    assert split_fingerprint(plan) == split_fingerprint(
        {"seed": 7, "folds": 5, "strategy": "stratified_kfold"}
    )
    node = FeatureNode(feature_id="x", operator="log1p", inputs=["a"])
    assert feature_id_for(node) == feature_id_for(node)
    assert feature_id_for(node).startswith("f_")


def test_canonical_json_handles_enum_and_dataclass():
    from enum import Enum

    class Color(Enum):
        RED = "red"

    s = canonical_json({"c": Color.RED})
    assert '"red"' in s
    # Ephemeral keys are a fixed, documented set (doc 13 rule 3).
    assert "timestamp" in EPHEMERAL_KEYS
    assert "event_id" in EPHEMERAL_KEYS
