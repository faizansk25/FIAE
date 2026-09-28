import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from fiae.cli import main


@pytest.fixture()
def csv_file(tmp_path):
    rows = ["id,age,target\n"]
    for i in range(50):
        rows.append(f"u{i},{20 + i},{i % 2}\n")
    p = tmp_path / "t.csv"
    p.write_text("".join(rows), encoding="utf-8")
    return p


def test_inspect_text_output(csv_file, capsys):
    assert main(["inspect", str(csv_file)]) == 0
    out = capsys.readouterr().out
    assert "fingerprint:" in out
    assert "id" in out and "age" in out


def test_inspect_json_output(csv_file, capsys):
    assert main(["inspect", str(csv_file), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile"]["dataset_fingerprint"].startswith("ds_")
    assert len(payload["profile"]["columns"]) == 3


def test_analyze_with_target(csv_file, capsys):
    assert main(["analyze", str(csv_file), "--target", "target", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["target"] == "target"
    assert payload["target_profile"]["distinct_estimate"] == 2
    assert payload["problem"]["task"] == "binary_classification"
    assert "roc_auc" in [m["name"] for m in payload["problem"]["metrics"]]
    assert payload["problem"]["positive_class"] == "1"


def test_analyze_text_shows_task(csv_file, capsys):
    assert main(["analyze", str(csv_file), "--target", "target"]) == 0
    out = capsys.readouterr().out
    assert "task:" in out
    assert "metrics:" in out
    assert "binary_classification" in out


def test_analyze_missing_target_fails_closed(csv_file, capsys):
    rc = main(["analyze", str(csv_file), "--target", "nope", "--json"])
    assert rc == 2
    err = json.loads(capsys.readouterr().out)["error"]
    assert err["code"] == "TARGET_MISSING"


def test_analyze_constant_target_invalid(tmp_path, capsys):
    p = tmp_path / "const.csv"
    p.write_text("id,t\n1,x\n2,x\n3,x\n", encoding="utf-8")
    rc = main(["analyze", str(p), "--target", "t", "--json"])
    assert rc == 2
    err = json.loads(capsys.readouterr().out)["error"]
    assert err["code"] == "TARGET_INVALID"


def test_inspect_ndjson_does_not_crash(tmp_path, capsys):
    """Regression: NdjsonAdapter must satisfy the dialect_report contract.

    `fiae inspect` calls adapter.dialect_report() directly; adapters for
    self-describing formats inherit the BaseAdapter default report.
    """
    p = tmp_path / "s.ndjson"
    with open(p, "w", encoding="utf-8") as f:
        for i in range(20):
            f.write(json.dumps({"a": i, "b": i * 0.5}) + "\n")
    assert main(["inspect", str(p), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dialect"]["delimiter"] is None
    assert payload["profile"]["columns"]


def test_inspect_json_error_hint_free_for_valid_source(csv_file, capsys):
    """Successful JSON inspect must not emit CLI tip lines."""
    assert main(["inspect", str(csv_file), "--json"]) == 0
    out = capsys.readouterr().out
    assert "tip:" not in out
