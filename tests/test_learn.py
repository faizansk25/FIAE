"""Tests for the fiae learn end-to-end pipeline."""

import json
import os
import random
import sys
import tempfile

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))

from fiae.errors import ErrorCode, FIAEError
from fiae.learn import (
    LearnConfig, LearnReport, PortfolioMember,
    _to_floats, learn, scan_columns,
)


def _make_csv(tmp_path, n=200, seed=42):
    NL = chr(10)
    random.seed(seed)
    rows = ["price,qty,cat,target"]
    for _i in range(n):
        price = round(random.uniform(10, 100), 2)
        qty = random.randint(1, 20)
        cat = random.choice(["A", "B", "C"])
        target = 1 if price * qty > 800 else 0
        rows.append(str(price) + "," + str(qty) + "," + cat + "," + str(target))
    p = tmp_path / "data.csv"
    p.write_text(NL.join(rows) + NL, encoding="utf-8")
    return p


def test_learn_basic(tmp_path):
    csv_path = _make_csv(tmp_path)
    report = learn(str(csv_path), "target")
    assert isinstance(report, LearnReport)
    assert report.task == "binary_classification"
    assert report.task_confidence == 1.0
    assert report.positive_class == "1"
    assert report.rows_in_source == 200
    assert report.columns_in_source == 4
    assert "roc_auc" in report.metrics


def test_learn_generates_proposals(tmp_path):
    csv_path = _make_csv(tmp_path)
    report = learn(str(csv_path), "target")
    assert report.proposals_generated > 0
    assert report.proposals_after_dedup > 0
    assert len(report.funnel_results) > 0


def test_learn_funnel_passes_some(tmp_path):
    csv_path = _make_csv(tmp_path)
    report = learn(str(csv_path), "target")
    f2_pass = sum(1 for r in report.funnel_results if r.f2_passed)
    assert f2_pass > 0, "Expected at least some F2 passes"


def test_learn_rejects_missing_target(tmp_path):
    csv_path = _make_csv(tmp_path)
    with pytest.raises(FIAEError) as exc:
        learn(str(csv_path), "nonexistent_column")
    assert exc.value.code is ErrorCode.TARGET_MISSING


def test_learn_json_output(tmp_path):
    csv_path = _make_csv(tmp_path)
    report = learn(str(csv_path), "target")
    d = report.to_dict()
    json_str = json.dumps(d, default=str)
    parsed = json.loads(json_str)
    assert "task" in parsed
    assert "portfolio_size" in parsed
    assert "funnel_results" in parsed


def test_learn_portfolio_features(tmp_path):
    csv_path = _make_csv(tmp_path, n=300)
    report = learn(str(csv_path), "target")
    assert report.portfolio_size > 0
    for m in report.portfolio:
        assert isinstance(m, PortfolioMember)
        assert m.feature_id.startswith("port_")
        assert isinstance(m.f4_passed, bool)
        assert isinstance(m.f6_passed, bool)


def test_learn_scan_columns():
    from fiae.intake import CsvDataSourceAdapter
    tmp = tempfile.mktemp(suffix=".csv")
    try:
        with open(tmp, "w") as f:
            NL = chr(10)
            f.write("a,b" + NL + "1,2" + NL + "3,4" + NL + "5,6" + NL)
        adapter = CsvDataSourceAdapter(tmp)
        result = scan_columns(adapter, max_rows=100)
        assert "a" in result
        assert "b" in result
        assert len(result["a"]) == 3
        assert result["a"][0] == "1"
    finally:
        os.remove(tmp)


def test_to_floats_conversion():
    result = _to_floats(["1.5", "2.0", "abc", "", "NaN", "3"])
    assert result[0] == 1.5
    assert result[1] == 2.0
    assert result[2] is None
    assert result[3] is None
    assert result[4] is None
    assert result[5] == 3.0


def test_learn_config_overrides(tmp_path):
    csv_path = _make_csv(tmp_path)
    cfg = LearnConfig(max_portfolio_features=2, max_proposals=5)
    report = learn(str(csv_path), "target", config=cfg)
    assert report.portfolio_size <= 2
    assert report.proposals_generated <= 5
