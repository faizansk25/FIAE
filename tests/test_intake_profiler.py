import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from fiae.errors import ErrorCode, FIAEError
from fiae.contracts import SemanticType
from fiae.intake import (
    CsvDataSourceAdapter,
    ProfileConfig,
    ProfileMode,
    profile_source,
)


def make_csv(tmp_path, name="data.csv", header="id,age,city,joined,points\n"):
    rows = [header]
    for i in range(200):
        rows.append(f"u{i:04d},{20 + i % 40},city_{i % 4},2024-0{1 + i % 9}-1{i % 9},{1000 + i}\n")
    p = tmp_path / name
    p.write_text("".join(rows), encoding="utf-8")
    return p


def profile_of(tmp_path, **cfg):
    p = make_csv(tmp_path)
    a = CsvDataSourceAdapter(p)
    return profile_source(a, ProfileConfig(**cfg)), a


def test_profile_end_to_end(tmp_path):
    profile, _adapter = profile_of(tmp_path)
    by_name = {c.name: c for c in profile.columns}
    assert set(by_name) == {"id", "age", "city", "joined", "points"}
    assert by_name["id"].semantic_type is SemanticType.IDENTIFIER
    assert by_name["age"].semantic_type is SemanticType.COUNT
    assert by_name["city"].semantic_type is SemanticType.LOW_CARDINALITY_CATEGORICAL
    assert by_name["joined"].semantic_type is SemanticType.DATETIME
    assert by_name["points"].semantic_type is SemanticType.COUNT
    assert profile.rows_observed == 200
    assert profile.dataset_fingerprint.startswith("ds_")
    assert "likely_identifier" in [f["type"] for f in profile.quality_findings]


def test_profile_fingerprint_same_content_different_path(tmp_path):
    p1 = make_csv(tmp_path, "a.csv")
    sub = tmp_path / "sub"
    sub.mkdir()
    p2 = make_csv(sub, "b.csv")
    fp1 = profile_source(CsvDataSourceAdapter(p1), ProfileConfig()).dataset_fingerprint
    fp2 = profile_source(CsvDataSourceAdapter(p2), ProfileConfig()).dataset_fingerprint
    assert fp1 == fp2


def test_profile_fingerprint_changes_with_content(tmp_path):
    p1 = make_csv(tmp_path, "a.csv")
    fp1 = profile_source(CsvDataSourceAdapter(p1), ProfileConfig()).dataset_fingerprint
    p2 = make_csv(tmp_path, "b.csv", header="id,age,city,joined,salary2\n")
    fp2 = profile_source(CsvDataSourceAdapter(p2), ProfileConfig()).dataset_fingerprint
    assert fp1 != fp2


def test_header_only_file_is_schema_ambiguity(tmp_path):
    p = tmp_path / "header_only.csv"
    p.write_text("a,b\n", encoding="utf-8")
    with pytest.raises(FIAEError) as exc:
        profile_source(CsvDataSourceAdapter(p), ProfileConfig())
    assert exc.value.code is ErrorCode.SCHEMA_AMBIGUITY


def test_zero_byte_file_fails_closed(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("", encoding="utf-8")
    with pytest.raises(FIAEError) as exc:
        profile_source(CsvDataSourceAdapter(p), ProfileConfig())
    assert exc.value.code is ErrorCode.DATA_FORMAT_ERROR


def test_malformed_rows_beyond_tolerance_abort(tmp_path):
    good = "a,b,c\n"
    rows = [good] + ["1,2,3\n"] * 90 + ["1,2\n"] * 10  # 10% malformed
    p = tmp_path / "mal.csv"
    p.write_text("".join(rows), encoding="utf-8")
    with pytest.raises(FIAEError) as exc:
        profile_source(CsvDataSourceAdapter(p), ProfileConfig())
    assert exc.value.code is ErrorCode.DATA_FORMAT_ERROR


def test_fast_mode_respects_row_budget(tmp_path):
    p = make_csv(tmp_path)
    a = CsvDataSourceAdapter(p)
    profile = profile_source(a, ProfileConfig(max_rows=50))
    assert profile.rows_observed <= 50 + a.batch_rows  # batch granularity
    assert profile.source_cost["exact_stats"] is False


def test_exact_mode_full_scan(tmp_path):
    p = make_csv(tmp_path)
    profile = profile_source(CsvDataSourceAdapter(p), ProfileConfig(mode=ProfileMode.EXACT))
    assert profile.rows_observed == 200
    assert profile.source_cost["exact_stats"] is True


def test_disguised_missing_and_missingness_taxonomy_in_profile(tmp_path):
    p = tmp_path / "sent.csv"
    p.write_text(
        "v\n" + "".join(f"{-999 if i < 30 else i}\n" for i in range(100)),
        encoding="utf-8",
    )
    profile = profile_source(CsvDataSourceAdapter(p), ProfileConfig())
    types = [f["type"] for f in profile.quality_findings]
    assert "disguised_missing_values" in types
    col = profile.columns[0]
    assert col.statistics["missingness"]["suspicious_sentinel"] == 30
    assert col.statistics["missingness"]["true_null"] == 0


def test_duplicate_columns_flagged(tmp_path):
    p = tmp_path / "dupcol.csv"
    p.write_text("a,b,c\n1,x,1\n2,y,2\n3,z,3\n", encoding="utf-8")
    profile = profile_source(CsvDataSourceAdapter(p), ProfileConfig())
    by_name = {c.name: c for c in profile.columns}
    assert by_name["c"].warnings and by_name["c"].warnings[0].startswith("duplicate_of:")
