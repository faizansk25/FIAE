import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from fiae.errors import ErrorCode, FIAEError
from fiae.intake.csv_source import (
    CsvDataSourceAdapter,
    detect_dialect,
    detect_encoding,
    detect_header,
)
from fiae.intake.base import SamplePlan


def write(tmp_path, name, text, encoding="utf-8"):
    p = tmp_path / name
    p.write_text(text, encoding=encoding)
    return p


def test_detect_encoding_bom_and_utf8():
    assert detect_encoding(b"\xef\xbb\xbfid,name") == "utf-8-sig"
    assert detect_encoding(b"id,name") == "utf-8"
    assert detect_encoding(b"\xff\xfename\x00") == "utf-16"


def test_detect_dialect_delimiters():
    assert detect_dialect("a,b\n1,2\n3,4\n")["delimiter"] == ","
    assert detect_dialect("a;b\n1;2\n3;4\n")["delimiter"] == ";"
    assert detect_dialect("a|b\n1|2\n3|4\n")["delimiter"] == "|"
    assert detect_dialect("a\tb\n1\t2\n3\t4\n")["delimiter"] == "\t"


def test_quoted_delimiters_and_newlines_do_not_confuse_width():
    text = 'name,note\n"smith, john","line1\nline2"\n"doe, jay","ok"\n'
    d = detect_dialect(text)
    assert d["delimiter"] == ","
    assert d["modal_width"] == 2


def test_header_detection():
    rows = list(__import__("csv").reader(
        ["id,age,city", "1,34,Berlin", "2,29,Paris"]))
    assert detect_header(rows) == (True, 0.95)
    data_first = __import__("csv").reader(["1,34,Berlin", "2,29,Paris"])
    rows2 = list(data_first)
    assert detect_header(rows2)[0] is False
    # duplicate header names weaken confidence below threshold
    rows3 = list(__import__("csv").reader(["a,a", "x,y"]))
    assert detect_header(rows3)[0] is False


def test_adapter_reads_quoted_multiline(tmp_path):
    p = write(tmp_path, "q.csv", 'id,note\n1,"line1\nline2"\n2,plain\n')
    a = CsvDataSourceAdapter(p)
    rows = []
    for batch in a.scan():
        for i in range(batch.n_rows):
            rows.append((batch.columns["id"][i], batch.columns["note"][i]))
    # Quoted CRLF/LF newlines are preserved verbatim inside quotes.
    assert rows[0][0] == "1" and rows[1][0] == "2"
    assert rows[1][1] == "plain"
    inner = rows[0][1]
    assert inner.startswith("line1") and inner.endswith("line2")
    # Quoted newlines (LF or CRLF) preserved inside quotes.
    assert chr(10) in inner  # contains newline between line1 and line2


def test_adapter_duplicate_column_names(tmp_path):
    p = write(tmp_path, "dup.csv", "a,a,b\n1,2,3\n")
    a = CsvDataSourceAdapter(p)
    assert a.column_names == ["a", "a_1", "b"]


def test_estimate_rows_and_bytes(tmp_path):
    lines = "id,val\n" + "".join(f"{i},{i * 2}\n" for i in range(1000))
    p = write(tmp_path, "rows.csv", lines)
    a = CsvDataSourceAdapter(p)
    est = a.estimate_rows()
    assert est is not None and 900 < est < 1100
    assert a.estimate_bytes() == p.stat().st_size


def test_multi_region_sampling_covers_middle_and_tail(tmp_path):
    # >2 blocks of data so head/middle/tail regions differ
    lines = "id,tag\n"
    for i in range(20_000):
        lines += f"{i},t{i}\n"
    p = write(tmp_path, "big.csv", lines)
    a = CsvDataSourceAdapter(p)
    seen_ids = []
    for batch in a.sample(SamplePlan(block_bytes=1024, n_middle=1, n_random=1)):
        seen_ids.extend(batch.columns["id"])
    assert len(seen_ids) > 0
    # sampled from more than just the head
    assert int(seen_ids[-1]) > 0
    high = [int(x) for x in seen_ids if int(x) > 5000]
    assert high, "sampling never reached beyond the head of the file"


def test_fingerprint_ignores_path(tmp_path):
    content = "a,b\n1,x\n2,y\n"
    p1 = write(tmp_path, "one.csv", content)
    sub = tmp_path / "sub"
    sub.mkdir()
    p2 = write(sub, "two.csv", content)
    a1, a2 = CsvDataSourceAdapter(p1), CsvDataSourceAdapter(p2)
    assert a1.fingerprint_material() == a2.fingerprint_material()


def test_verify_unchanged_detects_growth(tmp_path):
    p = write(tmp_path, "grow.csv", "a\n1\n")
    a = CsvDataSourceAdapter(p)
    p.write_text(p.read_text() + "2\n")
    with pytest.raises(FIAEError) as exc:
        a.verify_unchanged()
    assert exc.value.code is ErrorCode.SOURCE_CHANGED_DURING_RUN


def test_missing_file_fails_closed(tmp_path):
    with pytest.raises(FIAEError) as exc:
        CsvDataSourceAdapter(tmp_path / "nope.csv")
    assert exc.value.code is ErrorCode.DATA_FORMAT_ERROR
