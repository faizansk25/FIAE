"""M40.6: the CSV row estimate must not be biased by file position.

``estimate_rows`` measured average line length from a single probe of the file
*head*. Any file whose line length drifts -- a monotonically increasing id, a
timestamp column, appended records -- is therefore averaged over its shortest
or longest segment only. On a 300k-row CSV with an incrementing id the
estimate came out at 330,626 rows (+10.2%), and because coverage is computed
from ``rows_total`` that error propagated into the reported coverage too.

The fix samples evenly across the file. These tests pin the accuracy on a
deliberately drifting fixture, which is what the old implementation got wrong.
"""

import csv
import random

import pytest

from fiae.intake.csv_source import (
    ESTIMATE_PROBE_CHUNKS, CsvDataSourceAdapter,
)

# The old head-only estimate was +10.2% on this shape. Anything worse than
# this tolerance means position bias has come back.
TOLERANCE = 0.04


def _write_drifting_csv(path, rows: int) -> None:
    """Lines grow steadily: the id column gains digits as the file goes on.

    This is the shape that punishes a head-only probe -- the first 64KiB holds
    the shortest lines, so a head average underestimates the true mean line
    length and inflates the row count.
    """
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "value"])
        for i in range(rows):
            w.writerow([i, round(random.Random(i).random() * 1000, 6)])


class TestRowEstimateAccuracy:
    def test_small_file_is_exact(self, tmp_path):
        p = tmp_path / "small.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["a", "b"])
            for i in range(500):
                w.writerow([i, i * 2])

        est = CsvDataSourceAdapter(p).estimate_rows()

        assert est == 500

    def test_large_file_with_drifting_line_length(self, tmp_path):
        """The regression: head-only probing over-counted by ~10%."""
        p = tmp_path / "drifting.csv"
        rows = 300_000
        _write_drifting_csv(p, rows)

        est = CsvDataSourceAdapter(p).estimate_rows()

        assert est is not None
        error = abs(est - rows) / rows
        assert error < TOLERANCE, (
            f"row estimate {est} is {error:.1%} off the true {rows}; "
            "position bias has returned to estimate_rows")

    def test_uniform_file_is_still_accurate(self, tmp_path):
        """Spreading probes must not degrade the easy case."""
        p = tmp_path / "uniform.csv"
        rows = 200_000
        rng = random.Random(4)
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "value"])
            for i in range(rows):
                w.writerow([f"{i:08d}", f"{rng.random():.6f}"])

        est = CsvDataSourceAdapter(p).estimate_rows()

        assert abs(est - rows) / rows < TOLERANCE

    def test_header_is_not_counted_as_a_row(self, tmp_path):
        p = tmp_path / "hdr.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["a", "b"])
            for i in range(200):
                w.writerow([i, i])

        # 200 data rows; a header counted as data would report 201.
        assert CsvDataSourceAdapter(p).estimate_rows() == 200


class TestProbeSpreading:
    def test_estimate_reads_more_than_the_head(self, tmp_path):
        """More than one chunk is consulted for a large file."""
        p = tmp_path / "large.csv"
        _write_drifting_csv(p, 120_000)

        adapter = CsvDataSourceAdapter(p)
        total_bytes, _total_lines = adapter._spread_probe()

        assert total_bytes > adapter._probe_bytes, (
            "spread probe must add bytes beyond the head probe")
        assert ESTIMATE_PROBE_CHUNKS >= 2

    def test_small_file_reuses_the_head_probe(self, tmp_path):
        p = tmp_path / "tiny.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["a"])
            for i in range(50):
                w.writerow([i])

        adapter = CsvDataSourceAdapter(p)
        assert adapter._spread_probe() == (adapter._probe_bytes, adapter._probe_lines)


class TestOtherAdaptersAreNotAffected:
    """The bias was CSV-specific; these pin that the others stay honest."""

    def test_parquet_uses_exact_metadata_not_an_estimate(self, tmp_path):
        pytest.importorskip("pyarrow")
        import pyarrow as pa
        import pyarrow.parquet as pq

        from fiae.intake.adapter_file import ParquetAdapter

        table = pa.table({"a": list(range(1000))})
        p = tmp_path / "t.parquet"
        pq.write_table(table, p)

        # Exact from file metadata, so no drift bias is possible.
        assert ParquetAdapter(p).estimate_rows() == 1000

    def test_excel_declines_rather_than_guessing(self, tmp_path):
        from fiae.intake.adapter_file import ExcelAdapter

        p = tmp_path / "t.xlsx"
        p.write_bytes(b"not-really-a-workbook")

        assert ExcelAdapter(p).estimate_rows() is None
