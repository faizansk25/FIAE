"""M39.2 regression tests: profiling truth (row counts and provenance).

Findings from an external audit of what users actually see:

1. ``learn()`` reported the sampling-capped ``rows_observed`` as
   ``rows_in_source`` -- "Rows: 100000" for a 2.5M-row dataset.
2. The profiler checked ``max_rows`` *after* accumulating a whole batch, so
   statistics could include rows past the declared budget.
3. The EDA report reported ``rows_scanned`` as the pre-truncation count
   while every statistic used the post-truncation data.
4. ``exact_stats`` was set from the *requested* mode, so an EXACT run that
   hit ``max_rows`` still claimed exact statistics.
5. ``estimate_rows()`` counted the header line, over-reporting every CSV
   by exactly one row.
"""

from pathlib import Path

import pytest

from fiae.contracts import (
    ProfileCoverage,
    StopReason,
    TotalKind,
    format_row_coverage,
)
from fiae.intake import ProfileConfig, ProfileMode, auto_adapter, profile_source
from fiae.learn import LearnConfig, learn
from fiae.report import build_report


@pytest.fixture()
def small_csv(tmp_path) -> Path:
    p = tmp_path / "small.csv"
    p.write_text(
        "a,b,c\n" + "".join(f"{i},{i * 2},x{i % 3}\n" for i in range(200)),
        encoding="utf-8",
    )
    return p


@pytest.fixture()
def big_csv(tmp_path) -> Path:
    p = tmp_path / "big.csv"
    p.write_text(
        "a,b\n" + "".join(f"{i},{i * 3}\n" for i in range(5000)), encoding="utf-8"
    )
    return p


@pytest.fixture()
def spiked_csv(tmp_path) -> Path:
    """Rows 0-999 are ordinary; every row past the cap is an extreme value.

    If the row budget overshoots, the numeric statistics will show it.
    """
    p = tmp_path / "spiked.csv"
    body = [f"{i},{i}\n" for i in range(1000)]
    body += [f"{i},999999\n" for i in range(1000, 2000)]
    p.write_text("a,b\n" + "".join(body), encoding="utf-8")
    return p


class TestHardRowBudget:
    def test_max_rows_is_never_overshot(self, big_csv):
        # batch_rows 512 against max_rows 1000: pre-M39.2 the second batch
        # was accumulated whole, so rows_observed came out as 1024.
        profile = profile_source(
            auto_adapter(str(big_csv), batch_rows=512),
            ProfileConfig(mode=ProfileMode.FAST, max_rows=1000),
        )
        assert profile.rows_observed == 1000
        assert profile.rows_observed <= 1000
        assert profile.source_cost["stop_reason"] == StopReason.MAX_ROWS.value

    def test_statistics_exclude_the_overshoot(self, spiked_csv):
        """Rows past the cap must not reach the numeric statistics."""
        profile = profile_source(
            auto_adapter(str(spiked_csv), batch_rows=512),
            ProfileConfig(mode=ProfileMode.FAST, max_rows=1000),
        )
        assert profile.rows_observed == 1000
        col = next(c for c in profile.columns if c.name == "b")
        assert col.statistics["numeric"]["max"] == 999.0, (
            "statistics included rows beyond max_rows")


class TestExactStatsHonesty:
    def test_exact_mode_hitting_max_rows_is_not_exact(self, big_csv):
        profile = profile_source(
            auto_adapter(str(big_csv), batch_rows=512),
            ProfileConfig(mode=ProfileMode.EXACT, max_rows=1000),
        )
        assert profile.source_cost["mode"] == "EXACT"
        assert profile.source_cost["exact_stats"] is False, (
            "a budget-capped read must not claim exact statistics")
        assert profile.source_cost["stop_reason"] == StopReason.MAX_ROWS.value

    def test_exact_mode_reading_everything_is_exact(self, small_csv):
        profile = profile_source(
            auto_adapter(str(small_csv)), ProfileConfig(mode=ProfileMode.EXACT))
        assert profile.source_cost["exact_stats"] is True
        assert profile.coverage.total_kind is TotalKind.EXACT
        assert profile.coverage.rows_total == profile.rows_observed


class TestHeaderIsNotARow:
    def test_estimate_excludes_the_header_line(self, small_csv):
        profile = profile_source(
            auto_adapter(str(small_csv)), ProfileConfig(mode=ProfileMode.FAST))
        # Pre-M39.2: 201 estimated for a 200-row CSV.
        assert profile.rows_estimated == 200
        assert profile.rows_observed == profile.rows_estimated


class TestCoverageProvenance:
    def test_unknown_total_is_never_replaced_by_the_sample_size(self, small_csv):
        cov = ProfileCoverage(rows_observed=100, rows_total=None,
                              total_kind=TotalKind.UNKNOWN,
                              stop_reason=StopReason.MAX_ROWS)
        assert cov.coverage is None
        text = format_row_coverage(cov)
        assert "unknown" in text
        assert "100 exact" not in text

    def test_estimated_total_reports_coverage_and_stop_reason(self, big_csv):
        profile = profile_source(
            auto_adapter(str(big_csv), batch_rows=512),
            ProfileConfig(mode=ProfileMode.FAST, max_rows=1000),
        )
        text = format_row_coverage(profile.coverage)
        assert "~5,000 estimated" in text
        assert "rows profiled: 1,000" in text
        assert "stopped: max_rows" in text
        assert profile.coverage.coverage == pytest.approx(0.2, rel=0.01)

    def test_one_formatter_is_shared_by_every_surface(self, big_csv):
        """CLI/GUI/API must not each invent their own row semantics."""
        report = learn(str(big_csv), "a", config=LearnConfig(max_rows=1000))
        assert report.rows_profiled == 1000
        assert report.rows_in_source == 5000
        assert report.rows_in_source_kind == "estimated"
        assert report.profile_stop_reason == StopReason.MAX_ROWS.value
        assert report.profile_coverage == pytest.approx(0.2, rel=0.01)


class TestLearnDoesNotLieAboutRows:
    def test_rows_in_source_is_the_total_not_the_sample(self, big_csv):
        # Pre-M39.2 this reported 1000 (the sample) as "rows_in_source".
        report = learn(str(big_csv), "a", config=LearnConfig(max_rows=1000))
        assert report.rows_in_source != report.rows_profiled
        assert report.rows_in_source == 5000
        assert report.rows_profiled == 1000

    def test_metrics_log_the_profiled_rows(self, big_csv):
        report = learn(str(big_csv), "a", config=LearnConfig(max_rows=1000))
        assert report.to_dict()["rows"] == 5000
        assert report.to_dict()["rows_profiled"] == 1000
        assert report.to_dict()["rows_kind"] == "estimated"


class TestReportScansWhatItReports:
    def test_rows_scanned_equals_the_rows_analyzed(self, big_csv):
        report = build_report(str(big_csv), max_rows=1000)
        assert report["rows_scanned"] == 1000, (
            "rows_scanned reported a pre-truncation count while every "
            "statistic used 1000 rows")
        assert report["rows_truncated"] is True

    def test_under_the_cap_nothing_is_truncated(self, small_csv):
        report = build_report(str(small_csv), max_rows=20_000)
        assert report["rows_scanned"] == 200
        assert report["rows_truncated"] is False
