"""Tests for the FIAE web GUI + GUI backend endpoints.

Covers:
- gui_html: structure, tabs, no external CDN dependencies
- profile_job: real CSV profiling through the job path
- server routes: / (GUI), /gui, /api/sources, POST /api/profile,
  kind tagging on jobs, backpressure on profile endpoint
"""

from __future__ import annotations

import csv
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from fiae.server import (
    FIAEHTTPServer,
    JobRegistry,
    JobWorkerPool,
    RateLimiter,
    make_handler,
)
from fiae.webgui import gui_html, profile_job


# ---------------------------------------------------------------------------
# GUI HTML
# ---------------------------------------------------------------------------


class TestGuiHtml:
    def test_contains_all_tabs(self):
        html = gui_html()
        for tab in ("connect", "learn", "runs", "jobs", "about"):
            assert f'tab-{tab}' in html

    def test_no_external_resources(self):
        """Zero-dependency guarantee: no CDN, no http(s) asset links."""
        html = gui_html()
        for marker in ("cdn.jsdelivr", "unpkg.com", "googleapis.com",
                       "<script src=", 'link rel="stylesheet" href="http'):
            assert marker not in html

    def test_brand_colors_present(self):
        html = gui_html()
        assert "#c084fc" in html  # FIAE purple
        assert "#8b5cf6" in html

    def test_html_is_valid_ish(self):
        html = gui_html()
        assert html.startswith("<!DOCTYPE html>")
        assert html.count("<section") == html.count("</section>") == 12


# ---------------------------------------------------------------------------
# profile_job (real profiling through the worker path)
# ---------------------------------------------------------------------------


def _make_csv(tmp_path, n=80):
    p = tmp_path / "gui.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["num", "cat", "y"])
        for i in range(n):
            w.writerow([i, ["a", "b"][i % 2], i % 2])
    return str(p)


class TestProfileJob:
    def test_profile_csv(self, tmp_path):
        result = profile_job(_make_csv(tmp_path))
        assert result["rows_observed"] == 80
        assert len(result["columns"]) == 3
        names = {c["name"] for c in result["columns"]}
        assert names == {"num", "cat", "y"}
        assert result["source_id"].startswith("csv|")
        assert result["dataset_fingerprint"]

    def test_profile_with_table_kwarg(self, tmp_path):
        """table kwarg is accepted and ignored for CSV sources."""
        result = profile_job(_make_csv(tmp_path), table=None)
        assert result["rows_observed"] == 80

    def test_profile_missing_source_raises(self, tmp_path):
        with pytest.raises(Exception):
            profile_job(str(tmp_path / "missing.csv"))


class TestProfileEnrichment:
    def test_preview_rows_present(self, tmp_path):
        result = profile_job(_make_csv(tmp_path, n=50))
        assert len(result["preview"]["rows"]) == 20
        assert result["preview"]["columns"] == ["num", "cat", "y"]
        # cells are compact strings or None
        cell = result["preview"]["rows"][0][0]
        assert cell is None or isinstance(cell, str)

    def test_preview_truncates_long_cells(self, tmp_path):
        p = tmp_path / "long.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["txt"])
            w.writerow(["x" * 100])
        result = profile_job(str(p))
        assert len(result["preview"]["rows"][0][0]) <= 40

    def test_suggested_targets_ranks_and_excludes_id(self, tmp_path):
        p = tmp_path / "rich.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "flag", "dept"])
            for i in range(40):
                w.writerow([i, i % 2, ["a", "b"][i % 2]])
        result = profile_job(str(p))
        names = [t["column"] for t in result["suggested_targets"]]
        assert "id" not in names          # identifier excluded
        assert "dept" in names            # categorical suggested
        assert names[0] != ""             # ranked, not empty

    def test_suggested_targets_have_reasons(self, tmp_path):
        result = profile_job(_make_csv(tmp_path))
        for t in result["suggested_targets"]:
            assert t["reasons"]
            assert isinstance(t["score"], float)

    def test_constant_column_not_suggested(self, tmp_path):
        p = tmp_path / "const.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["const", "ok"])
            for i in range(20):
                w.writerow(["same", i % 2])
        result = profile_job(str(p))
        names = [t["column"] for t in result["suggested_targets"]]
        assert "const" not in names


# ---------------------------------------------------------------------------
# Server routes
# ---------------------------------------------------------------------------


@pytest.fixture()
def gui_server(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    registry = JobRegistry()
    pool = JobWorkerPool(registry, max_workers=2, max_queue_size=5)
    limiter = RateLimiter(max_per_second=10_000)
    pool.start()
    httpd = FIAEHTTPServer(
        ("127.0.0.1", 0),
        make_handler(registry, str(runs_dir), pool=pool, limiter=limiter))
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}", registry, pool
    httpd.shutdown()
    pool.shutdown()


def _post(base, path, payload):
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read()), ""
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read()), ""

def _get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=5) as resp:
            ct = resp.headers.get("Content-Type", "")
            raw = resp.read()
            if "json" in ct:
                return resp.status, json.loads(raw), ct
            return resp.status, raw.decode("utf-8", "replace"), ct
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read()), ""


def _wait_job(base, job_id, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        _, job, _ = _get(base, "/api/jobs/" + job_id)
        if job["state"] in ("COMPLETED", "FAILED"):
            return job
        time.sleep(0.1)
    raise TimeoutError(job_id)


class TestGuiRoutes:
    def test_root_serves_gui(self, gui_server):
        base, _, _ = gui_server
        status, _, ct = _get(base, "/")
        assert status == 200
        assert "text/html" in ct

    def test_gui_route_serves_gui(self, gui_server):
        base, _, _ = gui_server
        status, _, _ = _get(base, "/gui")
        assert status == 200

    def test_api_sources(self, gui_server):
        base, _, _ = gui_server
        status, body, _ = _get(base, "/api/sources")
        assert status == 200
        cats = body["sources"]
        assert "Data warehouses" in cats
        assert "File formats" in cats
        total = sum(len(v) for v in cats.values())
        assert total >= 20

    def test_profile_endpoint_success(self, gui_server, tmp_path):
        base, _, _ = gui_server
        status, body, _ = _post(base, "/api/profile",
                                {"source": _make_csv(tmp_path)})
        assert status == 202
        job = _wait_job(base, body["job_id"])
        assert job["state"] == "COMPLETED"
        assert job["params"]["kind"] == "profile"
        assert job["result"]["rows_observed"] == 80

    def test_profile_endpoint_missing_file(self, gui_server, tmp_path):
        base, _, _ = gui_server
        status, _body, _ = _post(base, "/api/profile",
                                {"source": str(tmp_path / "nope.csv")})
        assert status == 400

    def test_profile_requires_source(self, gui_server):
        base, _, _ = gui_server
        status, _body, _ = _post(base, "/api/profile", {})
        assert status == 400

    def test_learn_accepts_uri_source(self, gui_server, tmp_path):
        """Regression: /api/learn must accept sqlite:// URIs, not only files."""
        base, _, _ = gui_server
        # Build a real sqlite DB for the job to consume
        import sqlite3
        db_path = tmp_path / "gui.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE t (a INTEGER, y INTEGER)")
        conn.executemany("INSERT INTO t VALUES (?, ?)",
                         [(i, i % 2) for i in range(40)])
        conn.commit()
        conn.close()
        _, body, _ = _post(base, "/api/learn",
                           {"source": f"sqlite:///{db_path}", "target": "y",
                            "max_rows": 100})
        assert "error" not in body
        job = _wait_job(base, body["job_id"])
        assert job["state"] == "COMPLETED"

    def test_learn_job_tagged_learn(self, gui_server, tmp_path):
        base, _, _ = gui_server
        _, body, _ = _post(base, "/api/learn",
                           {"source": _make_csv(tmp_path), "target": "y",
                            "max_rows": 100})
        job = _wait_job(base, body["job_id"])
        assert job["params"]["kind"] == "learn"
        assert job["state"] == "COMPLETED"

    def test_job_listing_does_not_leak_results(self, gui_server, tmp_path):
        base, _, _ = gui_server
        _, body, _ = _post(base, "/api/profile",
                           {"source": _make_csv(tmp_path)})
        _wait_job(base, body["job_id"])
        _, listing, _ = _get(base, "/api/jobs")
        assert all("result" not in j for j in listing["jobs"])

    def test_export_json_for_profile_job(self, gui_server, tmp_path):
        base, _, _ = gui_server
        _, body, _ = _post(base, "/api/profile",
                           {"source": _make_csv(tmp_path)})
        _wait_job(base, body["job_id"])
        req = urllib.request.Request(
            base + "/api/jobs/" + body["job_id"] + "/export")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            assert "text/csv" not in resp.headers.get("Content-Type", "")
        assert data["rows_observed"] == 80

    def test_export_csv_for_learn_job(self, gui_server, tmp_path):
        base, _, _ = gui_server
        _, body, _ = _post(base, "/api/learn",
                           {"source": _make_csv(tmp_path), "target": "y",
                            "max_rows": 100})
        _wait_job(base, body["job_id"])
        req = urllib.request.Request(
            base + "/api/jobs/" + body["job_id"] + "/export")
        with urllib.request.urlopen(req, timeout=5) as resp:
            ctype = resp.headers.get("Content-Type", "")
            raw = resp.read().decode()
        assert "text/csv" in ctype
        header = raw.splitlines()[0]
        assert "operator" in header and "incremental_gain" in header

    def test_export_incomplete_job_409(self, gui_server, tmp_path):
        base, registry, _pool = gui_server
        jid = registry.create({"kind": "learn"})  # never submitted
        status, _body, _ = _get(base, f"/api/jobs/{jid}/export")
        assert status == 409

    def test_export_unknown_job_404(self, gui_server):
        base, _, _ = gui_server
        status, _, _ = _get(base, "/api/jobs/nope/export")
        assert status == 404


# ---------------------------------------------------------------------------
# Report engine (Kaggle-style auto-EDA)
# ---------------------------------------------------------------------------


class TestReportJob:
    def test_report_numeric_and_categorical(self, tmp_path):
        from fiae.report import build_report
        p = tmp_path / "rep.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["age", "city", "y"])
            for i in range(100):
                w.writerow([i * 1.5, ["nyc", "sf", "la"][i % 3], i % 2])
        r = build_report(str(p), target="y")
        assert r["rows_scanned"] == 100
        assert r["n_columns"] == 3
        assert "age" in r["column_stats"]
        age = r["column_stats"]["age"]
        assert age["kind"] == "numeric"
        assert age["count"] == 100
        assert age["min"] == 0.0 and age["max"] == 148.5
        assert len(age["histogram"]["counts"]) > 0
        city = r["column_stats"]["city"]
        assert city["kind"] == "categorical"
        assert city["distinct"] == 3
        top = {t["value"]: t["count"] for t in city["top_values"]}
        assert top == {"nyc": 34, "sf": 33, "la": 33}

    def test_report_target_balance(self, tmp_path):
        from fiae.report import build_report
        p = tmp_path / "bal.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["y"])
            for i in range(90):
                w.writerow(["yes" if i % 3 else "no"])
        r = build_report(str(p), target="y")
        bal = r["target_balance"]
        assert bal["column"] == "y"
        vals = {c["value"]: c["count"] for c in bal["classes"]}
        assert vals == {"yes": 60, "no": 30}
        assert bal["imbalance_ratio"] == 2.0

    def test_report_correlation(self, tmp_path):
        from fiae.report import build_report
        p = tmp_path / "corr.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["a", "b"])
            for i in range(50):
                w.writerow([i, 2 * i])
        r = build_report(str(p))
        cols = r["correlation"]["columns"]
        assert cols == ["a", "b"]
        assert r["correlation"]["matrix"][0][1] == 1.0

    def test_report_missing_map(self, tmp_path):
        from fiae.report import build_report
        p = tmp_path / "miss.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["a", "b"])
            for i in range(10):
                w.writerow([i, "" if i < 5 else "x"])
        r = build_report(str(p))
        mm = {m["column"]: m["null_fraction"] for m in r["missing_map"]}
        assert mm["a"] == 0.0 and mm["b"] == 0.5

    def test_report_endpoint_optional_target(self, gui_server, tmp_path):
        base, _, _ = gui_server
        status, body, _ = _post(base, "/api/job/run",
                                {"kind": "report",
                                 "source": _make_csv(tmp_path)})
        assert status == 202
        job = _wait_job(base, body["job_id"])
        assert job["state"] == "COMPLETED"
        res = job["result"]
        assert res["rows_scanned"] == 80
        assert res["n_columns"] == 3
        assert res["target_balance"] is None

    def test_report_endpoint_with_target(self, gui_server, tmp_path):
        base, _, _ = gui_server
        status, body, _ = _post(base, "/api/job/run",
                                {"kind": "report",
                                 "source": _make_csv(tmp_path),
                                 "target": "y"})
        assert status == 202
        job = _wait_job(base, body["job_id"])
        assert job["state"] == "COMPLETED"
        assert job["result"]["target_balance"]["column"] == "y"

    def test_report_endpoint_missing_source_400(self, gui_server):
        base, _, _ = gui_server
        status, _body, _ = _post(base, "/api/job/run", {"kind": "report"})
        assert status == 400

    def test_gui_has_report_tab(self):
        html = gui_html()
        assert 'id="btnReport"' in html
        assert 'id="repContent"' in html
        assert "svgBarChart" in html
        assert "renderReport" in html


class TestReportAdvancedCharts:
    """M20: advanced chart data (KDE, outliers, scatter, ridgeline)."""

    def test_kde_curve_present_and_normalized(self, tmp_path):
        from fiae.report import build_report
        p = tmp_path / "kde.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["a"])
            for i in range(60):
                w.writerow([i + (2 if i % 5 else 0)])
        r = build_report(str(p))
        d = r["column_stats"]["a"]["density"]
        assert len(d["x"]) == 64 and len(d["x"]) == len(d["y"])
        assert max(d["y"]) == 1.0 and min(d["y"]) >= 0.0

    def test_outlier_summary_iqr(self, tmp_path):
        from fiae.report import build_report
        p = tmp_path / "out.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["a"])
            for i in range(100):
                w.writerow([i])
            w.writerow([10000])  # extreme outlier
        r = build_report(str(p))
        of = r["column_stats"]["a"]["outliers"]
        assert of["count"] == 1
        assert of["examples"] == [10000.0]
        assert of["lo_fence"] is not None and of["hi_fence"] is not None

    def test_scatter_pairs_strongest_first(self, tmp_path):
        from fiae.report import build_report
        p = tmp_path / "sc.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["a", "b", "noise"])
            for i in range(120):
                w.writerow([i, 2 * i, (i * 37) % 101])
        r = build_report(str(p))
        pairs = r["scatter_pairs"]
        assert pairs, "expected at least one scatter pair"
        assert pairs[0]["r"] == 1.0
        assert pairs[0]["x"] in ("a", "b") and pairs[0]["y"] in ("a", "b")
        assert 0 < len(pairs[0]["points"]) <= 400

    def test_ridgeline_present_for_categorical_target(self, tmp_path):
        from fiae.report import build_report
        p = tmp_path / "ridge.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["v", "g"])
            for i in range(200):
                w.writerow([i * (1 if i % 2 else 2), "even" if i % 2 else "odd"])
        r = build_report(str(p), target="g")
        ridge = r["ridgeline"]
        assert ridge is not None and ridge["target"] == "g"
        assert "v" in ridge["columns"]
        groups = ridge["columns"]["v"]["groups"]
        assert {g["value"] for g in groups} == {"even", "odd"}
        assert all(len(g["density"]["x"]) == 64 for g in groups)

    def test_ridgeline_absent_without_target(self, tmp_path):
        from fiae.report import build_report
        p = tmp_path / "notgt.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["a"])
            for i in range(30):
                w.writerow([i])
        assert build_report(str(p))["ridgeline"] is None

    def test_gui_contains_advanced_renderers(self):
        html = gui_html()
        for tok in ("svgBoxViolin", "svgHexbin", "svgPareto", "svgRidgeline",
                    "svgDonut", "svgWaffle", "repScatterPanel", "repCompPanel"):
            assert tok in html


class TestAnalystLayer:
    """M21: data-first chart planning (classification, gating, insights)."""

    def _mk(self, tmp_path, rows):
        import csv as _csv
        p = tmp_path / "analyst.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(list(rows[0].keys()))
            for r in rows:
                w.writerow(list(r.values()))
        return str(p)

    def test_identifier_and_constant_excluded(self, tmp_path):
        from fiae.report import build_report
        p = self._mk(tmp_path, [
            {"id": i, "flag": "X", "val": i * 2} for i in range(50)])
        plan = build_report(p)["chart_plan"]
        assert plan["roles"]["id"]["role"] == "identifier"
        assert plan["roles"]["flag"]["role"] == "constant"
        assert "id" in plan["excluded"] and "flag" in plan["excluded"]
        assert "val" in plan["plan"]

    def test_id_named_column_excluded_even_if_not_unique(self, tmp_path):
        from fiae.report import build_report
        # account_id repeats but is named *_id -> treated as identifier
        p = self._mk(tmp_path, [
            {"account_id": i % 5, "amount": i} for i in range(50)])
        plan = build_report(p)["chart_plan"]
        assert plan["roles"]["account_id"]["role"] == "identifier"
        assert "account_id" in plan["excluded"]

    def test_numeric_plan_has_reasons(self, tmp_path):
        from fiae.report import build_report
        p = self._mk(tmp_path, [
            {"v": i, "c": "ab"[i % 2]} for i in range(40)])
        plan = build_report(p)["chart_plan"]
        charts = plan["plan"]["v"]
        assert all("why" in c for c in charts)
        assert {"histogram", "violin"} <= {c["chart"] for c in charts}

    def test_skewed_numeric_gets_density(self, tmp_path):
        from fiae.report import build_report
        p = self._mk(tmp_path, [
            {"v": 2 ** i} for i in range(30)])  # heavily right-skewed
        plan = build_report(p)["chart_plan"]
        assert "density" in {c["chart"] for c in plan["plan"]["v"]}

    def test_weak_correlations_drop_scatter(self, tmp_path):
        from fiae.report import build_report
        import random
        random.seed(3)
        p = self._mk(tmp_path, [
            {"a": random.random(), "b": random.random(),
             "c": random.random()} for _ in range(200)])
        plan = build_report(p)["chart_plan"]
        assert plan["strong_pairs"] == []

    def test_strong_correlation_keeps_scatter(self, tmp_path):
        from fiae.report import build_report
        p = self._mk(tmp_path, [{"a": i, "b": 3 * i} for i in range(100)])
        plan = build_report(p)["chart_plan"]
        assert len(plan["strong_pairs"]) == 1
        assert plan["strong_pairs"][0]["r"] >= 0.99

    def test_insights_generated(self, tmp_path):
        from fiae.report import build_report
        rows = [{"y": "a" if i % 3 else "b", "v": i}
                for i in range(90)]
        p = self._mk(tmp_path, rows)
        r = build_report(p, target="y")
        texts = " ".join(i["text"] for i in r["insights"])
        assert "imbalanced" in texts
        assert any(i["severity"] in ("info", "warning") for i in r["insights"])

    def test_insights_flag_outliers_and_skew(self, tmp_path):
        from fiae.report import build_report
        rows = [{"v": i} for i in range(100)] + [{"v": 100000}]
        p = self._mk(tmp_path, rows)
        r = build_report(p)
        texts = " ".join(i["text"] for i in r["insights"])
        assert "outliers" in texts


class TestTimeSeriesLayer:
    """M22: time-series intelligence (trend, ACF, seasonality)."""

    def _mk_ts(self, tmp_path, rows, name="ts.csv"):
        p = tmp_path / name
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(list(rows[0].keys()))
            for r in rows:
                w.writerow(list(r.values()))
        return str(p)

    def test_unit_helpers(self):
        from fiae.report import _acf, _dominant_lag, _seasonality_strength
        # strong linear ramp -> lag-1 autocorrelation near 1
        ramp = [float(i) for i in range(200)]
        a = _acf(ramp, max_lag=10)
        assert a[0] == 1.0
        assert a[1] > 0.9
        # white-ish alternating series -> lag-1 strongly negative
        alt = [1.0 if i % 2 else -1.0 for i in range(200)]
        assert _acf(alt, max_lag=2)[1] < -0.9
        # dominant lag: season of 7 shows up above threshold
        import math as _m
        seas = [(_m.sin(2 * _m.pi * i / 7),)[0] for i in range(140)]
        assert _dominant_lag(_acf(seas, max_lag=24)) in (6, 7, 8)
        # seasonality strength: pure period-7 sine explains almost everything
        assert _seasonality_strength(seas, 7) > 0.9
        assert _seasonality_strength(ramp, 7) < 0.2

    def test_trend_slope_and_direction(self):
        from fiae.report import _linear_trend
        import datetime as dt
        t0 = dt.datetime(2026, 1, 1)
        ts = [(t0 + dt.timedelta(days=i)).timestamp() for i in range(60)]
        up = _linear_trend(ts, [float(i) for i in range(60)])
        assert up["slope_per_day"] > 0.9 and up["r"] > 0.99
        down = _linear_trend(ts, [float(60 - i) for i in range(60)])
        assert down["slope_per_day"] < -0.9
        flat = _linear_trend(ts, [5.0] * 60)
        assert flat["r"] == 0.0

    def test_timeseries_detected_on_datetime_index(self, tmp_path):
        from fiae.report import build_report
        import datetime as dt
        t0 = dt.datetime(2026, 1, 1)
        rows = []
        for i in range(120):
            d = (t0 + dt.timedelta(days=i)).strftime("%Y-%m-%d")
            rows.append({"date": d, "v": float(i) + (5 if i % 7 == 0 else 0)})
        p = self._mk_ts(tmp_path, rows)
        r = build_report(p)
        tsr = r["timeseries"]
        assert tsr is not None
        assert tsr["index_column"] == "date"
        assert "v" in tsr["series"]
        entry = tsr["series"]["v"]
        assert entry["n"] == 120
        assert abs(entry["trend"]["slope_per_day"] - 1.0) < 0.1
        assert len(entry["acf"]) == 25  # lag 0..24
        assert entry["acf"][0] == 1.0

    def test_timeseries_absent_without_datetime(self, tmp_path):
        from fiae.report import build_report
        p = self._mk_ts(tmp_path, [{"a": i, "b": 2 * i} for i in range(80)])
        assert build_report(p)["timeseries"] is None

    def test_timeseries_absent_when_series_too_short(self, tmp_path):
        from fiae.report import build_report
        import datetime as dt
        t0 = dt.datetime(2026, 1, 1)
        rows = [{"date": (t0 + dt.timedelta(days=i)).strftime("%Y-%m-%d"),
                 "v": float(i)} for i in range(10)]  # < MIN_TS_POINTS
        p = self._mk_ts(tmp_path, rows)
        assert build_report(p)["timeseries"] is None

    def test_trend_insight_advises_time_aware_splits(self, tmp_path):
        from fiae.report import build_report
        import datetime as dt
        t0 = dt.datetime(2026, 1, 1)
        rows = [{"date": (t0 + dt.timedelta(days=i)).strftime("%Y-%m-%d"),
                 "v": float(3 * i)} for i in range(100)]
        p = self._mk_ts(tmp_path, rows)
        r = build_report(p)
        texts = " ".join(i["text"] for i in r["insights"])
        assert "time-aware splits" in texts

    def test_seasonality_insight_fires(self, tmp_path):
        from fiae.report import build_report
        import datetime as dt
        import math
        t0 = dt.datetime(2026, 1, 1)
        rows = [{"date": (t0 + dt.timedelta(days=i)).strftime("%Y-%m-%d"),
                 "v": 100 * math.sin(2 * math.pi * i / 7)}
                for i in range(140)]
        p = self._mk_ts(tmp_path, rows)
        r = build_report(p)
        tsr = r["timeseries"]
        assert tsr is not None
        entry = tsr["series"]["v"]
        assert entry["seasonality"]["strength"] > 0.5
        texts = " ".join(i["text"] for i in r["insights"])
        assert "seasonal" in texts

    def test_gui_contains_timeseries_renderers(self):
        html = gui_html()
        for tok in ("svgACF", "svgTrend", "repTsPanel", "repTsIndex"):
            assert tok in html
