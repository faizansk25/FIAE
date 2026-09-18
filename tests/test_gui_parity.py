"""Tests for CLI-parity GUI endpoints (leakage, experience, tune, codegen,
benchmarks, pipeline) via POST /api/job/run and the guijobs module."""

from __future__ import annotations

import csv
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from fiae.guijobs import (
    benchmarks_job,
    codegen_job,
    experience_job,
    leakage_job,
    tune_job,
)
from fiae.server import (
    FIAEHTTPServer,
    JobRegistry,
    JobWorkerPool,
    RateLimiter,
    make_handler,
)


def _make_csv(tmp_path, n=60):
    p = tmp_path / "parity.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "num", "cat", "y"])
        for i in range(n):
            w.writerow([i, i * 1.5, ["a", "b"][i % 2], i % 2])
    return str(p)


# ---------------------------------------------------------------------------
# Direct job functions
# ---------------------------------------------------------------------------


class TestLeakageJob:
    def test_finds_identifier(self, tmp_path):
        r = leakage_job(_make_csv(tmp_path), "y")
        types = {f["type"] for f in r["findings"]}
        assert "potential_identifier" in types  # id column
        assert r["columns_checked"] == 3
        assert r["target"] == "y"

    def test_clean_dataset(self, tmp_path):
        p = tmp_path / "clean.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["a", "y"])
            for i in range(50):
                w.writerow([i % 5, i % 2])
        r = leakage_job(str(p), "y")
        assert r["findings"] == []


class TestExperienceAndTune:
    def test_experience_missing_store(self, tmp_path):
        r = experience_job(str(tmp_path / "none.db"))
        assert r["available"] is False

    def test_experience_with_cases(self, tmp_path):

        from fiae.contracts import TrialStatus
        from fiae.experience.case import (
            CaseDiagnosis,
            CaseRecord,
        )
        from fiae.experience.store import ExperienceStore, StoreConfig

        db = str(tmp_path / "exp.db")
        store = ExperienceStore(StoreConfig(db_path=db))
        try:
            for i in range(3):
                store.write_case(CaseRecord(
                    case_id=f"c{i}", dataset_fingerprint=f"ds{i}",
                    schema_fingerprint="s",
                    diagnosis=CaseDiagnosis(status=TrialStatus.COMPLETED)))
        finally:
            store.close()
        r = experience_job(db)
        assert r["available"] is True
        assert r["total_cases"] == 3
        assert r["success_rate"] == 1.0

    def test_tune_missing_store(self, tmp_path):
        r = tune_job(str(tmp_path / "none.db"))
        assert r["available"] is False


class TestCodegenAndBenchmarks:
    def test_codegen_compliance(self):
        r = codegen_job()
        assert r["total_operators"] >= 90
        assert "compliance_rate" in r

    def test_benchmarks(self):
        r = benchmarks_job()
        assert r["total_operators"] >= 90
        assert r["transform_ops"] > 0
        assert r["catalog_ms"] >= 0


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


@pytest.fixture()
def parity_server(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    registry = JobRegistry()
    pool = JobWorkerPool(registry, max_workers=2, max_queue_size=5)
    pool.start()
    httpd = FIAEHTTPServer(
        ("127.0.0.1", 0),
        make_handler(registry, str(runs_dir), pool=pool,
                     limiter=RateLimiter(max_per_second=10_000)))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()
    pool.shutdown()


def _post(base, path, payload):
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def _wait_job(base, job_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with urllib.request.urlopen(base + "/api/jobs/" + job_id,
                                    timeout=5) as resp:
            job = json.loads(resp.read())
        if job["state"] in ("COMPLETED", "FAILED"):
            return job
        time.sleep(0.1)
    raise TimeoutError(job_id)


class TestJobRunEndpoint:
    def test_benchmarks_end_to_end(self, parity_server):
        status, body = _post(parity_server, "/api/job/run", {"kind": "benchmarks"})
        assert status == 202
        job = _wait_job(parity_server, body["job_id"])
        assert job["state"] == "COMPLETED"
        assert job["result"]["total_operators"] >= 90

    def test_leakage_end_to_end(self, parity_server, tmp_path):
        status, body = _post(parity_server, "/api/job/run", {
            "kind": "leakage", "source": _make_csv(tmp_path), "target": "y"})
        assert status == 202
        job = _wait_job(parity_server, body["job_id"])
        assert job["state"] == "COMPLETED"
        assert job["params"]["kind"] == "leakage"

    def test_unknown_kind_400(self, parity_server):
        status, _body = _post(parity_server, "/api/job/run", {"kind": "bogus"})
        assert status == 400

    def test_missing_kind_400(self, parity_server):
        status, _body = _post(parity_server, "/api/job/run", {})
        assert status == 400

    def test_leakage_requires_target(self, parity_server, tmp_path):
        status, _body = _post(parity_server, "/api/job/run", {
            "kind": "leakage", "source": _make_csv(tmp_path)})
        assert status == 400

    def test_leakage_missing_source_400(self, parity_server, tmp_path):
        status, _body = _post(parity_server, "/api/job/run", {
            "kind": "leakage", "source": str(tmp_path / "nope.csv"),
            "target": "y"})
        assert status == 400

    def test_gui_html_contains_all_tabs(self):
        from fiae.webgui import gui_html

        html = gui_html()
        for tab in ("leakage", "pipeline", "tune", "experience",
                    "operators", "benchmarks"):
            assert f'tab-{tab}' in html
        assert html.count('data-tab=') == 13  # 12 tabs + the tab-switch helper selector
