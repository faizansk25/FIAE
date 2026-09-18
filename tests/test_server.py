"""Tests for the FIAE REST API + dashboard server (stdlib http.server)."""

import json
import threading
import urllib.request

import pytest

from fiae.server import JobRegistry, _safe_run_name, make_handler


# ----------------------------------------------------------------- unit
def test_safe_run_name_rejects_traversal():
    assert not _safe_run_name("../etc/passwd")
    assert not _safe_run_name("a/b")
    assert not _safe_run_name("a\\b")
    assert not _safe_run_name("..")
    assert not _safe_run_name("")
    assert not _safe_run_name("x" * 200)
    assert _safe_run_name("demo_track")
    assert _safe_run_name("run-01.v2")


def test_job_registry_lifecycle():
    reg = JobRegistry()
    jid = reg.create({"source": "a.csv", "target": "y"})
    assert reg.get(jid)["state"] == "PENDING"
    reg.start(jid)
    assert reg.get(jid)["state"] == "RUNNING"
    reg.complete(jid, {"portfolio_size": 3})
    assert reg.get(jid)["state"] == "COMPLETED"
    assert reg.get(jid)["result"]["portfolio_size"] == 3
    # listing never leaks full results
    assert all("result" not in j for j in reg.list())


# ----------------------------------------------------------------- http
@pytest.fixture()
def server(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    # fabricate one completed run
    run = runs_dir / "demo_track"
    run.mkdir()
    (run / "run_start.json").write_text(
        json.dumps({"run_id": "demo", "params": {"target": "dept"}}))
    (run / "run_end.json").write_text(json.dumps({"state": "COMPLETED"}))
    (run / "metrics.jsonl").write_text(
        '{"rows": 100}\n{"rows": 200}\n')

    from http.server import ThreadingHTTPServer
    handler = make_handler(JobRegistry(), str(runs_dir))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()


def _get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def test_health(server):
    status, body = _get(server, "/api/health")
    assert status == 200
    assert body["status"] == "ok"


def test_list_runs(server):
    status, body = _get(server, "/api/runs")
    assert status == 200
    assert body["runs"][0]["run_id"] == "demo_track"
    assert body["runs"][0]["state"] == "COMPLETED"


def test_run_detail(server):
    status, body = _get(server, "/api/runs/demo_track")
    assert status == 200
    assert body["metrics"] == [{"rows": 100}, {"rows": 200}]


def test_run_detail_rejects_traversal(server):
    status, _body = _get(server, "/api/runs/..%2F..%2Fetc")
    assert status == 400


def test_unknown_routes_404(server):
    status, _ = _get(server, "/api/nope")
    assert status == 404
    status, _ = _get(server, "/api/runs/does_not_exist")
    assert status == 404


def test_dashboard_html(server):
    with urllib.request.urlopen(server + "/dashboard", timeout=5) as resp:
        html = resp.read().decode()
    assert "FIAE Dashboard" in html
    assert "demo_track" in html


def test_root_serves_gui(server):
    with urllib.request.urlopen(server + "/", timeout=5) as resp:
        html = resp.read().decode()
    assert "FIAE" in html
    assert "Connect to a Data Source" in html


def test_learn_endpoint_validates_source(server, tmp_path):
    req = urllib.request.Request(
        server + "/api/learn",
        data=json.dumps({"source": "missing.csv", "target": "y"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST")
    try:
        urllib.request.urlopen(req, timeout=5)
        raise AssertionError("expected HTTP 400")
    except urllib.error.HTTPError as err:
        assert err.code == 400
