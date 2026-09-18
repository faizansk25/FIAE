"""FIAE REST API and web dashboard (doc 10 — observability).

Zero-dependency HTTP server built on the Python standard library
(``http.server``). Serves:

- ``GET  /``                 — minimal HTML dashboard
- ``GET  /api/health``       — liveness + version + server stats
- ``GET  /api/runs``         — list tracked runs
- ``GET  /api/runs/<name>``  — manifest / metrics / audit for one run
- ``POST /api/learn``        — launch an async learn job
- ``GET  /api/jobs``         — list jobs
- ``GET  /api/jobs/<id>``    — job status/result

Concurrency model (doc 08 — parallelism, backpressure):
- Learn jobs run on a **bounded worker pool** (default 4 workers), never an
  unbounded thread-per-request model.
- When the job queue is full the server responds **429 Too Many Requests**
  (backpressure) instead of silently degrading or exhausting memory.
- A per-client sliding-window **rate limiter** sheds excess request load.
- The job registry prunes old completed jobs so thousands of sequential
  jobs cannot leak memory.

Security model (doc 11):
- Binds to 127.0.0.1 by default; remote binding requires --host 0.0.0.0
- Path traversal is rejected; only JSON artifacts inside the runs root
  are ever served.
- POST /api/learn only accepts local file paths that exist.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional

__version__ = "0.0.1"
_MAX_RUN_NAME_LEN = 128


# ---------------------------------------------------------------------------
# Job registry (thread-safe, bounded memory)
# ---------------------------------------------------------------------------

class JobRegistry:
    """Thread-safe registry of async learn jobs.

    Completed jobs are pruned (oldest first) beyond ``max_completed`` so a
    long-running server cannot grow without bound.
    """

    def __init__(self, max_completed: int = 1000) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._max_completed = max_completed

    def create(self, params: dict) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "state": "PENDING",
                "created_at": time.time(),
                "params": params,
                "result": None,
                "error": None,
            }
        return job_id

    def remove(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)

    def start(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["state"] = "RUNNING"

    def complete(self, job_id: str, result: dict) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job["state"] = "COMPLETED"
            job["result"] = result
            job["finished_at"] = time.time()
            self._prune_locked()

    def fail(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job["state"] = "FAILED"
            job["error"] = error
            job["finished_at"] = time.time()
            self._prune_locked()

    def _prune_locked(self) -> None:
        """Drop oldest COMPLETED jobs beyond the retention limit."""
        completed = [jid for jid, j in self._jobs.items()
                     if j["state"] == "COMPLETED"]
        excess = len(completed) - self._max_completed
        if excess <= 0:
            return
        # dict preserves insertion order → oldest first
        for jid in completed[:excess]:
            del self._jobs[jid]

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[dict]:
        with self._lock:
            return [
                {k: v for k, v in job.items() if k != "result"}
                for job in self._jobs.values()
            ]

    def stats(self) -> dict:
        with self._lock:
            counts: dict[str, int] = {}
            for job in self._jobs.values():
                counts[job["state"]] = counts.get(job["state"], 0) + 1
            return counts


# ---------------------------------------------------------------------------
# Rate limiter (per-client sliding window)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Sliding-window per-client rate limiter (stdlib only).

    ``allow(key)`` grants at most ``max_per_second`` requests per rolling
    one-second window. Client table is pruned when it exceeds
    ``max_clients`` so attacker IP rotation cannot leak memory.
    """

    def __init__(self, max_per_second: float = 50.0,
                 max_clients: int = 10_000) -> None:
        self.max_per_second = max_per_second
        self.max_clients = max_clients
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            if len(self._hits) > self.max_clients:
                self._prune_locked(now)
            times = self._hits.get(key)
            if times is None:
                times = deque()
                self._hits[key] = times
            window_start = now - 1.0
            while times and times[0] <= window_start:
                times.popleft()
            if len(times) >= self.max_per_second:
                return False
            times.append(now)
            return True

    def _prune_locked(self, now: float) -> None:
        # Drop clients whose last hit is older than 60s.
        stale = [k for k, dq in self._hits.items()
                 if not dq or dq[-1] < now - 60.0]
        for k in stale:
            del self._hits[k]
        if len(self._hits) > self.max_clients:
            # Still over: evict oldest by last hit.
            ordered = sorted(self._hits.items(), key=lambda kv: kv[1][-1] if kv[1] else 0)
            for k, _ in ordered[: len(self._hits) - self.max_clients]:
                del self._hits[k]


# ---------------------------------------------------------------------------
# Bounded job worker pool (doc 08: bounded parallelism + backpressure)
# ---------------------------------------------------------------------------

class JobWorkerPool:
    """Fixed-size worker pool draining a bounded FIFO job queue.

    ``submit`` returns False when the queue is full — the HTTP layer maps
    that to HTTP 429 (backpressure). Jobs never run on request threads, so
    a burst of clients cannot spawn thousands of concurrent learn runs.
    """

    def __init__(self, registry: JobRegistry, max_workers: int = 4,
                 max_queue_size: int = 100) -> None:
        self._registry = registry
        self._max_workers = max(1, max_workers)
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, max_queue_size))
        self._workers: list[threading.Thread] = []
        self._stop = threading.Event()
        self._busy = 0
        self._busy_lock = threading.Lock()

    def start(self) -> None:
        if self._workers:
            return
        for i in range(self._max_workers):
            t = threading.Thread(
                target=self._worker_loop, name=f"fiae_job_{i}", daemon=True)
            t.start()
            self._workers.append(t)

    def submit(self, job_id: str, fn: Callable[[], dict]) -> bool:
        """Queue a job. Returns False (backpressure) if the queue is full."""
        try:
            self._queue.put_nowait((job_id, fn))
            return True
        except queue.Full:
            return False

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job_id, fn = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            with self._busy_lock:
                self._busy += 1
            self._registry.start(job_id)
            try:
                result = fn()
            except Exception as exc:
                self._registry.fail(job_id, str(exc))
            else:
                self._registry.complete(job_id, result)
            finally:
                with self._busy_lock:
                    self._busy -= 1
                self._queue.task_done()

    def stats(self) -> dict:
        with self._busy_lock:
            busy = self._busy
        return {
            "workers": self._max_workers,
            "busy_workers": busy,
            "queued_jobs": self._queue.qsize(),
        }

    def shutdown(self, wait: bool = False) -> None:
        self._stop.set()
        if wait:
            for t in self._workers:
                t.join(timeout=2.0)
        self._workers.clear()


def run_learn_job(params: dict) -> dict:
    """Execute one learn job (runs on a JobWorkerPool worker thread)."""
    from .learn import learn, LearnConfig

    cfg = LearnConfig(
        max_rows=params["max_rows"],
        max_portfolio_features=params["max_features"],
        seed=params["seed"],
    )
    report = learn(params["source"], params["target"], config=cfg)
    return report.to_dict()


def run_profile_job(params: dict) -> dict:
    """Execute one profile job (GUI Connect tab)."""
    from .webgui import profile_job

    kwargs = {k: params[k] for k in ("table",) if k in params}
    return profile_job(params["source"], **kwargs)


def run_generic_job(params: dict) -> dict:
    """Dispatch a kind-tagged GUI job (leakage/experience/tune/codegen/
    benchmarks/pipeline) to its implementation in guijobs."""
    from . import guijobs

    kind = params.get("kind", "")
    if kind == "leakage":
        return guijobs.leakage_job(params["source"], params["target"])
    if kind == "experience":
        return guijobs.experience_job(params.get("db_path"))
    if kind == "tune":
        return guijobs.tune_job(params.get("db_path"),
                                int(params.get("min_cases", 5)))
    if kind == "codegen":
        return guijobs.codegen_job(params.get("operator"))
    if kind == "benchmarks":
        return guijobs.benchmarks_job()
    if kind == "pipeline":
        return guijobs.pipeline_job(params["source"], params["target"])
    if kind == "report":
        from .report import build_report
        return build_report(params["source"],
                            target=params.get("target"),
                            table=params.get("table"))
    raise ValueError("unknown job kind: " + str(kind))


def _source_is_acceptable(source: str) -> bool:
    """Local files must exist; URI-style sources pass through and the
    adapter reports a readable error if they are unreachable."""
    if "://" in source:
        return True
    return os.path.isfile(source)


def _safe_run_name(name: str) -> bool:
    """Reject path traversal and unsafe run names."""
    if not name or len(name) > _MAX_RUN_NAME_LEN:
        return False
    if name in (".", "..") or "/" in name or "\\" in name:
        return False
    return all(c.isalnum() or c in "-_." for c in name)


def make_handler(
    registry: JobRegistry,
    runs_dir: str,
    pool: Optional[JobWorkerPool] = None,
    limiter: Optional[RateLimiter] = None,
):

    if pool is None:
        pool = JobWorkerPool(registry)
        pool.start()
    if limiter is None:
        limiter = RateLimiter()

    class FIAEHandler(BaseHTTPRequestHandler):
        server_version = "FIAE/" + __version__

        # ------------------------------------------------------- helpers
        def _send_json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, indent=2, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> Optional[dict]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_048_576:
                    return None
                return json.loads(self.rfile.read(length))
            except (ValueError, json.JSONDecodeError):
                return None

        def log_message(self, fmt, *args):  # quieter logs
            pass

        def _client_key(self) -> str:
            return self.client_address[0] if self.client_address else "unknown"

        # ------------------------------------------------------- routes
        def do_GET(self) -> None:
            if not limiter.allow(self._client_key()):
                self._send_json({"error": "rate limit exceeded"}, 429)
                return
            path = self.path.split("?", 1)[0].rstrip("/") or "/"

            if path == "/" or path == "/gui":
                self._send_gui()
            elif path == "/dashboard":
                self._send_dashboard()
            elif path == "/api/sources":
                from .intake import list_supported_sources
                self._send_json(
                    {"sources": list_supported_sources()})
            elif path == "/api/health":
                self._send_json({
                    "status": "ok",
                    "version": __version__,
                    "runs_dir": os.path.abspath(runs_dir),
                    "jobs": registry.stats(),
                    "pool": pool.stats(),
                })
            elif path == "/api/runs":
                self._send_json({"runs": self._list_runs()})
            elif path.startswith("/api/runs/"):
                name = path[len("/api/runs/"):]
                if not _safe_run_name(name):
                    self._send_json({"error": "invalid run name"}, 400)
                    return
                self._send_run_detail(name)
            elif path == "/api/jobs":
                self._send_json({"jobs": registry.list()})
            elif path.startswith("/api/jobs/"):
                rest = path[len("/api/jobs/"):]
                if rest.endswith("/export"):
                    job_id = rest[: -len("/export")]
                    job = registry.get(job_id)
                    if job is None:
                        self._send_json({"error": "unknown job"}, 404)
                        return
                    if job["state"] != "COMPLETED" or not job.get("result"):
                        self._send_json(
                            {"error": "job not completed"}, 409)
                        return
                    self._export_job(job)
                else:
                    job = registry.get(rest)
                    if job is None:
                        self._send_json({"error": "unknown job"}, 404)
                    else:
                        self._send_json(job)
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            if not limiter.allow(self._client_key()):
                self._send_json({"error": "rate limit exceeded"}, 429)
                return
            path = self.path.split("?", 1)[0].rstrip("/")
            if path == "/api/profile":
                self._handle_profile_post()
                return
            if path == "/api/job/run":
                self._handle_generic_job_post()
                return
            if path != "/api/learn":
                self._send_json({"error": "not found"}, 404)
                return
            body = self._read_json_body()
            if not body or "source" not in body or "target" not in body:
                self._send_json(
                    {"error": "body must include 'source' and 'target'"}, 400)
                return
            source = str(body["source"])
            if not _source_is_acceptable(source):
                self._send_json(
                    {"error": "source file does not exist: " + source}, 400)
                return
            params = {
                "source": source,
                "target": str(body["target"]),
                "max_rows": int(body.get("max_rows", 100_000)),
                "max_features": int(body.get("max_features", 8)),
                "seed": int(body.get("seed", 0)),
            }
            job_id = registry.create({**params, "kind": "learn"})
            if not pool.submit(job_id, lambda p=params: run_learn_job(p)):
                # Backpressure (doc 08): queue full — reject, don't degrade.
                registry.remove(job_id)
                self._send_json(
                    {"error": "server busy, retry later",
                     "queued_jobs": pool.stats()["queued_jobs"]},
                    429)
                return
            self._send_json({"job_id": job_id, "state": "PENDING"}, 202)

        def _handle_profile_post(self) -> None:
            body = self._read_json_body()
            if not body or "source" not in body:
                self._send_json({"error": "body must include 'source'"}, 400)
                return
            source = str(body["source"])
            if not _source_is_acceptable(source):
                self._send_json(
                    {"error": "source file does not exist: " + source}, 400)
                return
            params = {
                "source": source,
                "table": body.get("table"),
                "kind": "profile",
            }
            job_id = registry.create(params)
            if not pool.submit(job_id,
                               lambda p=params: run_profile_job(
                                   {k: v for k, v in p.items()
                                    if k != "kind"})):
                registry.remove(job_id)
                self._send_json(
                    {"error": "server busy, retry later",
                     "queued_jobs": pool.stats()["queued_jobs"]},
                    429)
                return
            self._send_json({"job_id": job_id, "state": "PENDING"}, 202)

        # ------------------------------------------------------- data
        def _list_runs(self) -> list[dict]:
            runs = []
            if not os.path.isdir(runs_dir):
                return runs
            for name in sorted(os.listdir(runs_dir)):
                if not _safe_run_name(name):
                    continue
                run_path = os.path.join(runs_dir, name)
                if not os.path.isdir(run_path):
                    continue
                info: dict = {"run_id": name}
                start_file = os.path.join(run_path, "run_start.json")
                if os.path.isfile(start_file):
                    try:
                        with open(start_file, encoding="utf-8") as f:
                            info["params"] = json.load(f).get("params", {})
                    except (OSError, json.JSONDecodeError):
                        info["params"] = {}
                end_file = os.path.join(run_path, "run_end.json")
                info["state"] = "COMPLETED" if os.path.isfile(end_file) \
                    else "RUNNING"
                runs.append(info)
            return runs

        def _send_run_detail(self, name: str) -> None:
            run_path = os.path.join(runs_dir, name)
            if not os.path.isdir(run_path):
                self._send_json({"error": "unknown run"}, 404)
                return
            detail: dict = {"run_id": name}
            for fname, key in (("run_start.json", "start"),
                               ("run_end.json", "end"),
                               ("metrics.jsonl", "metrics")):
                fpath = os.path.join(run_path, fname)
                if not os.path.isfile(fpath):
                    continue
                try:
                    with open(fpath, encoding="utf-8") as f:
                        if fname.endswith(".jsonl"):
                            detail[key] = [
                                json.loads(line) for line in f if line.strip()]
                        else:
                            detail[key] = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
            audit_path = os.path.join(run_path, "audit.jsonl")
            if os.path.isfile(audit_path):
                try:
                    with open(audit_path, encoding="utf-8") as f:
                        detail["audit_events"] = sum(1 for _ in f)
                except OSError:
                    pass
            self._send_json(detail)

        def _handle_generic_job_post(self) -> None:
            """POST /api/job/run {kind, ...} — CLI-parity jobs."""
            body = self._read_json_body()
            if not body or "kind" not in body:
                self._send_json({"error": "body must include 'kind'"}, 400)
                return
            kind = str(body["kind"])
            if kind not in ("leakage", "experience", "tune", "codegen",
                            "benchmarks", "pipeline", "report"):
                self._send_json({"error": "unknown kind: " + kind}, 400)
                return
            if kind in ("leakage", "pipeline", "report"):
                if not body.get("source"):
                    self._send_json(
                        {"error": kind + " requires 'source'"},
                        400)
                    return
                if kind in ("leakage", "pipeline") and not body.get("target"):
                    self._send_json(
                        {"error": kind + " requires 'target'"},
                        400)
                    return
                if not _source_is_acceptable(str(body["source"])):
                    self._send_json(
                        {"error": "source file does not exist: "
                         + str(body["source"])}, 400)
                    return
            params = {k: v for k, v in body.items() if k != "_"}
            params["kind"] = kind
            job_id = registry.create(params)
            if not pool.submit(
                    job_id,
                    lambda p=dict(params): run_generic_job(p)):
                registry.remove(job_id)
                self._send_json(
                    {"error": "server busy, retry later",
                     "queued_jobs": pool.stats()["queued_jobs"]}, 429)
                return
            self._send_json({"job_id": job_id, "state": "PENDING"}, 202)

        # ------------------------------------------------------- export
        def _export_job(self, job: dict) -> None:
            """Download a completed job result as CSV (learn) or JSON."""
            result = job["result"]
            fmt = "csv" if result.get("portfolio") is not None else "json"
            if fmt == "csv":
                import csv as _csv
                import io

                buf = io.StringIO()
                w = _csv.writer(buf)
                w.writerow(["rank", "operator", "inputs", "incremental_gain",
                            "fold_stability", "f4_passed", "f6_passed"])
                for i, m in enumerate(result.get("portfolio") or [], 1):
                    w.writerow([
                        i, m.get("operator", ""),
                        ";".join(m.get("inputs") or []),
                        m.get("incremental_gain", ""),
                        m.get("fold_stability", ""),
                        m.get("f4_passed", ""), m.get("f6_passed", ""),
                    ])
                body = buf.getvalue().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition",
                                 f"attachment; filename=fiae_{job['job_id']}.csv")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                body = json.dumps(result, indent=2, default=str).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Disposition",
                                 f"attachment; filename=fiae_{job['job_id']}.json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        # ------------------------------------------------------- GUI
        def _send_gui(self) -> None:
            from .webgui import gui_html
            body = gui_html(__version__).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # ------------------------------------------------------- dashboard
        def _send_dashboard(self) -> None:
            runs = self._list_runs()
            jobs = registry.list()
            pstats = pool.stats()
            jstats = registry.stats()
            rows = "".join(
                f"<tr><td><code>{r['run_id']}</code></td>"
                f"<td><span class='badge {r['state'].lower()}'>"
                f"{r['state']}</span></td>"
                f"<td>{r.get('params', {}).get('target', '-')}</td>"
                f"<td>{r.get('params', {}).get('max_rows', '-')}</td></tr>"
                for r in runs) or "<tr><td colspan=4>No runs yet</td></tr>"
            jrows = "".join(
                f"<tr><td><code>{j['job_id']}</code></td>"
                f"<td><span class='badge {j['state'].lower()}'>"
                f"{j['state']}</span></td>"
                f"<td>{j['params'].get('target', '-')}</td></tr>"
                for j in jobs) or "<tr><td colspan=3>No jobs yet</td></tr>"
            html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>FIAE Dashboard</title>
<style>
body{{font-family:Segoe UI,sans-serif;background:#0d1117;color:#e6edf3;
margin:0;padding:2rem}}
h1{{color:#c084fc}} h2{{color:#a78bfa;margin-top:2rem}}
table{{border-collapse:collapse;width:100%;margin-top:.5rem}}
th,td{{padding:.5rem .8rem;border-bottom:1px solid #30363d;text-align:left}}
th{{color:#8b5cf6;text-transform:uppercase;font-size:.75rem}}
.badge{{padding:.15rem .6rem;border-radius:10px;font-size:.75rem}}
.completed{{background:#1a4731;color:#7ee2a8}}
.running,.pending{{background:#3b2f63;color:#c4b5fd}}
.failed{{background:#4d1f24;color:#ffa1a1}}
footer{{margin-top:3rem;color:#6e7681;font-size:.8rem}}
.stats span{{display:inline-block;margin-right:1.5rem;color:#a78bfa}}
</style></head><body>
<h1>FIAE Dashboard</h1>
<p>Feature Intelligence &amp; Architecture Engine — v{__version__}</p>
<p class="stats">
<span>Workers: {pstats['workers']} ({pstats['busy_workers']} busy)</span>
<span>Queued: {pstats['queued_jobs']}</span>
<span>Jobs: {jstats.get('RUNNING', 0)} running ·
{jstats.get('PENDING', 0)} pending ·
{jstats.get('COMPLETED', 0)} completed ·
{jstats.get('FAILED', 0)} failed</span>
</p>
<h2>Runs</h2>
<table><tr><th>Run</th><th>State</th><th>Target</th><th>Max Rows</th></tr>
{rows}</table>
<h2>Jobs</h2>
<table><tr><th>Job</th><th>State</th><th>Target</th></tr>
{jrows}</table>
<footer>API: /api/health · /api/runs · /api/jobs · POST /api/learn ·
POST /api/profile · <a href="/gui" style="color:#a78bfa">Open GUI</a></footer>
</body></html>"""
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return FIAEHandler


class FIAEHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with a deep listen backlog.

    The stdlib default ``request_queue_size`` is 5, which drops TCP
    connections during bursts of simultaneous clients (hundreds of
    concurrent connects). A deep backlog lets the OS hold pending
    connections while request threads are created.
    """

    request_queue_size = 128
    daemon_threads = True
    allow_reuse_address = True


def serve(
    host: str = "127.0.0.1",
    port: int = 8420,
    runs_dir: str = "runs",
    *,
    max_workers: int = 4,
    max_queue_size: int = 100,
    rate_limit: float = 50.0,
) -> None:
    """Start the FIAE dashboard/API server (blocking).

    Parameters
    ----------
    max_workers : int
        Number of learn-job worker threads (bounded parallelism, doc 08).
    max_queue_size : int
        Pending-job queue depth. Full queue → HTTP 429 backpressure.
    rate_limit : float
        Max requests per second per client IP.
    """
    registry = JobRegistry()
    pool = JobWorkerPool(registry, max_workers=max_workers,
                         max_queue_size=max_queue_size)
    limiter = RateLimiter(max_per_second=rate_limit)
    pool.start()
    httpd = FIAEHTTPServer((host, port),
                           make_handler(registry, runs_dir,
                                        pool=pool, limiter=limiter))
    print(f"  FIAE server v{__version__} listening on http://{host}:{port}")
    print(f"  Serving runs from: {os.path.abspath(runs_dir)}")
    print(f"  Workers: {max_workers} · Queue: {max_queue_size} · "
          f"Rate limit: {rate_limit:.0f} req/s/client")
    print("  Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
    finally:
        httpd.server_close()
        pool.shutdown()
