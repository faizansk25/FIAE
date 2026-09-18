"""Concurrency and warehouse adapter tests.

Covers:
- Warehouse adapters (BigQuery/Snowflake/Redshift): URI parsing, factory
  routing, thread-local connections, streaming scans with mocked drivers.
- SqlAdapter thread-safety: concurrent scans on one adapter instance must
  use independent connections per thread.
- Server concurrency: bounded JobWorkerPool, backpressure (429), rate
  limiting, registry pruning — verified under simulated multi-client load.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import urllib.error
import urllib.request

import pytest

from fiae.intake import auto_adapter
from fiae.intake.adapter_warehouse import (
    BigQueryAdapter,
    RedshiftAdapter,
    SnowflakeAdapter,
)
from fiae.server import (
    JobRegistry,
    JobWorkerPool,
    RateLimiter,
    make_handler,
    run_learn_job,
)


# ---------------------------------------------------------------------------
# Warehouse adapter construction + factory routing
# ---------------------------------------------------------------------------


class TestWarehouseConstruction:
    def test_bigquery_basic(self):
        a = BigQueryAdapter("bigquery://my-project/my_dataset", table="events")
        assert a._engine == "bigquery"
        assert a._project == "my-project"
        assert a.source_id() == "bigquery|bigquery://my-project/my_dataset|events"

    def test_bigquery_requires_table_or_query(self):
        with pytest.raises(ValueError):
            BigQueryAdapter("bigquery://proj/ds")

    def test_snowflake_uri_parsing(self):
        a = SnowflakeAdapter(
            "snowflake://user:pass@acct/mydb/public", table="orders")
        user, password, database, schema = a._parse_uri()
        assert user == "user"
        assert password == "pass"
        assert database == "mydb"
        assert schema == "public"
        assert a._parse_account() == "acct"

    def test_redshift_uri(self):
        a = RedshiftAdapter(
            "redshift://u:p@cluster.abc.us-east-1.redshift.amazonaws.com:5439/dev",
            table="sales")
        assert a._engine == "redshift"

    def test_factory_routes_bigquery(self):
        adapter = auto_adapter("bigquery://proj/ds", table="t")
        assert type(adapter) is BigQueryAdapter

    def test_factory_routes_snowflake(self):
        adapter = auto_adapter("snowflake://u:p@acct/db/s", table="t")
        assert type(adapter) is SnowflakeAdapter

    def test_factory_routes_redshift(self):
        adapter = auto_adapter("redshift://u:p@host:5439/db", table="t")
        assert type(adapter) is RedshiftAdapter

    def test_fingerprint_is_deterministic(self):
        a1 = BigQueryAdapter("bigquery://p/d", table="t")
        a2 = BigQueryAdapter("bigquery://p/d", table="t")
        assert a1.fingerprint_material() == a2.fingerprint_material()

    def test_fingerprint_differs_by_table(self):
        a1 = BigQueryAdapter("bigquery://p/d", table="t1")
        a2 = BigQueryAdapter("bigquery://p/d", table="t2")
        assert a1.fingerprint_material() != a2.fingerprint_material()


# ---------------------------------------------------------------------------
# Warehouse scan with a mocked driver (exercises the shared scan machinery)
# ---------------------------------------------------------------------------


class FakeCursor:
    """DB-API cursor returning a fixed row set in fetchmany batches.

    Responds to ``SELECT COUNT(*)`` queries with the configured row count,
    mirroring how real warehouse engines answer COUNT.
    """

    def __init__(self, rows, columns):
        self._rows = list(rows)
        self._columns = columns
        self.description = [(c,) for c in columns]
        self._is_count = False

    def execute(self, sql):
        assert "SELECT" in sql.upper()
        self._is_count = "COUNT(*)" in sql.upper()

    def fetchone(self):
        if self._is_count:
            return (len(self._rows),)
        return self._rows[0] if self._rows else None

    def fetchmany(self, size):
        if self._is_count:
            out = [(len(self._rows),)]
            self._is_count = False
            return out
        out, self._rows = self._rows[:size], self._rows[size:]
        return out

    def close(self):
        pass


class FakeConnection:
    def __init__(self, rows, columns):
        self._rows = rows
        self._columns = columns
        self.thread_id = threading.get_ident()

    def cursor(self):
        return FakeCursor(self._rows, self._columns)

    def close(self):
        pass


def _sample_rows(n=10):
    columns = ["id", "amount", "region"]
    rows = [(i, float(i * 10), f"r{i % 3}") for i in range(n)]
    return rows, columns


class TestWarehouseScan:
    def _make_bq(self, rows, columns):
        a = BigQueryAdapter("bigquery://p/d", table="t")
        # Patch _connect to return the fake connection (per-thread local)
        def fake_connect():
            return FakeConnection(rows, columns)
        a._connect = fake_connect  # type: ignore[method-assign]
        return a

    def test_scan_streams_in_batches(self):
        rows, columns = _sample_rows(10)
        a = self._make_bq(rows, columns)
        batches = list(a.scan())
        total = sum(len(b.columns["id"]) for b in batches)
        assert total == 10
        # Default batch_rows=2048 → single batch for 10 rows
        assert len(batches) == 1

    def test_scan_small_batches(self):
        rows, columns = _sample_rows(7)
        a = BigQueryAdapter("bigquery://p/d", table="t", batch_rows=3)
        a._connect = lambda: FakeConnection(rows, columns)
        batches = list(a.scan())
        total = sum(len(b.columns["id"]) for b in batches)
        assert total == 7
        assert len(batches) == 3  # ceil(7/3)

    def test_projection_filters_columns(self):
        rows, columns = _sample_rows(5)
        a = self._make_bq(rows, columns)
        batches = list(a.scan(projection=["id", "amount"]))
        assert set(batches[0].columns.keys()) == {"id", "amount"}

    def test_estimate_rows_uses_count(self):
        rows, columns = _sample_rows(4)
        a = self._make_bq(rows, columns)
        assert a.estimate_rows() == 4

    def test_thread_local_connections_are_independent(self):
        """Each thread gets its own FakeConnection instance."""
        rows, columns = _sample_rows(2)
        made: list[FakeConnection] = []

        def fake_connect():
            conn = FakeConnection(rows, columns)
            made.append(conn)
            return conn

        a = BigQueryAdapter("bigquery://p/d", table="t")
        a._connect = fake_connect  # type: ignore[method-assign]

        got = []

        def worker():
            got.append(a._get_connection())

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(made) == 4  # one connection per thread
        assert len({id(c) for c in got}) == 4  # all distinct

    def test_concurrent_scans_on_one_adapter(self):
        """4 threads scanning simultaneously all see complete data."""
        rows, columns = _sample_rows(6)
        a = BigQueryAdapter("bigquery://p/d", table="t", batch_rows=2)
        a._connect = lambda: FakeConnection(rows, columns)
        results = []
        errors = []

        def worker():
            try:
                total = sum(
                    len(b.columns["id"]) for b in a.scan())
                results.append(total)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert results == [6, 6, 6, 6]


# ---------------------------------------------------------------------------
# SqlAdapter thread-safety
# ---------------------------------------------------------------------------


class TestSqlAdapterThreadSafety:
    def test_concurrent_scans_sqlite(self, tmp_path):
        from fiae.intake import SqlAdapter

        db_path = tmp_path / "t.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE t (a INTEGER, b TEXT)")
        rows = [(i, f"v{i}") for i in range(50)]
        conn.executemany("INSERT INTO t VALUES (?, ?)", rows)
        conn.commit()
        conn.close()

        adapter = SqlAdapter(f"sqlite:///{db_path}", table="t", batch_rows=5)
        results = []
        errors = []

        def worker():
            try:
                total_a = 0
                total_b = 0
                for batch in adapter.scan():
                    total_a += len(batch.columns["a"])
                    total_b += len(batch.columns["b"])
                results.append((total_a, total_b))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert all(r == (50, 50) for r in results)

    def test_thread_local_conn_not_shared(self, tmp_path):
        from fiae.intake import SqlAdapter

        db_path = tmp_path / "t.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE t (a INTEGER)")
        conn.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(5)])
        conn.commit()
        conn.close()

        adapter = SqlAdapter(f"sqlite:///{db_path}", table="t")
        conns = []

        def worker():
            conns.append(adapter._get_connection())

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len({id(c) for c in conns}) == 3


# ---------------------------------------------------------------------------
# Server: bounded worker pool + backpressure
# ---------------------------------------------------------------------------


class TestJobWorkerPool:
    def test_sequential_execution_all_complete(self):
        registry = JobRegistry()
        pool = JobWorkerPool(registry, max_workers=2, max_queue_size=10)
        pool.start()
        ids = []
        for i in range(5):
            jid = registry.create({"n": i})
            assert pool.submit(jid, lambda n=i: {"n": n})
            ids.append(jid)
        deadline = time.time() + 5
        while time.time() < deadline:
            if all(registry.get(j)["state"] == "COMPLETED" for j in ids):
                break
            time.sleep(0.02)
        assert all(registry.get(j)["state"] == "COMPLETED" for j in ids)
        assert registry.get(ids[0])["result"] == {"n": 0}
        pool.shutdown()

    def test_queue_full_returns_backpressure(self):
        registry = JobRegistry()
        pool = JobWorkerPool(registry, max_workers=1, max_queue_size=1)
        pool.start()

        release = threading.Event()

        def slow():
            release.wait(timeout=5)
            return {}

        j1 = registry.create({})
        assert pool.submit(j1, slow)  # occupies the single worker
        time.sleep(0.1)  # let worker pick it up
        j2 = registry.create({})
        assert pool.submit(j2, lambda: {})  # fills the queue (size 1)
        j3 = registry.create({})
        assert not pool.submit(j3, lambda: {})  # queue full → backpressure
        registry.remove(j3)
        release.set()
        pool.shutdown()

    def test_worker_failure_marks_job_failed(self):
        registry = JobRegistry()
        pool = JobWorkerPool(registry, max_workers=1, max_queue_size=5)

        def boom():
            raise RuntimeError("kaput")

        pool.start()
        jid = registry.create({})
        pool.submit(jid, boom)
        deadline = time.time() + 5
        while time.time() < deadline:
            job = registry.get(jid)
            if job["state"] == "FAILED":
                break
            time.sleep(0.02)
        job = registry.get(jid)
        assert job["state"] == "FAILED"
        assert "kaput" in job["error"]
        pool.shutdown()

    def test_stats_reflect_pool_state(self):
        registry = JobRegistry()
        pool = JobWorkerPool(registry, max_workers=3, max_queue_size=10)
        assert pool.stats() == {"workers": 3, "busy_workers": 0,
                                "queued_jobs": 0}
        pool.start()
        pool.shutdown()


class TestRateLimiter:
    def test_allows_under_limit(self):
        rl = RateLimiter(max_per_second=5)
        assert all(rl.allow("c1") for _ in range(5))

    def test_blocks_over_limit(self):
        rl = RateLimiter(max_per_second=5)
        for _ in range(5):
            assert rl.allow("c1")
        assert not rl.allow("c1")

    def test_window_slides(self):
        rl = RateLimiter(max_per_second=2)
        assert rl.allow("c1")
        assert rl.allow("c1")
        assert not rl.allow("c1")
        time.sleep(1.05)
        assert rl.allow("c1")

    def test_clients_independent(self):
        rl = RateLimiter(max_per_second=1)
        assert rl.allow("a")
        assert not rl.allow("a")
        assert rl.allow("b")

    def test_client_table_pruned(self):
        rl = RateLimiter(max_per_second=1, max_clients=3)
        for i in range(10):
            rl.allow(f"client_{i}")
        assert len(rl._hits) <= 3 + 1  # prune + current


class TestRegistryPruning:
    def test_completed_jobs_pruned_beyond_limit(self):
        registry = JobRegistry(max_completed=5)
        for i in range(10):
            jid = registry.create({"i": i})
            registry.complete(jid, {"i": i})
        stats = registry.stats()
        assert stats.get("COMPLETED", 0) == 5

    def test_running_jobs_never_pruned(self):
        registry = JobRegistry(max_completed=2)
        running = []
        for i in range(5):
            jid = registry.create({"i": i})
            if i < 2:
                registry.start(jid)
                running.append(jid)
            else:
                registry.complete(jid, {})
        assert all(registry.get(j)["state"] == "RUNNING" for j in running)
        assert registry.stats().get("COMPLETED", 0) == 2


# ---------------------------------------------------------------------------
# Server HTTP: end-to-end concurrency under load
# ---------------------------------------------------------------------------


@pytest.fixture()
def loaded_server(tmp_path):
    """Server with a tiny pool so backpressure is reachable in tests."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    registry = JobRegistry(max_completed=10)
    pool = JobWorkerPool(registry, max_workers=1, max_queue_size=2)
    limiter = RateLimiter(max_per_second=10_000)  # effectively off
    pool.start()
    from http.server import ThreadingHTTPServer
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(registry, str(runs_dir), pool=pool, limiter=limiter))
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}", registry, pool
    httpd.shutdown()
    pool.shutdown()


def _post_json(base, path, payload):
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def _get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


class TestServerUnderLoad:
    def test_health_includes_pool_stats(self, loaded_server):
        base, _, _ = loaded_server
        status, body = _get(base, "/api/health")
        assert status == 200
        assert body["pool"]["workers"] == 1
        assert "jobs" in body

    def test_burst_of_posts_gets_202_or_429_never_crash(self, loaded_server):
        """30 simultaneous submissions against queue depth 2 → mix of
        202 (accepted) and 429 (backpressure), never a 5xx."""
        base, _registry, _ = loaded_server
        csv_path = str(_make_csv())
        codes = []
        threads = []

        lock = threading.Lock()

        def post():
            code, _ = _post_json(base, "/api/learn", {
                "source": csv_path, "target": "y"})
            with lock:
                codes.append(code)

        for _ in range(30):
            t = threading.Thread(target=post)
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(c in (202, 429) for c in codes)
        assert 202 in codes and 429 in codes  # both paths exercised

    def test_rate_limiter_returns_429(self, tmp_path):
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        registry = JobRegistry()
        pool = JobWorkerPool(registry, max_workers=1, max_queue_size=5)
        pool.start()
        limiter = RateLimiter(max_per_second=2)
        from http.server import ThreadingHTTPServer
        httpd = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(registry, str(runs_dir), pool=pool, limiter=limiter))
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        base = f"http://127.0.0.1:{port}"
        codes = [_get(base, "/api/health")[0] for _ in range(5)]
        assert 429 in codes
        httpd.shutdown()
        pool.shutdown()

    def test_dashboard_shows_pool_stats(self, loaded_server):
        base, _, _ = loaded_server
        with urllib.request.urlopen(base + "/dashboard", timeout=5) as resp:
            html = resp.read().decode()
        assert "Workers:" in html
        assert "Queued:" in html


def _make_csv(n=30):
    import csv
    import os
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x1", "x2", "y"])
        for i in range(n):
            w.writerow([i, i % 3, float(i % 2)])
    return path


class TestRunLearnJob:
    def test_end_to_end_job(self):
        csv_path = _make_csv()
        result = run_learn_job({
            "source": csv_path, "target": "y",
            "max_rows": 1000, "max_features": 3, "seed": 0,
        })
        assert result["target"] == "y"
        assert "portfolio_size" in result
