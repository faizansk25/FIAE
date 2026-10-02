"""M38 regression tests: external-audit correctness quick-batch.

Each test targets one verified finding; each would fail against the
pre-M38 code.
"""

import threading
import urllib.request
from http.server import ThreadingHTTPServer

from fiae.model_registry import ROUTING_TABLE, route_model
from fiae.server import JobRegistry, make_handler


class TestRegistryPrunesFailedJobs:
    def test_failed_jobs_do_not_accumulate(self):
        reg = JobRegistry(max_completed=2)
        for _ in range(4):
            jid = reg.create({"source": "a.csv", "target": "y"})
            reg.start(jid)
            reg.fail(jid, "boom")
        # Pre-M38: FAILED jobs were never pruned -> 4 entries retained.
        assert len(reg.list()) <= 2, "FAILED jobs leak the registry"


class TestRoutingDoesNotMutateGlobals:
    def test_repeated_routing_keeps_priorities_stable(self):
        before = [s.priority for s in ROUTING_TABLE]
        for _ in range(3):
            route_model("binary classification", n_rows=100,
                        n_features=5, has_categoricals=True)
        after = [s.priority for s in ROUTING_TABLE]
        assert before == after, "route_model mutated the global routing table"

    def test_routing_is_repeatable(self):
        a = [(s.family, s.priority) for s in
             route_model("binary classification", n_rows=100, n_features=5,
                         has_categoricals=True)]
        b = [(s.family, s.priority) for s in
             route_model("binary classification", n_rows=100, n_features=5,
                         has_categoricals=True)]
        assert a == b, "routing results drift between identical calls"


class TestDashboardEscapesHtml:
    def test_malicious_target_is_escaped(self, tmp_path):
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        # Seed a job whose params carry an XSS payload, exactly as API
        # input would (pre-M38 this rendered as live markup).
        reg = JobRegistry()
        jid = reg.create({"source": "a.csv",
                          "target": "<script>alert(1)</script>"})
        reg.start(jid)
        reg.complete(jid, {"portfolio_size": 1})

        handler = make_handler(reg, str(runs_dir))
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/dashboard", timeout=5) as r:
                body = r.read().decode()
            assert "<script>alert(1)</script>" not in body
            assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body
        finally:
            httpd.shutdown()
