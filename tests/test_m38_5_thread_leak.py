"""M38.5 regression tests: the CI-red thread leak.

GitHub CI run 37008735186 (commit d056a9f) failed all 9 matrix jobs plus the
coverage job.  Root cause, reproduced from a clean checkout:

``tests/test_m38_correctness.py`` called ``make_handler(registry, runs_dir)``
with no pool, so ``make_handler`` created a ``JobWorkerPool`` and started it.
That pool was unreachable from the caller — 4 daemon worker threads with no
stop path, alive for the rest of the process.  ``test_sandbox_hardening``'s
W-6 test then asserted ``threading.active_count() < 2`` (a process-global
proxy for "the sandbox reaped its own runaway thread"), which those idle
worker threads made false from that point on.

Both halves are pinned here: the pool is stoppable, and the W-6 invariant
survives unrelated live threads.
"""

import threading

from fiae.security.sandbox import SandboxPolicy, run_in_sandbox
from fiae.server import JobRegistry, JobWorkerPool, make_handler


def _spin_until_interrupted(stop: threading.Event) -> None:
    x = 0
    while not stop.is_set():
        x += 1


class TestAutoCreatedPoolIsStoppable:
    def test_implicit_pool_is_published_and_stoppable(self, tmp_path):
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        handler = make_handler(JobRegistry(), str(runs_dir))

        # Pre-M38.5: ``handler.job_pool`` did not exist at all, so the
        # 4 threads started here could never be stopped.
        pool = getattr(handler, "job_pool", None)
        assert pool is not None, "make_handler must publish its job pool"

        workers = list(pool._workers)
        assert len(workers) == 4
        assert all(t.is_alive() for t in workers)

        pool.shutdown(wait=True)
        assert not any(t.is_alive() for t in workers), (
            "job worker threads outlived pool.shutdown()"
        )


class TestW6InvariantSurvivesBackgroundThreads:
    def test_runaway_thread_is_reaped_with_unrelated_threads_live(self):
        """The CI condition: unrelated live threads must not break W-6."""
        pool = JobWorkerPool(JobRegistry(), max_workers=2)
        pool.start()
        stop = threading.Event()
        try:
            assert any(t.is_alive() for t in pool._workers)
            before = {t.ident for t in threading.enumerate()}

            result = run_in_sandbox(
                lambda: _spin_until_interrupted(stop),
                policy=SandboxPolicy(max_execution_time_s=0.5),
            )

            assert not result.success
            assert "timed out" in (result.error or "")
            # Pre-M38.5 this was expressed as ``active_count() < 2``, which
            # the 2 pool workers above made false 100% of the time.
            assert result.thread_reaped, result.warnings
            leaked = {t.ident for t in threading.enumerate() if t.is_alive()}
            assert leaked <= before, "sandbox left a thread behind"
        finally:
            stop.set()
            pool.shutdown(wait=True)
