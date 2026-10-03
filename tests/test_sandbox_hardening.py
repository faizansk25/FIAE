"""Regression tests for W-6 (runaway thread interrupt) and W-7 (memory gate).

Before M31: a sandboxed infinite loop survived its timeout as a live daemon
thread burning CPU until process exit, and ``SandboxPolicy.max_memory_bytes``
was declared but never enforced.  These tests pin the fixed behavior.
"""

import threading


from fiae.security.sandbox import SandboxPolicy, run_in_sandbox


def _spin_forever():
    x = 0
    while True:
        x += 1


class TestW6RunawayThreadInterrupt:
    def test_timeout_thread_is_interrupted(self):
        """After timeout the runaway pure-Python thread must unwind, leaving
        no live spinning thread behind."""
        policy = SandboxPolicy(max_execution_time_s=0.5)
        marker = threading.Event()

        def spin_until_interrupted():
            x = 0
            while not marker.is_set():
                x += 1

        result = run_in_sandbox(spin_until_interrupted, policy=policy)

        assert not result.success
        assert "timed out" in (result.error or "")
        # M38.5: assert the sandbox reaped *its own* thread.  The previous
        # assertion was ``threading.active_count() < 2`` — a process-global
        # proxy that any unrelated live thread (a JobWorkerPool worker from
        # an earlier test) breaks, which turned this into a whole-suite CI
        # failure on every matrix job.
        try:
            assert result.thread_reaped, (
                "runaway sandbox thread survived the timeout — W-6 regressed"
            )
        finally:
            marker.set()

    def test_interrupt_result_reports_warning(self):
        policy = SandboxPolicy(max_execution_time_s=0.5)
        result = run_in_sandbox(_spin_forever, policy=policy)
        assert any("interrupt" in w for w in result.warnings), result.warnings

    def test_normal_execution_unaffected(self):
        result = run_in_sandbox(lambda: 2 + 3, policy=SandboxPolicy(max_execution_time_s=5))
        assert result.success
        assert result.result == 5


class TestW7MemoryGate:
    def test_oversized_payload_rejected_before_execution(self):
        """A declared payload above the policy budget must be rejected
        without ever running the function."""
        policy = SandboxPolicy(max_memory_bytes=1024)
        big = bytearray(4096)
        ran = []

        def payload(b):
            ran.append(True)
            return len(b)

        result = run_in_sandbox(payload, args=(big,), policy=policy)

        assert not result.success
        assert "Memory gate" in (result.error or "")
        assert ran == [], "function must not execute when gate trips"

    def test_within_budget_executes(self):
        policy = SandboxPolicy(max_memory_bytes=1024 * 1024)
        result = run_in_sandbox(lambda b: len(b), args=(bytearray(100),), policy=policy)
        assert result.success
        assert result.result == 100

    def test_default_budget_does_not_block_normal_args(self):
        result = run_in_sandbox(lambda a, b: a + b, args=(1, 2))
        assert result.success
        assert result.result == 3
