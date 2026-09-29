"""Regression tests for W-6 (runaway thread interrupt) and W-7 (memory gate).

Before M31: a sandboxed infinite loop survived its timeout as a live daemon
thread burning CPU until process exit, and ``SandboxPolicy.max_memory_bytes``
was declared but never enforced.  These tests pin the fixed behavior.
"""

import threading
import time


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
        before = threading.active_count()

        result = run_in_sandbox(_spin_forever, policy=policy)

        assert not result.success
        assert "timed out" in (result.error or "")
        # Give the async interrupt a moment to land, then verify.
        deadline = time.monotonic() + 3.0
        while threading.active_count() > before and time.monotonic() < deadline:
            time.sleep(0.05)
        assert threading.active_count() <= before, (
            "runaway sandbox thread is still alive after timeout — W-6 regressed"
        )

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
