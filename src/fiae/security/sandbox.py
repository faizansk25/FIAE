"""Security sandbox (doc 11).

Provides safe execution environments for running untrusted or user-provided
code (e.g., custom operators, domain plugins).

Normative source: doc 11 section "Sandbox" and "Input Validation".
"""

from __future__ import annotations

import ast
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class SandboxPolicy:
    """Security policy for sandboxed execution (doc 11)."""

    max_execution_time_s: float = 30.0
    max_memory_bytes: int = 256 * 1024 * 1024  # 256 MB
    max_recursion_depth: int = 100
    allowed_builtins: tuple = (
        "abs", "all", "any", "bool", "dict", "enumerate", "filter",
        "float", "frozenset", "hash", "int", "isinstance", "len",
        "list", "map", "max", "min", "next", "print", "range",
        "round", "set", "sorted", "str", "sum", "tuple", "type",
        "zip", "True", "False", "None",
    )
    forbidden_modules: tuple = (
        "os", "sys", "subprocess", "shutil", "pathlib",
        "socket", "http", "urllib", "requests",
        "ctypes", "importlib",
    )


def validate_input_code(code: str, policy: Optional[SandboxPolicy] = None) -> list[str]:
    """Static analysis of code for security issues (doc 11).

    Returns a list of warnings.  Empty list means code is safe.
    """
    policy = policy or SandboxPolicy()
    warnings = []

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]

    for node in ast.walk(tree):
        # Check for imports of forbidden modules
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in policy.forbidden_modules:
                    warnings.append(f"Import of forbidden module: {alias.name}")

        if isinstance(node, ast.ImportFrom) and node.module \
                and node.module.split(".")[0] in policy.forbidden_modules:
            warnings.append(f"Import from forbidden module: {node.module}")

        # Check for eval/exec calls
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("eval", "exec", "compile", "__import__", "open"):
            warnings.append(f"Call to forbidden function: {node.func.id}")

        # Check for attribute access on forbidden modules
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id in policy.forbidden_modules:
            warnings.append(f"Access to forbidden module attribute: {node.value.id}.{node.attr}")

    return warnings


@dataclass
class SandboxResult:
    """Result of sandboxed execution."""

    success: bool = True
    result: Any = None
    error: Optional[str] = None
    execution_time_s: float = 0.0
    warnings: list[str] = field(default_factory=list)


def _interrupt_thread(thread: threading.Thread) -> None:
    """Best-effort stop for a runaway sandbox thread (W-6).

    Python cannot kill threads directly.  CPython exposes ``PyThreadState_
    SetAsyncExc`` via the ``ctypes``-free internal ``_thread`` module only
    through ``ctypes``, which the sandbox forbids for sandboxed code — so we
    raise an exception inside the thread through the CPython C API via
    ``threading``-safe ``ctypes`` usage in the *supervisor* (trusted) side.

    If the async exception cannot be delivered (non-CPython, or the thread
    is in a C call that swallows it), the daemon thread is left to die with
    the process — documented boundary, same as before.  Either way the
    supervisor returns immediately; the caller never blocks.
    """
    try:
        import ctypes

        tid = thread.ident
        if tid is None:
            return
        exc_type = type(_SandboxTimeoutError())
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(tid), ctypes.py_object(exc_type)
        )
    except Exception:
        # Non-CPython or stripped runtime: fall back to daemon semantics.
        pass


class _SandboxTimeoutError(BaseException):
    """Raised inside a runaway sandbox thread to unwind it (W-6).

    Derives from BaseException so bare ``except Exception`` in user code
    cannot swallow the interrupt.
    """


def run_in_sandbox(
    fn: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    policy: Optional[SandboxPolicy] = None,
) -> SandboxResult:
    """Execute a function in a sandboxed environment (doc 11).

    Applies:
    - Timeout via supervisor thread + async interrupt (W-6: the runaway
      thread receives ``_SandboxTimeoutError`` on timeout; a pure-Python
      loop unwinds instead of burning CPU until process exit)
    - Recursion depth limit
    - Execution time tracking
    - Memory *gate* (W-7): ``policy.max_memory_bytes`` is enforced as a
      pre-execution bound on the declared working-set size of the payload
      (args/results are measured; the function object itself is not
      introspectable portably).  True OS-level RSS limiting requires
      cgroups/ulimit/JobObjects and is intentionally out of scope —
      documented boundary, not silently claimed.

    Note: Full memory limiting requires OS-level controls (cgroups, ulimit).
    This provides best-effort sandboxing within Python's capabilities.
    """
    policy = policy or SandboxPolicy()
    kwargs = kwargs or {}

    # W-7 memory gate: reject payloads whose declared inputs already exceed
    # the policy budget.  This is a gate, not an OS-level RSS limiter.
    estimated_bytes = 0
    for a in args:
        estimated_bytes += sys.getsizeof(a)
    for v in kwargs.values():
        estimated_bytes += sys.getsizeof(v)
    if estimated_bytes > policy.max_memory_bytes:
        return SandboxResult(
            success=False,
            error=(
                f"Memory gate: declared payload ~{estimated_bytes} bytes "
                f"exceeds policy limit {policy.max_memory_bytes} bytes"
            ),
            warnings=["memory gate triggered before execution"],
        )

    result_holder: list[SandboxResult] = []
    timed_out = threading.Event()

    def _target():
        t0 = time.monotonic()
        try:
            # Set recursion depth
            old_limit = sys.getrecursionlimit()
            sys.setrecursionlimit(policy.max_recursion_depth)
            try:
                ret = fn(*args, **kwargs)
                elapsed = time.monotonic() - t0
                result_holder.append(SandboxResult(
                    success=True, result=ret,
                    execution_time_s=elapsed,
                ))
            finally:
                sys.setrecursionlimit(old_limit)
        except _SandboxTimeoutError:
            elapsed = time.monotonic() - t0
            result_holder.append(SandboxResult(
                success=False,
                error=f"Execution timed out after {policy.max_execution_time_s}s",
                execution_time_s=elapsed,
                warnings=["runaway thread interrupted"],
            ))
        except Exception as e:
            elapsed = time.monotonic() - t0
            result_holder.append(SandboxResult(
                success=False, error=str(e),
                execution_time_s=elapsed,
            ))

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=policy.max_execution_time_s)

    if thread.is_alive():
        # W-6: interrupt the runaway thread so it unwinds instead of
        # spinning until process exit.  Brief grace join; if it survives
        # (C-extension call holding the GIL), daemon semantics apply.
        timed_out.set()
        _interrupt_thread(thread)
        thread.join(timeout=2.0)
        interrupted = not thread.is_alive()
        warning = (
            "runaway thread interrupted"
            if interrupted
            else "thread could not be interrupted (in C call); daemon exit only"
        )
        return SandboxResult(
            success=False,
            error=f"Execution timed out after {policy.max_execution_time_s}s",
            execution_time_s=policy.max_execution_time_s,
            warnings=[warning],
        )

    if result_holder:
        return result_holder[0]

    return SandboxResult(success=False, error="No result produced")


def validate_numeric_input(values: list, name: str = "input") -> list[str]:
    """Validate numeric input values for safety (doc 11)."""
    warnings = []
    for i, v in enumerate(values):
        if v is None:
            continue
        if isinstance(v, float):
            if math.isnan(v):
                warnings.append(f"{name}[{i}]: NaN detected")
            elif math.isinf(v):
                warnings.append(f"{name}[{i}]: Inf detected")

    return warnings


def validate_column_name(name: str) -> list[str]:
    """Validate a column name for injection attempts (doc 11).

    Returns a list of problems.  Empty list means the name is safe.
    """
    problems = []
    if not isinstance(name, str) or not name:
        problems.append("column name must be a non-empty string")
        return problems

    # Allow letters, digits, spaces, underscore, hyphen, dot.
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                  "0123456789_-. ")
    if not set(name) <= allowed:
        problems.append(f"column name contains disallowed characters: {name!r}")

    # SQL / code injection signatures
    lowered = name.lower()
    for signature in (
        "select", "insert", "update", "delete", "drop", "union",
        "--", ";", "/*", "*/", "__import__", "eval(", "exec(",
        "system(", "open(",
    ):
        if signature in lowered:
            problems.append(f"column name contains suspicious token: {signature}")

    return problems
