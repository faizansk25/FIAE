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
                and node.func.id in ("eval", "exec", "compile"):
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


def run_in_sandbox(
    fn: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    policy: Optional[SandboxPolicy] = None,
) -> SandboxResult:
    """Execute a function in a sandboxed environment (doc 11).

    Applies:
    - Timeout via threading
    - Recursion depth limit
    - Execution time tracking

    Note: Full memory limiting requires OS-level controls (cgroups, ulimit).
    This provides best-effort sandboxing within Python's capabilities.
    """
    policy = policy or SandboxPolicy()
    kwargs = kwargs or {}
    result_holder: list[SandboxResult] = []

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
        return SandboxResult(
            success=False,
            error=f"Execution timed out after {policy.max_execution_time_s}s",
            execution_time_s=policy.max_execution_time_s,
            warnings=["timeout exceeded — possible infinite loop"],
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
        elif not isinstance(v, (int, bool)):
            warnings.append(f"{name}[{i}]: unexpected type {type(v).__name__}")
    return warnings


def validate_column_name(name: str) -> list[str]:
    """Validate a column name for safety (doc 11)."""
    warnings = []
    if not name:
        warnings.append("Empty column name")
    if len(name) > 256:
        warnings.append(f"Column name too long: {len(name)} chars")
    # Check for injection patterns
    dangerous = [";", "--", "/*", "*/", "UNION", "DROP", "DELETE"]
    for pattern in dangerous:
        if pattern.lower() in name.lower():
            warnings.append(f"Potentially dangerous pattern in column name: {pattern}")
    return warnings
