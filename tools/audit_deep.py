"""Deep static audit: call signatures, dead code, duplicate defs, version compat.

Not part of CI. This is the investigative tool used to surface defect classes
ruff/pyflakes do not cover:

A. Wrong function calls  - resolve every intra-project call against the real
   definition and compare positional count, keyword names and required args.
B. Dead definitions      - module-level functions/classes never referenced.
C. Duplicate definitions - same name defined twice in one scope.
D. Version compat        - syntax newer than the declared requires-python.

Run:  python tools/audit_deep.py [scan-root]

The scan root defaults to the current working directory. It is a parameter
rather than a hardcoded parent lookup because a fixed ``parents[1]`` made the
tool walk an entire drive whenever it was copied elsewhere.
"""

from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()
SRC = ROOT / "src" / "fiae"
REQUIRES_PY = (3, 10)

# ---------------------------------------------------------------- collection


# Source trees worth auditing. Anything else (``.fiae/runs`` artifacts,
# dist/, build/, virtualenvs) is generated output, where a ``dict.get()`` call
# is not a bug and must not be reported as one.
SCAN_DIRS = ("src", "tests", "tools", "examples", "benchmarks")
SCAN_FILES = ("conftest.py", "setup.py")

# Attribute names that are so widely used as builtin container/str/file
# methods that a project-local definition of the same name would produce
# thousands of false positives (``d.get(k, default)`` matching some unrelated
# ``get()`` method). Resolution here is name-based, so these are excluded.
BUILTIN_ATTRS = frozenset({
    "get", "keys", "values", "items", "pop", "popitem", "update", "setdefault",
    "clear", "copy", "add", "append", "extend", "insert", "remove", "discard",
    "index", "count", "sort", "reverse", "read", "readlines", "write",
    "writelines", "close", "flush", "seek", "tell", "open", "join", "split",
    "rsplit", "strip", "lstrip", "rstrip", "startswith", "endswith",
    "replace", "format", "encode", "decode", "upper", "lower", "title",
    "find", "add_note", "with_traceback", "bit_length", "conjugate",
})


def py_files() -> list[Path]:
    found: list[Path] = []
    for name in SCAN_FILES:
        p = ROOT / name
        if p.is_file():
            found.append(p)
    for d in SCAN_DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        found.extend(sorted(
            p for p in base.rglob("*.py")
            if "__pycache__" not in p.parts and ".venv" not in p.parts
        ))
    return sorted(set(found))


def load() -> dict[str, ast.Module]:
    trees: dict[str, ast.Module] = {}
    for path in py_files():
        try:
            text = path.read_text("utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            print(f"SKIP (unreadable) {path}: {exc}")
            continue
        try:
            trees[str(path.relative_to(ROOT))] = ast.parse(text)
        except SyntaxError as exc:
            print(f"SYNTAX ERROR {path}: {exc}")
    return trees


# ---------------------------------------------------------- A. call signatures


class DefInfo:
    """A callable's signature, harvested from its AST."""

    def __init__(self, name: str, node: ast.FunctionDef | ast.AsyncFunctionDef, path: str):
        self.name = name
        self.node = node
        self.path = path
        a = node.args
        self.posonly = [p.arg for p in a.posonlyargs]
        self.args = [p.arg for p in a.args]
        self.kwonly = [p.arg for p in a.kwonlyargs]
        self.vararg = a.vararg.arg if a.vararg else None
        self.kwarg = a.kwarg.arg if a.kwarg else None
        self.decorators = {ast.unparse(d) for d in node.decorator_list}

    @property
    def accepted_positional(self) -> list[str]:
        return self.posonly + self.args

    def required(self) -> list[str]:
        """Argument names that must be supplied by the caller.

        Keyword-only args are only required when they have no default; the
        naive version counted every kwonly name and flagged ~141 valid calls
        such as ``content_hash(obj)`` whose ``*, exclude_keys=None`` is
        optional.
        """
        req = list(self.accepted_positional)
        n_defaults = len(self.node.args.defaults)
        if n_defaults:
            req = req[: len(req) - n_defaults]
        a = self.node.args
        required_kwonly = [
            arg.arg for arg, default in zip(a.kwonlyargs, a.kw_defaults)
            if default is None
        ]
        return req + required_kwonly


def collect_defs(trees: dict[str, ast.Module]) -> dict[str, list[DefInfo]]:
    """Map bare name -> definitions, for functions defined at module/class level."""
    out: dict[str, list[DefInfo]] = defaultdict(list)
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("__") and node.name.endswith("__"):
                    continue  # dunders are framework-called
                out[node.name].append(DefInfo(node.name, node, path))
    return out


def collect_methods(trees: dict[str, ast.Module]) -> dict[str, list[DefInfo]]:
    """Map method name -> definitions declared inside a class body.

    Kept separate from bare functions: a method call ``obj.m()`` is a bound
    call (no implicit ``self``), whereas a bare ``m()`` inside a class body
    is unbound. Conflating them produced wrong verdicts, so the two are
    checked against different definitions.
    """
    out: dict[str, list[DefInfo]] = defaultdict(list)
    for path, tree in trees.items():
        for cls in ast.walk(tree):
            if not isinstance(cls, ast.ClassDef):
                continue
            for node in cls.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("__") and node.name.endswith("__"):
                        continue
                    out[node.name].append(DefInfo(node.name, node, path))
    return out


def check_calls(trees: dict[str, ast.Module], defs: dict[str, list[DefInfo]],
                methods: dict[str, list[DefInfo]]) -> list[str]:
    """Flag intra-project calls whose shape cannot match any real definition."""
    problems: list[str] = []
    for path, tree in trees.items():
        # names bound by `from .x import y` / `import x.y` within this module
        local_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    local_names.add(alias.asname or alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    local_names.add(alias.asname or alias.name.split(".")[0])

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Name):
                name = fn.id
            elif isinstance(fn, ast.Attribute):
                name = fn.attr
            else:
                continue
            bound = isinstance(fn, ast.Attribute)
            if name in BUILTIN_ATTRS:
                continue  # name-based resolution cannot disambiguate these
            if bound:
                # obj.m() / self.m() -> look among class methods only
                candidates = methods.get(name, [])
                is_bound_call = True
            else:
                if name not in local_names:
                    continue  # not an intra-project symbol we can resolve
                candidates = defs.get(name, [])
                is_bound_call = False
            if not candidates:
                continue

            ok = False
            for d in candidates:
                n_pos = len(node.args)
                kwnames = {k.arg for k in node.keywords if k.arg}
                has_star = any(k.arg is None for k in node.keywords)
                if has_star or any(isinstance(a, ast.Starred) for a in node.args):
                    ok = True
                    break
                acc = list(d.accepted_positional)
                if is_bound_call and acc and acc[0] in ("self", "cls"):
                    acc = acc[1:]  # bound call supplies self implicitly
                min_req = len(d.required())
                if is_bound_call and min_req:
                    min_req = max(0, min_req - 1)
                supplied = n_pos + len(kwnames)
                if d.vararg:
                    min_req = min(min_req, len(acc))
                if n_pos > len(acc) and not d.vararg:
                    continue
                if supplied < min_req:
                    continue
                if kwnames - set(acc) - set(d.kwonly) and not d.kwarg:
                    continue
                ok = True
                break
            if not ok:
                kw = sorted(k.arg for k in node.keywords if k.arg)
                problems.append(
                    f"{path}:{node.lineno}: call {name}() with "
                    f"{len(node.args)} positional, kwargs={kw} "
                    f"does not match {[d.required() for d in candidates]}")
    return problems


# ------------------------------------------------------------- B. dead defs


def check_dead(trees: dict[str, ast.Module], defs: dict[str, list[DefInfo]]) -> list[str]:
    """Module-level defs whose name is never referenced anywhere in the repo."""
    referenced: set[str] = set()
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                referenced.add(node.id)
            elif isinstance(node, ast.Attribute):
                referenced.add(node.attr)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    referenced.add(alias.name)
    # names referenced only inside strings (getattr, __all__) count as used
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                referenced.add(node.value)

    dead = []
    for name, infos in defs.items():
        if name in referenced:
            continue
        for d in infos:
            if d.path.startswith("src/fiae"):
                dead.append(f"{d.path}:{d.node.lineno}: {d.name}() defined, never referenced")
    return dead


# ------------------------------------------------------------ C. duplicates


def check_dupes(trees: dict[str, ast.Module]) -> list[str]:
    problems = []
    for path, tree in trees.items():
        seen: dict[str, int] = defaultdict(int)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                seen[node.name] += 1
        for name, count in seen.items():
            if count > 1:
                lines = [n.lineno for n in tree.body
                         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                           ast.ClassDef)) and n.name == name]
                problems.append(f"{path}: {name} defined {count}x at lines {lines}")
    return problems


# ------------------------------------------------------- D. version compat


def check_versions(trees: dict[str, ast.Module]) -> list[str]:
    """Flag syntax/idioms newer than requires-python."""
    problems = []
    for path, tree in trees.items():
        for node in ast.walk(tree):
            # PEP 695 type parameter syntax -> 3.12+ only.
            # getattr: ast.TypeAlias does not exist before 3.12.
            type_alias = getattr(ast, "TypeAlias", None)
            if type_alias is not None and isinstance(node, type_alias):
                problems.append(f"{path}:{node.lineno}: PEP 695 type alias (3.12+)")
            type_var = getattr(ast, "TypeVar", None)
            if type_var is not None and isinstance(node, type_var):
                problems.append(f"{path}:{node.lineno}: PEP 695 TypeVar (3.12+)")
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    and getattr(node, "type_params", None)):
                    problems.append(
                        f"{path}:{node.lineno}: PEP 695 type params on "
                        f"{node.name} (3.12+)")
            # match statement -> 3.10+, fine; but flag anything newer
            if isinstance(node, ast.Attribute) and node.attr == "except_of":
                problems.append(f"{path}:{node.lineno}: except* (3.11+)")
    return problems


def main() -> int:
    trees = load()
    defs = collect_defs(trees)
    methods = collect_methods(trees)
    print(f"parsed {len(trees)} files, {len(defs)} function names, "
          f"{len(methods)} method names\n")

    for title, items in (
        ("A. CALL SIGNATURE MISMATCH", check_calls(trees, defs, methods)),
        ("B. DEAD DEFINITIONS", check_dead(trees, defs)),
        ("C. DUPLICATE DEFINITIONS", check_dupes(trees)),
        ("D. VERSION INCOMPATIBILITY", check_versions(trees)),
    ):
        print(f"=== {title} ({len(items)}) ===")
        for item in items[:40]:
            print("  ", item)
        if len(items) > 40:
            print(f"   ... {len(items) - 40} more")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
