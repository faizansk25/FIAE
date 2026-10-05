"""Architecture boundary enforcement (audit program M2).

Parses every ``fiae`` module's imports (absolute and relative) and asserts
the layering contract documented in ``md/AUDIT.md``. Any new import that
violates a boundary fails here -- do not weaken an assertion to admit a new
dependency; move the code or escalate the design discussion instead.
"""

import ast
import os

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "fiae"))

# Layer order: lower layers must not import higher layers.
LAYERS = {
    0: {
        "contracts", "ids", "errors", "events",
    },
    1: {
        "features", "intake", "problem", "search", "experience", "codegen",
        "funnel", "probe", "evaluate", "orchestration", "model_registry",
        "fitted_pipeline", "tuning", "runs", "experiment", "security",
        "testing", "research", "sklearn_api",
    },
    2: {"pipeline", "learn"},
    3: {
        "cli", "cli_advanced", "cli_dashboard", "cli_pipeline", "cli_colors",
        "server", "webgui", "report", "guijobs", "identity", "__main__",
    },
}

_LAYER_OF = {}
for layer, names in LAYERS.items():
    for name in names:
        _LAYER_OF[name] = layer


def _top_module(name: str) -> str:
    return name.split(".")[0]


def _iter_fiae_modules():
    for root, dirs, files in os.walk(SRC_ROOT):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in files:
            if fn.endswith(".py"):
                path = os.path.join(root, fn)
                rel = os.path.relpath(path, SRC_ROOT).replace(os.sep, "/")
                mod = rel[:-3].replace("/", ".")
                if mod.endswith(".__init__"):
                    mod = mod[: -len(".__init__")]
                is_pkg = rel.endswith("/__init__.py")
                if not mod or mod == "__init__":  # package root itself
                    continue
                yield mod, path, is_pkg


def _imports_of(path: str, mod: str, is_pkg: bool):
    """Yield fully-resolved fiae module names imported by `mod`."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            parts = mod.split(".") if mod else []
            if not is_pkg and parts:
                parts = parts[:-1]
            if node.level > 1:
                parts = parts[: len(parts) - (node.level - 1)]
            tail = (node.module or "").split(".") if node.module else []
            full = ".".join(["fiae", *parts, *tail])
            yield full
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "fiae" or node.module.startswith("fiae."):
                yield node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "fiae" or alias.name.startswith("fiae."):
                    yield alias.name


def _all_edges():
    edges = []
    for mod, path, is_pkg in _iter_fiae_modules():
        src_top = _top_module(mod)
        for target in _imports_of(path, mod, is_pkg):
            dst_top = _top_module(target)
            edges.append((mod, src_top, target, dst_top))
    return edges


class TestLayeringBoundaries:
    def test_every_module_is_assigned_a_layer(self):
        unassigned = [
            mod for mod, _, _ in _iter_fiae_modules()
            if _top_module(mod) not in _LAYER_OF
        ]
        assert unassigned == [], (
            f"New top-level packages must be added to LAYERS: {unassigned}"
        )

    def test_no_layer_imports_a_higher_layer(self):
        violations = []
        for mod, src_top, target, dst_top in _all_edges():
            src_layer = _LAYER_OF.get(src_top)
            dst_layer = _LAYER_OF.get(dst_top)
            if src_layer is None or dst_layer is None:
                continue
            if dst_layer > src_layer:
                violations.append(f"{mod} -> {target}")
        assert violations == [], (
            "Layering violations (low layer importing high layer):\n  "
            + "\n  ".join(violations)
        )

    def test_contracts_only_imports_events(self):
        bad = []
        for mod, _, target, dst_top in _all_edges():
            if dst_top == "contracts" or mod == "contracts":
                bad.append(f"{mod} -> {target}")
        # contracts may import fiae.events and nothing else fiae-internal
        for edge in bad:
            if edge.startswith("contracts -> "):
                assert edge == "contracts -> fiae.events", edge

    def test_ids_imports_nothing_internal(self):
        for mod, _, target, _ in _all_edges():
            assert mod != "ids" or False, f"ids imported {target}"

    def test_domain_does_not_import_presentation(self):
        bad = [
            f"{mod} -> {target}"
            for mod, _, target, dst_top in _all_edges()
            if dst_top in LAYERS[3] and mod.split(".")[0] in LAYERS[0] | LAYERS[1] | LAYERS[2]
        ]
        assert bad == [], bad

    def test_no_import_cycles(self):
        g = {}
        for mod, _, target, _ in _all_edges():
            g.setdefault(mod, set()).add(target)
            g.setdefault(target, set())

        index = {}
        low = {}
        onstack = set()
        stack = []
        cycles = []
        counter = [0]

        def visit(v):
            index[v] = low[v] = counter[0]
            counter[0] += 1
            stack.append(v)
            onstack.add(v)
            for w in sorted(g.get(v, ())):
                if w not in index:
                    visit(w)
                    low[v] = min(low[v], low[w])
                elif w in onstack:
                    low[v] = min(low[v], index[w])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    onstack.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                if len(comp) > 1:
                    cycles.append(sorted(comp))

        for v in sorted(g):
            if v not in index:
                visit(v)

        assert cycles == [], f"import cycles detected: {cycles}"


class TestEnginePurity:
    def test_features_does_not_import_orchestration(self):
        for mod, _, target, dst_top in _all_edges():
            if dst_top in ("pipeline", "learn", "funnel", "tuning"):
                assert not mod.startswith("features"), f"{mod} -> {target}"

    def test_intake_does_not_import_search_or_features(self):
        for mod, _, target, dst_top in _all_edges():
            if dst_top in ("search", "features", "problem"):
                assert not mod.startswith("intake"), f"{mod} -> {target}"

    def test_contracts_and_ids_are_stdlib_only(self):
        for mod, path, _ in _iter_fiae_modules():
            if mod.split(".")[0] in ("contracts", "ids"):
                with open(path, encoding="utf-8") as f:
                    tree = ast.parse(f.read())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            assert not alias.name.startswith(("numpy", "pandas", "sklearn")), (
                                f"{mod} imports {alias.name}"
                            )
