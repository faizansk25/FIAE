"""One-shot import-graph analysis for the FIAE architecture audit (M2).

Writes tools/import_graph.json and prints top-level layer edges, fan-in,
LOC hotspots, and cycles. Safe to delete after the audit is captured.
"""

import ast
import collections
import json
import os

SRC = "src/fiae"


def module_name(rel_path: str) -> str:
    parts = rel_path.replace(os.sep, "/")[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def collect():
    edges = set()  # (src_module, dst_module)
    loc = {}
    for root, dirs, files in os.walk(SRC):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, SRC)
            mod = module_name(rel)
            is_pkg = rel.replace(os.sep, "/").endswith("/__init__.py")
            with open(path, encoding="utf-8") as f:
                text = f.read()
            loc[mod] = text.count("\n") + 1
            tree = ast.parse(text)
            for node in ast.walk(tree):
                targets = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.level:  # relative import: from .contracts import x
                        parts = mod.split(".") if mod else []
                        if not is_pkg and parts:
                            parts = parts[:-1]  # module -> its package
                        if node.level > 1:
                            parts = parts[: len(parts) - (node.level - 1)]
                        full = ".".join(["fiae", *parts, *node.module.split(".")])
                    else:
                        full = node.module
                    if full == "fiae":
                        targets.append("")
                    elif full.startswith("fiae."):
                        targets.append(full[len("fiae."):])
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "fiae":
                            targets.append("")
                        elif alias.name.startswith("fiae."):
                            targets.append(alias.name[len("fiae."):])
                for t in targets:
                    edges.add((mod, t))
    return edges, loc


def main():
    edges, loc = collect()

    # Package-level view: a module imports a package if it imports any
    # submodule of it or the package __init__.
    pkg_edges = set()
    for src, dst in edges:
        dst_pkg = dst.split(".")[0] if dst else ""
        src_pkg = src.split(".")[0]
        if dst_pkg and dst_pkg != src_pkg:
            pkg_edges.add((src_pkg, dst_pkg))

    print("TOP-LEVEL LAYER EDGES (package -> package):")
    by_src = collections.defaultdict(set)
    for a, b in sorted(pkg_edges):
        by_src[a].add(b)
    for a in sorted(by_src):
        print(f"  {a} -> {sorted(by_src[a])}")

    fanin = collections.Counter(b for _, b in pkg_edges)
    print("\nPACKAGE FAN-IN:", dict(fanin.most_common()))

    print("\nTOP 10 LOC:")
    for mod, n in sorted(loc.items(), key=lambda x: -x[1])[:10]:
        print(f"  {mod}: {n}")

    # Module-level fan-in
    mfanin = collections.Counter(b for _, b in edges)
    print("\nMODULE FAN-IN (top 10):", mfanin.most_common(10))

    def scc(g):
        index = {}
        low = {}
        onstack = set()
        stack = []
        out = []
        counter = [0]

        def visit(v):
            index[v] = low[v] = counter[0]
            counter[0] += 1
            stack.append(v)
            onstack.add(v)
            for w in g.get(v, ()):
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
                    out.append(sorted(comp))

        for v in list(g):
            if v not in index:
                visit(v)
        return out

    g = collections.defaultdict(set)
    for a, b in pkg_edges:
        g[a].add(b)
        g[b].add(a) if False else None
    print("\nPACKAGE CYCLES (directed):", scc(g))

    with open("tools/import_graph.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "module_edges": sorted(edges),
                "package_edges": sorted(pkg_edges),
                "loc": loc,
            },
            f,
            indent=1,
        )
    print("\nwrote tools/import_graph.json")


if __name__ == "__main__":
    main()
