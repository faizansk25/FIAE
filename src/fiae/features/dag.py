"""Acyclic feature graph (doc 04).

A ``FeatureDAG`` accumulates ``FeatureNode`` IR candidates, canonicalizes them
(commutative operand fold, identity collapse), collapses exact duplicates, and
enforces the acyclic invariant (Layer 0 invariant 5).  Nodes are keyed by their
``feature_id`` (content hash of the canonical signature), so two structurally
identical expressions always map to the same node.

Raw features (e.g. dataset columns) enter the graph as leaf ``raw:<name>``
identifiers.  Derived nodes list their input ``feature_id`` strings.
"""

from __future__ import annotations

from typing import Any, Optional

from ..contracts import FeatureNode
from .canonical import drop_identity_inputs, make_feature_node
from .registry import get_operator


class CycleError(ValueError):
    """Raised when adding an edge would create a cycle (invariant 5)."""


class FeatureDAG:
    """Mutable DAG of canonical ``FeatureNode`` expressions.

    Parameters
    ----------
    nodes:
        Optional pre-populated set of nodes to initialize from.
    """

    def __init__(self, nodes: Optional[list[FeatureNode]] = None) -> None:
        self._nodes: dict[str, FeatureNode] = {}
        self._dependents: dict[str, set[str]] = {}  # feature_id -> ids that depend on it

        for n in nodes or []:
            self._nodes[n.feature_id] = n
            self._dependents.setdefault(n.feature_id, set())
            for inp in n.inputs:
                self._dependents.setdefault(inp, set()).add(n.feature_id)

    # ------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------
    def __contains__(self, feature_id: str) -> bool:
        return feature_id in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self):
        return iter(self._nodes.values())

    def get(self, feature_id: str) -> FeatureNode:
        """Look up a node by feature_id; raises ``KeyError`` if absent."""
        return self._nodes[feature_id]

    def lookup(self, feature_id: str) -> Optional[FeatureNode]:
        """Return the node or ``None`` if not present."""
        return self._nodes.get(feature_id)

    @property
    def nodes(self) -> list[FeatureNode]:
        """All nodes in the graph."""
        return list(self._nodes.values())

    def roots(self) -> list[FeatureNode]:
        """Nodes with no inputs inside the graph (leaf raw sources)."""
        return [
            n for n in self._nodes.values()
            if not any(inp in self._nodes for inp in n.inputs)
        ]

    def leaves(self) -> list[FeatureNode]:
        """Nodes that no other node depends on (final outputs)."""
        used: set[str] = set()
        for n in self._nodes.values():
            used.update(n.inputs)
        return [n for n in self._nodes.values() if n.feature_id not in used]

    # ------------------------------------------------------------------
    # mutation
    # ------------------------------------------------------------------
    def add_or_create(
        self,
        op_name: str,
        inputs: list[str],
        params: Optional[dict[str, Any]] = None,
    ) -> FeatureNode:
        """Create (or collapse to existing) a node for the given expression.

        Applies identity-collapse (``x+0 -> x``, ``x*1 -> x``) and commutative
        operand fold before computing the canonical signature.  Raises
        ``CycleError`` if any input transitively references the new node.
        """
        # Validates the operator name (unknown operators raise) before folding.
        get_operator(op_name)
        folded = drop_identity_inputs(op_name, inputs)
        if folded is None:
            raise ValueError(f"degenerate inputs for operator '{op_name}': {inputs}")

        node = make_feature_node(op_name, folded, params)
        self._add_node(node)
        return node

    def add_node(self, node: FeatureNode) -> FeatureNode:
        """Add a pre-built ``FeatureNode`` (canonicalizes against duplicates)."""
        self._add_node(node)
        return node

    def _add_node(self, node: FeatureNode) -> None:
        if node.feature_id in self._nodes:
            # Duplicate -- collapse to existing node (doc 04 invariant).
            return

        # Cycle check: traverse inputs transitively. If we encounter the
        # new node's own feature_id, a cycle would be created.
        visited: set[str] = set()
        stack = list(node.inputs)
        while stack:
            cur = stack.pop()
            if cur == node.feature_id:
                raise CycleError(
                    f"adding node '{node.feature_id}' would create a cycle"
                )
            if cur in visited:
                continue
            visited.add(cur)
            if cur in self._nodes:
                stack.extend(self._nodes[cur].inputs)

        self._nodes[node.feature_id] = node
        self._dependents.setdefault(node.feature_id, set())
        for inp in node.inputs:
            self._dependents.setdefault(inp, set()).add(node.feature_id)

    # ------------------------------------------------------------------
    # ordering / materialization
    # ------------------------------------------------------------------
    def topological_order(self) -> list[FeatureNode]:
        """Return nodes in dependency order (inputs before outputs).

        Raises ``CycleError`` if the graph contains a cycle.
        """
        in_degree: dict[str, int] = dict.fromkeys(self._nodes, 0)
        consumers: dict[str, list[str]] = {fid: [] for fid in self._nodes}

        for node in self._nodes.values():
            for inp in node.inputs:
                if inp in self._nodes:
                    consumers[inp].append(node.feature_id)
                    in_degree[node.feature_id] += 1

        queue = sorted(fid for fid, d in in_degree.items() if d == 0)
        order: list[FeatureNode] = []
        while queue:
            fid = queue.pop(0)
            order.append(self._nodes[fid])
            next_ready: list[str] = []
            for consumer in consumers[fid]:
                in_degree[consumer] -= 1
                if in_degree[consumer] == 0:
                    next_ready.append(consumer)
            queue = sorted(queue + next_ready)

        if len(order) != len(self._nodes):
            raise CycleError("graph contains a cycle; cannot topologically order")

        return order

    def materialize_order(self) -> list[FeatureNode]:
        """Alias: topological order for evaluation/materialization."""
        return self.topological_order()
