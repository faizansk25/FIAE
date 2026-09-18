"""Pipeline Intermediate Representation (doc 09, doc 14, FR-014).

A structured, serializable representation of a feature engineering pipeline
that can be:
- Verified against internal predictions
- Code-generated into standalone Python
- Compared across runs for reproducibility

Normative source: doc 09 "Code generation", doc 14 "Pipeline IR",
FR-014 "Generated code is derived from structured pipeline IR".
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from ..ids import content_hash, new_id


@dataclass
class IRNode:
    """A single node in the pipeline IR."""

    node_id: str
    operator: str
    inputs: list[str]  # node_ids of dependencies
    params: dict[str, Any] = field(default_factory=dict)
    fit_scope: str = "none"  # "none" | "training_fold" | "development"
    output_type: str = "numeric"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> IRNode:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PipelineIR:
    """Complete pipeline intermediate representation (FR-014).

    A directed acyclic graph of IRNodes representing the full feature
    engineering pipeline from raw inputs to final feature set.
    """

    pipeline_id: str = ""
    version: str = "1.0"
    source_fingerprint: str = ""
    target: str = ""
    task: str = ""
    nodes: list[IRNode] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)  # node_ids of final features
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.pipeline_id:
            self.pipeline_id = new_id("pipe")

    def add_node(self, node: IRNode) -> None:
        """Add a node to the pipeline."""
        self.nodes.append(node)

    def get_node(self, node_id: str) -> Optional[IRNode]:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None

    def topological_order(self) -> list[IRNode]:
        """Return nodes in topological order (dependencies first)."""
        visited = set()
        order = []

        def _visit(node_id: str):
            if node_id in visited:
                return
            visited.add(node_id)
            node = self.get_node(node_id)
            if node is None:
                return
            for inp in node.inputs:
                if not inp.startswith("raw:"):
                    _visit(inp)
            order.append(node)

        for node in self.nodes:
            _visit(node.node_id)

        return order

    def verify_dag(self) -> list[str]:
        """Verify the IR forms a valid DAG. Returns list of errors."""
        errors = []
        node_ids = {n.node_id for n in self.nodes}

        for node in self.nodes:
            for inp in node.inputs:
                if not inp.startswith("raw:") and inp not in node_ids:
                    errors.append(f"Node {node.node_id}: missing dependency {inp}")

        # Check for cycles
        in_degree = {n.node_id: 0 for n in self.nodes}
        adj = {n.node_id: [] for n in self.nodes}
        for node in self.nodes:
            for inp in node.inputs:
                if inp in adj:
                    adj[inp].append(node.node_id)
                    in_degree[node.node_id] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        visited = 0
        while queue:
            nid = queue.pop(0)
            visited += 1
            for neighbor in adj[nid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited != len(self.nodes):
            errors.append("Cycle detected in pipeline DAG")

        return errors

    def compute_hash(self) -> str:
        """Compute deterministic hash of the pipeline structure."""
        data = {
            "nodes": [n.to_dict() for n in self.topological_order()],
            "outputs": sorted(self.outputs),
        }
        return content_hash(data)

    def to_dict(self) -> dict:
        return {
            "pipeline_id": self.pipeline_id,
            "version": self.version,
            "source_fingerprint": self.source_fingerprint,
            "target": self.target,
            "task": self.task,
            "nodes": [n.to_dict() for n in self.nodes],
            "outputs": self.outputs,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PipelineIR:
        nodes = [IRNode.from_dict(n) for n in d.get("nodes", [])]
        return cls(
            pipeline_id=d.get("pipeline_id", ""),
            version=d.get("version", "1.0"),
            source_fingerprint=d.get("source_fingerprint", ""),
            target=d.get("target", ""),
            task=d.get("task", ""),
            nodes=nodes,
            outputs=d.get("outputs", []),
            metadata=d.get("metadata", {}),
        )


def build_ir_from_proposals(
    proposals: list, source_fingerprint: str = "",
    target: str = "", task: str = "",
) -> PipelineIR:
    """Build a PipelineIR from a list of FeatureProposals."""
    ir = PipelineIR(
        source_fingerprint=source_fingerprint,
        target=target, task=task,
    )
    seen = {}

    for node_counter, proposal in enumerate(proposals):
        inputs = []
        for fid in proposal.inputs:
            if fid.startswith("raw:"):
                inputs.append(fid)
            elif fid in seen:
                inputs.append(seen[fid])
            else:
                inputs.append(fid)

        node_id = f"n_{node_counter}"

        node = IRNode(
            node_id=node_id,
            operator=proposal.op,
            inputs=inputs,
            params=dict(proposal.params),
        )
        ir.add_node(node)
        seen[proposal.op + "(" + "_".join(inputs) + ")"] = node_id
        ir.outputs.append(node_id)

    return ir


def generate_python_code(ir: PipelineIR) -> str:
    """Generate standalone Python code from PipelineIR (FR-014).

    The generated code:
    - Takes a pandas DataFrame or dict of columns as input
    - Applies each transform in topological order
    - Returns a dict of output feature arrays
    - Is self-contained (no FIAE imports needed)
    """
    # Raw columns referenced by the IR get a type-coercion preamble: CSV/JSON
    # sources deliver strings, and the fitted runtime parses them to floats
    # before transforms run. The exported code must do the same or numeric
    # comparisons (v >= 0, abs(v), v/…) crash on strings — and parity with the
    # fitted pipeline is lost in exactly the case real users hit first.
    raw_cols = sorted({inp[4:] for node in ir.topological_order()
                       for inp in node.inputs if inp.startswith("raw:")})
    lines = [
        '\"\"\"Auto-generated feature pipeline.\"\"\"',
        'from __future__ import annotations',
        'import math',
        'from hashlib import md5',
        '',
        '',
        'def apply_pipeline(data: dict[str, list]) -> dict[str, list]:',
        '    \"\"\"Apply the feature pipeline to input data.\"\"\"',
        '    outputs = {}',
        '',
    ]
    if raw_cols:
        cols_lit = ", ".join('"' + c + '"' for c in raw_cols)
        lines += [
            '    # Coerce raw inputs to floats (matches fitted-runtime intake);',
            '    # works on a copy so the caller\'s dict is never mutated.',
            '    _data = dict(data)',
            '    _num = ' + '{' + cols_lit + '}',
            '    for _c in _num:',
            '        _vals = _data.get(_c)',
            '        if _vals is None:',
            '            continue',
            '        _conv = []',
            '        for _v in _vals:',
            '            if isinstance(_v, bool):',
            '                _conv.append(1.0 if _v else 0.0)',
            '            elif isinstance(_v, (int, float)):',
            '                _conv.append(float(_v))',
            '            else:',
            '                try:',
            '                    _conv.append(float(str(_v).strip()) if _v is not None else None)',
            '                except (ValueError, TypeError):',
            '                    _conv.append(None)',
            '        _data[_c] = _conv',
            '',
        ]

    for node in ir.topological_order():
        # Map inputs
        input_refs = []
        for inp in node.inputs:
            if inp.startswith("raw:"):
                col_name = inp[4:]
                input_refs.append(f'_data.get("{col_name}", [])')
            else:
                input_refs.append(f'outputs.get("{inp}", [])')

        if node.operator == "hash_encode":
            # Export the real bucketing so the exported pipeline matches the
            # fitted one (n_buckets/seed change the output distribution).
            lines.append(f'    # hash_encode({", ".join(node.inputs)})')
            lines.append(f'    _vals = {input_refs[0]}')
            lines.append(f'    _n_buckets = {int(node.params.get("n_buckets", 16))}')
            lines.append(f'    _seed = {int(node.params.get("seed", 0))}')
            lines.append('    outputs["' + node.node_id + '"] = [')
            lines.append('        None if v is None else int(md5(f"{_seed}:{v}".encode()).hexdigest(), 16) % _n_buckets')
            lines.append('        for v in _vals')
            lines.append('    ]')
        elif node.operator in ("sqrt", "log1p", "abs", "square", "cbrt",
                             "sign", "reciprocal", "exp_clip"):
            if len(input_refs) == 1:
                lines.append(f'    # {node.operator}({", ".join(node.inputs)})')
                if node.operator == "sqrt":
                    lines.append(f'    _vals = {input_refs[0]}')
                    lines.append(f'    outputs["{node.node_id}"] = [')
                    lines.append('        math.sqrt(v) if v is not None and v >= 0 else None')
                    lines.append('        for v in _vals')
                    lines.append('    ]')
                elif node.operator == "log1p":
                    lines.append(f'    _vals = {input_refs[0]}')
                    lines.append(f'    outputs["{node.node_id}"] = [')
                    lines.append('        math.log1p(v) if v is not None and v >= -1.0 else None')
                    lines.append('        for v in _vals')
                    lines.append('    ]')
                elif node.operator == "abs":
                    lines.append(f'    _vals = {input_refs[0]}')
                    lines.append(f'    outputs["{node.node_id}"] = [')
                    lines.append('        abs(v) if v is not None else None for v in _vals')
                    lines.append('    ]')
                else:
                    lines.append(f'    outputs["{node.node_id}"] = {input_refs[0]}  # passthrough for {node.operator}')
        elif node.operator in ("sum", "difference", "product") and len(input_refs) == 2:
            lines.append(f'    # {node.operator}({", ".join(node.inputs)})')
            lines.append(f'    _a, _b = {input_refs[0]}, {input_refs[1]}')
            if node.operator == "sum":
                lines.append(f'    outputs["{node.node_id}"] = [')
                lines.append('        (x + y if x is not None and y is not None else None)')
                lines.append('        for x, y in zip(_a, _b)')
                lines.append('    ]')
            elif node.operator == "difference":
                lines.append(f'    outputs["{node.node_id}"] = [')
                lines.append('        (x - y if x is not None and y is not None else None)')
                lines.append('        for x, y in zip(_a, _b)')
                lines.append('    ]')
            elif node.operator == "product":
                lines.append(f'    outputs["{node.node_id}"] = [')
                lines.append('        (x * y if x is not None and y is not None else None)')
                lines.append('        for x, y in zip(_a, _b)')
                lines.append('    ]')
        elif node.operator == "safe_ratio" and len(input_refs) == 2:
            # a/b with explicit missing for null/near-zero denominators,
            # matching tf_safe_ratio semantics exactly.
            lines.append(f'    # safe_ratio({", ".join(node.inputs)})')
            lines.append(f'    _a, _b = {input_refs[0]}, {input_refs[1]}')
            lines.append(f'    _eps = {float(node.params.get("eps", 1e-12))!r}')
            lines.append(f'    outputs["{node.node_id}"] = [')
            lines.append('        (')
            lines.append('            (x / y) if (x is not None and y is not None and abs(y) > _eps)')
            lines.append('            else None')
            lines.append('        )')
            lines.append('        for x, y in zip(_a, _b)')
            lines.append('    ]')
        elif node.operator == "zero_indicator":
            lines.append(f'    _vals = {input_refs[0]}')
            lines.append(f'    outputs["{node.node_id}"] = [')
            lines.append('        (v == 0.0 if v is not None else None) for v in _vals')
            lines.append('    ]')
        elif node.operator == "missing_indicator":
            lines.append(f'    # missing_indicator({", ".join(node.inputs)})')
            lines.append(f'    _vals = {input_refs[0]}')
            lines.append(f'    outputs["{node.node_id}"] = [')
            lines.append('        (v is None) for v in _vals')
            lines.append('    ]')
        elif node.operator == "identity":
            lines.append(f'    outputs["{node.node_id}"] = {input_refs[0]}')
        elif node.operator == "cbrt" and len(input_refs) == 1:
            lines.append(f'    # cbrt({", ".join(node.inputs)})')
            lines.append(f'    _vals = {input_refs[0]}')
            lines.append(f'    outputs["{node.node_id}"] = [')
            lines.append('        math.copysign(abs(v) ** (1.0 / 3.0), v) if v is not None else None')
            lines.append('        for v in _vals')
            lines.append('    ]')
        elif node.operator == "sign" and len(input_refs) == 1:
            lines.append(f'    # sign({", ".join(node.inputs)})')
            lines.append(f'    _vals = {input_refs[0]}')
            lines.append(f'    outputs["{node.node_id}"] = [')
            lines.append('        None if v is None else (-1 if v < 0 else (0 if v == 0 else 1))')
            lines.append('        for v in _vals')
            lines.append('    ]')
        elif node.operator == "reciprocal" and len(input_refs) == 1:
            lines.append(f'    # reciprocal({", ".join(node.inputs)})')
            lines.append(f'    _vals = {input_refs[0]}')
            lines.append(f'    _eps = {float(node.params.get("eps", 1e-12))!r}')
            lines.append(f'    outputs["{node.node_id}"] = [')
            lines.append('        (1.0 / v) if (v is not None and abs(v) > _eps) else None')
            lines.append('        for v in _vals')
            lines.append('    ]')
        elif node.operator == "exp_clip" and len(input_refs) == 1:
            lines.append(f'    # exp_clip({", ".join(node.inputs)})')
            lines.append(f'    _vals = {input_refs[0]}')
            lines.append(f'    _lo = {float(node.params.get("lo", -20.0))!r}')
            lines.append(f'    _hi = {float(node.params.get("hi", 20.0))!r}')
            lines.append(f'    outputs["{node.node_id}"] = [')
            lines.append('        math.exp(min(_hi, max(_lo, v))) if v is not None else None')
            lines.append('        for v in _vals')
            lines.append('    ]')
        else:
            # Supported operators must be handled above. Anything else must
            # FAIL LOUD, not emit raw inputs under the feature's name — that
            # silently corrupted exported pipelines (pre-0.0.2 behavior).
            raise ValueError(
                f"generate_python_code: operator '{node.operator}' has no "
                f"export handler; refusing to emit a silent passthrough"
            )

        lines.append("")

    # Return final outputs
    lines.append('    return {k: outputs[k] for k in outputs if k in [')
    for oid in ir.outputs:
        lines.append(f'        "{oid}",')
    lines.append('    ]}')

    return "\n".join(lines)
