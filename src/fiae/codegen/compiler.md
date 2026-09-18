# Module Overview

This module implements the **doc 09 "Compiler steps" / "Verification loop"** as a pipeline of discrete gates that run against a frozen `PipelineIR`, plus an exporter that packages the compiled pipeline as a standalone sklearn project. Two distinct halves:

1. **Verification gates** — 10 gate functions (covering steps 1–7, 16, 18, 19, 21 of the 22-step loop) that each return a `VerificationResult`.
2. **Orchestration & export** — `compile_pipeline` runs all gates and assembles a `CompilerReport`; `generate_sklearn_project` emits a file dict for a deployable project.

Every gate shares the same **timing envelope**:

```python
t0 = time.monotonic()
... work ...
dt = (time.monotonic() - t0) * 1000      # milliseconds
return VerificationResult(..., duration_ms=dt)
```

```mermaid
flowchart LR
    IR[("PipelineIR")]

    subgraph COMPILE["compile_pipeline — verification loop"]
        direction TB
        G1["gate_validate_ir<br/>step 1"] --> G2["gate_topological_sort<br/>step 2"]
        G2 --> G3["gate_deduplicate<br/>step 3"]
        G3 --> G4["gate_partition_fit_transform<br/>step 4"]
        G4 --> G5["gate_select_backend<br/>step 5"]
        G5 --> G6["gate_emit_contracts<br/>steps 6-7"]
        G6 --> G7["gate_syntax_check<br/>step 16"]
        G7 --> G8["gate_feature_parity<br/>step 18"]
        G8 --> G9["gate_prediction_parity<br/>step 19"]
        G9 --> G10["gate_latency_resource<br/>step 21"]
    end

    GEN["generate_python_code"]
    IR --> COMPILE
    G7 -.->|generates code internally| GEN
    COMPILE --> RPT["CompilerReport<br/>gates + generated_code + manifest"]
    IR --> EXPORT["generate_sklearn_project"]
    GEN --> EXPORT
    EXPORT --> FILES["5-file project dict:<br/>pyproject.toml, src/features.py,<br/>src/contracts.py, manifest.json, README.md"]
```

| Gate | Doc 09 step | Pass criterion | Can fail? |
|---|---|---|---|
| `gate_validate_ir` | 1 | `ir.verify_dag()` returns no errors | Yes |
| `gate_topological_sort` | 2 | topo order covers every node | Yes |
| `gate_deduplicate` | 3 | — | No (always passes) |
| `gate_partition_fit_transform` | 4 | — | No (always passes) |
| `gate_select_backend` | 5 | — | No (always passes) |
| `gate_emit_contracts` | 6–7 | every input is `raw:*` or an existing node | Yes |
| `gate_syntax_check` | 16 | generated code compiles | Yes |
| `gate_feature_parity` | 18 | every expected feature exists in IR | Yes |
| `gate_prediction_parity` | 19 | `class_mapping` is a dict; threshold ∈ [0, 1] | Yes |
| `gate_latency_resource` | 21 | node count and estimated latency within limits | Yes |

Steps **8–15, 17, 20, 22** (test emission, benchmarks, packaging, etc.) are not implemented here — per the docstring, they "happen at the integration boundary."

---

## 1. `VerificationResult` (dataclass)

Value object for one gate's outcome. Sensible defaults (`message=""`, `details={}` via `default_factory`, `duration_ms=0.0`) so minimal results are easy to construct.

| Field | Meaning |
|---|---|
| `gate_name` | Which gate produced this result |
| `passed` | Outcome flag |
| `message` | Human-readable summary or joined error list |
| `details` | Structured extras (node ids, orders, backend lists…) |
| `duration_ms` | Gate execution time from the timing envelope |

```mermaid
classDiagram
    class VerificationResult {
        +str gate_name
        +bool passed
        +str message
        +dict~str,Any~ details
        +float duration_ms
    }
```

---

## 2. `CompilerReport` (dataclass)

Accumulates gate results plus the compilation artifacts. `all_passed` starts `True` and is only ever driven to `False` — a **sticky** aggregate.

```mermaid
classDiagram
    class CompilerReport {
        +str pipeline_id
        +list~VerificationResult~ gates
        +str generated_code
        +dict~str,Any~ manifest
        +bool all_passed
        +float total_duration_ms
        +add_gate(result)
        +summary() dict
    }
    CompilerReport "1" o-- "*" VerificationResult : gates
```

### `add_gate(result)`

```mermaid
flowchart TD
    A([add_gate]) --> B["append result to gates list"]
    B --> C{"result.passed is False?"}
    C -->|"yes"| D["all_passed = False<br/>(sticky — never reset)"]
    C -->|"no"| E([return])
    D --> E
```

### `summary()`

Single-pass counts over `gates` using generator expressions:

```mermaid
flowchart LR
    A([summary]) --> B["count passed gates<br/>count failed gates"]
    B --> C([Return dict: pipeline_id, total_gates,<br/>passed, failed, all_passed, total_duration_ms])
```

---

## 3. `gate_validate_ir(ir)` — Step 1

Delegates entirely to `ir.verify_dag()`; the gate is a thin wrapper that converts an error list into a pass/fail with the errors joined by semicolons as the message. Assumes `verify_dag()` returns a list rather than raising.

```mermaid
flowchart TD
    A([gate_validate_ir]) --> B["errors = ir.verify_dag"]
    B --> C{"errors empty?"}
    C -->|"yes"| D["passed = True<br/>message: IR structure valid"]
    C -->|"no"| E["passed = False<br/>message: errors joined with semicolons"]
    D --> F([Return VerificationResult<br/>gate_name = validate_ir])
    E --> F
```

---

## 4. `gate_topological_sort(ir)` — Step 2

Calls `ir.topological_order()` and checks **coverage**: the sorted order must contain exactly as many nodes as the IR. A shortfall implies a cycle or unreachable node (assuming the method returns a partial list rather than raising on cycles).

```mermaid
flowchart TD
    A([gate_topological_sort]) --> B["order = ir.topological_order"]
    B --> C{"len of order ==<br/>len of ir.nodes?"}
    C -->|"yes"| D["passed = True<br/>full DAG coverage"]
    C -->|"no"| E["passed = False<br/>cycle or unreachable node"]
    D --> F([Return VerificationResult<br/>details: ordered node ids])
    E --> F
```

---

## 5. `gate_deduplicate(ir)` — Step 3

Builds a **structural fingerprint** per node:

```
h = f"{node.operator}:{tuple(sorted(node.inputs))}:{tuple(sorted(node.params.items()))}"
```

- Order-insensitive: same operator + same *set* of inputs + same *set* of params ⇒ same hash.
- **First occurrence wins**; later nodes with an identical hash are flagged in `duplicates`.
- The gate **always passes** — duplicates are recorded in `details`, not treated as a failure. (The actual merge presumably happens in a later compiler step not implemented here.)

```mermaid
flowchart TD
    A([gate_deduplicate]) --> B["seen_hashes = empty set<br/>duplicates = empty list"]
    B --> C{"More nodes in ir.nodes?"}
    C -->|"yes"| D["node = next"]
    D --> E["h = operator + sorted inputs<br/>+ sorted params"]
    E --> F{"h already in seen_hashes?"}
    F -->|"yes"| G["mark node_id as duplicate<br/>and record h"]
    F -->|"no"| H["record h"]
    G --> C
    H --> C
    C -->|"no"| I(["Return passed = always True<br/>duplicates listed in details"])
```

---

## 6. `gate_partition_fit_transform(ir)` — Step 4

Splits nodes by `fit_scope` to separate **fit-time state** (must be learned on training data) from **transform-time state** (stateless). Informational — always passes.

⚠️ **Edge case:** a node whose `fit_scope` is *not* one of `"training_fold"`, `"development"`, or `"none"` falls into **neither** bucket — the two counts may not sum to `len(ir.nodes)`, and such nodes are silently unaccounted for.

```mermaid
flowchart TD
    A([gate_partition_fit_transform]) --> B["Iterate over ir.nodes"]
    B --> C{"fit_scope is training_fold<br/>or development?"}
    C -->|"yes"| D["classify as fit-time node"]
    C -->|"no"| E{"fit_scope is none?"}
    E -->|"yes"| F["classify as transform-time node"]
    E -->|"other value"| G["falls in neither bucket"]
    D --> H(["Return passed = always True<br/>counts and node ids per bucket in details"])
    F --> H
    G --> H
```

---

## 7. `gate_select_backend(ir)` + `_select_backend(node)` — Step 5

Maps each node to an implementation backend via a hardcoded operator set. Seven operators go to **sklearn** (`pca`, `truncated_svd`, `text_svd`, `kmeans_label`, `kmeans_distances`, `tfidf_word`, `tfidf_char`); everything else is **numpy**. The gate aggregates the distinct backend set — informational, always passes.

```mermaid
flowchart TD
    A([gate_select_backend]) --> B{"More nodes in ir.nodes?"}
    B -->|"yes"| C["node = next"]
    C --> D["backends.add<br/>_select_backend of node"]
    D --> B
    B -->|"no"| E(["Return passed = always True<br/>details: sorted backend list"])

    F(["_select_backend node"]) --> G{"operator in sklearn set?<br/>pca, truncated_svd, text_svd,<br/>kmeans_label, kmeans_distances,<br/>tfidf_word, tfidf_char"}
    G -->|"yes"| H([Return sklearn])
    G -->|"no"| I([Return numpy])

    D -.-> F
```

---

## 8. `gate_emit_contracts(ir)` — Steps 6–7

Validates that every node's inputs are resolvable. An input is legal iff it **starts with `"raw:"`** (a raw source column) **or** `ir.get_node(inp)` finds it (a derived feature reference). All errors are collected (not fail-fast) as `"node_id: missing input inp"` strings.

```mermaid
flowchart TD
    A([gate_emit_contracts]) --> B["errors = empty list"]
    B --> C{"More nodes?"}
    C -->|"yes"| D["node = next"]
    D --> E{"More inputs on node?"}
    E -->|"yes"| F["inp = next input"]
    F --> G{"inp starts with raw: ?"}
    G -->|"yes"| E
    G -->|"no"| H{"ir.get_node inp<br/>returns a node?"}
    H -->|"yes"| E
    H -->|"no"| I["append error:<br/>node id missing input inp"]
    I --> E
    E -->|"no"| C
    C -->|"no"| J{"errors empty?"}
    J -->|"yes"| K([Return passed = True<br/>All contracts valid])
    J -->|"no"| L([Return passed = False<br/>errors joined with semicolons])
```

---

## 9. `gate_feature_parity(ir, expected_features, tolerance)` — Step 18

⚠️ **This is a structural stub.** The docstring describes full parity checking (identity, dtype, row alignment, null mask, numeric tolerance), but the implementation only verifies that each expected feature's `node_id` **exists in the IR**. Empty value lists are skipped; no values are compared, and `tolerance` is accepted but **never used**. The real value comparison happens at the integration boundary.

```mermaid
flowchart TD
    A([gate_feature_parity]) --> B{"expected_features provided?"}
    B -->|"no"| J([Return passed = True<br/>Feature parity verified])
    B -->|"yes"| C{"More entries?"}
    C -->|"yes"| D["node_id, values = next entry"]
    D --> E{"ir.get_node node_id<br/>returns None?"}
    E -->|"yes"| F["append error:<br/>expected feature not in IR"]
    E -->|"no"| G["values accepted — no value<br/>comparison performed (stub)"]
    F --> C
    G --> C
    C -->|"no"| H{"errors empty?"}
    H -->|"yes"| J
    H -->|"no"| I([Return passed = False<br/>errors joined with semicolons])
```

---

## 10. `gate_prediction_parity(ir, expected_predictions, tolerance)` — Step 19

Another **structural check** rather than a true parity test. Only two keys of `expected_predictions` are inspected:

- `"class_mapping"` → must be a `dict`
- `"threshold"` → must be within `[0.0, 1.0]`

All other keys are **silently ignored**, and `tolerance` is unused. Note that if both checks fail, only the **last** failure message survives (message is overwritten, not appended).

```mermaid
flowchart TD
    A([gate_prediction_parity]) --> B{"expected_predictions provided?"}
    B -->|"no"| Z([Return passed = True<br/>message: Prediction parity verified])
    B -->|"yes"| C{"More entries?"}
    C -->|"yes"| D["key, val = next entry"]
    D --> E{"key == class_mapping?"}
    E -->|"yes"| F{"val is a dict?"}
    F -->|"yes"| C
    F -->|"no"| G["passed = False<br/>message: invalid class mapping type"]
    E -->|"no"| H{"key == threshold?"}
    H -->|"yes"| I{"val within 0.0 to 1.0?"}
    I -->|"yes"| C
    I -->|"no"| K["passed = False<br/>message: threshold out of range"]
    H -->|"no — other keys ignored"| C
    G --> C
    K --> C
    C -->|"no"| Z2([Return VerificationResult<br/>gate_name = prediction_parity])
```

---

## 11. `gate_latency_resource(ir, max_latency_ms, max_nodes)` — Step 21

A **cheap static estimate**, not a benchmark: `estimated_ms = n_nodes × 0.5`. Passes iff both `n_nodes ≤ max_nodes` and `estimated_ms ≤ max_latency_ms`.

⚠️ With defaults, the node cap is always the binding constraint: 500 nodes × 0.5 ms = **250 ms ≪ 5000 ms**. The latency limit can only fail if configured below roughly `n_nodes × 0.5`.

```mermaid
flowchart TD
    A([gate_latency_resource]) --> B["n_nodes = len of ir.nodes"]
    B --> C["estimated_ms = n_nodes × 0.5<br/>(rough per-node estimate)"]
    C --> D{"n_nodes ≤ max_nodes<br/>AND estimated_ms ≤ max_latency_ms?"}
    D -->|"yes"| E([Return passed = True<br/>details: n_nodes, estimated_ms])
    D -->|"no"| F([Return passed = False])
```

---

## 12. `gate_syntax_check(ir)` — Step 16

Generates the pipeline's Python code, then runs the builtin `compile(code, "<pipeline_ir>", "exec")`. This **parses and byte-compiles without executing**, so it catches syntax errors but *not* missing imports or runtime errors (e.g., sklearn availability is unverified until execution). Only `SyntaxError` is caught.

```mermaid
flowchart TD
    A([gate_syntax_check]) --> B["code = generate_python_code ir"]
    B --> C["compile code with mode exec"]
    C --> D{"SyntaxError raised?"}
    D -->|"no"| E([Return passed = True<br/>Generated code compiles successfully])
    D -->|"yes"| F([Return passed = False<br/>message includes the error])
```

---

## 13. `compile_pipeline(ir, expected_features, expected_predictions)`

The orchestrator. Creates a `CompilerReport` and runs all ten gates in **fixed order** (matching the step numbering: 1–5, 6–7, 16, 18, 19, 21). Key behaviors:

- **Non-blocking:** every gate runs even if an earlier one failed; failures only flip `all_passed`.
- **Code is generated twice** — once inside `gate_syntax_check`, then again for `report.generated_code`.
- The **manifest** records `pipeline_hash` (via `ir.compute_hash()`), node/output counts, and gate tallies.
- Total duration measured around the whole loop, independent of per-gate timings.

```mermaid
flowchart TD
    A([compile_pipeline]) --> B["report = new CompilerReport<br/>pipeline_id = ir.pipeline_id<br/>t_start = monotonic clock"]
    B --> C1["add gate_validate_ir<br/>(step 1)"]
    C1 --> C2["add gate_topological_sort<br/>(step 2)"]
    C2 --> C3["add gate_deduplicate<br/>(step 3)"]
    C3 --> C4["add gate_partition_fit_transform<br/>(step 4)"]
    C4 --> C5["add gate_select_backend<br/>(step 5)"]
    C5 --> C6["add gate_emit_contracts<br/>(steps 6-7)"]
    C6 --> C7["add gate_syntax_check<br/>(step 16)"]
    C7 --> C8["add gate_feature_parity<br/>with expected_features (step 18)"]
    C8 --> C9["add gate_prediction_parity<br/>with expected_predictions (step 19)"]
    C9 --> C10["add gate_latency_resource<br/>(step 21)"]
    C10 --> D["generated_code =<br/>generate_python_code ir<br/>(second generation)"]
    D --> E["manifest = pipeline hash,<br/>n_nodes, outputs,<br/>gates_passed, gates_total"]
    E --> F["total_duration_ms =<br/>elapsed since t_start"]
    F --> G([Return CompilerReport])
```

---

## 14. `generate_sklearn_project(ir, output_dir)`

Emits a **5-file project** as a `{relative_path: content}` dict — the caller is responsible for writing it to disk. Contents:

| File | Content |
|---|---|
| `pyproject.toml` | setuptools build config; project `fiae-export` 1.0.0; deps `scikit-learn>=1.3`, `numpy>=1.24` |
| `src/features.py` | The compiled pipeline (`generate_python_code(ir)`) |
| `src/contracts.py` | `InputSchema` (column → dtype) and `OutputSchema` dataclasses |
| `artifacts/manifest.json` | Provenance: pipeline hash, source fingerprint, target, task, node/output counts |
| `README.md` | Pipeline metadata + `apply_pipeline` usage snippet |

⚠️ The `output_dir` parameter is **accepted but unused** — paths are always relative.

```mermaid
flowchart TD
    A([generate_sklearn_project]) --> B["files = empty dict"]
    B --> C["pyproject.toml — setuptools build,<br/>fiae-export 1.0.0, sklearn + numpy deps"]
    C --> D["src/features.py —<br/>generate_python_code ir"]
    D --> E["src/contracts.py —<br/>InputSchema and OutputSchema dataclasses"]
    E --> F["artifacts/manifest.json — pipeline hash,<br/>source fingerprint, target, task,<br/>node count, output count"]
    F --> G["README.md — pipeline id, target, task,<br/>nodes, outputs, usage snippet"]
    G --> H([Return dict mapping<br/>relative path to file content])
```

The resulting project layout:

```mermaid
flowchart TD
    ROOT["project/"] --> PY["pyproject.toml"]
    ROOT --> SRC["src/"]
    SRC --> FEAT["features.py — compiled pipeline"]
    SRC --> CONTR["contracts.py — I/O schemas"]
    ROOT --> ART["artifacts/"]
    ART --> MAN["manifest.json — provenance"]
    ROOT --> RD["README.md — usage"]
```

---

## Key observations

- **Partial step coverage:** only steps 1–7, 16, 18, 19, 21 of the 22-step loop exist here; steps 8–15, 17, 20, 22 (test emission, benchmarks, packaging) are deferred to the integration boundary.
- **Three gates can never fail** (`deduplicate`, `partition_fit_transform`, `select_backend`) — they're informational passes that enrich `details`.
- **Parity gates are stubs:** their docstrings describe full value-level comparison, but the implementations only do structural checks; both `tolerance` parameters are dead.
- **Gates are non-blocking and `all_passed` is sticky** — one failure anywhere permanently marks the report.
- **Redundant code generation:** `generate_python_code` runs up to twice per compile (inside `syntax_check` and for the report).
- **Partition blind spot:** unknown `fit_scope` values land in neither the fit nor transform bucket, so counts can silently under-total.
- **Latency check is dormant at defaults:** the node cap (500 → 250 ms) binds long before the 5000 ms latency limit.
- **Minor hygiene:** `output_dir` is unused in the exporter, and `FeatureNode`, `new_id`, and `Optional` are imported but never referenced.