# Architecture & Correctness Audit — 2026-09-27

Audit program M2 output. Companion enforcement: `tests/test_architecture.py`,
`tests/test_api_stability.py`, `tests/test_claims.py`. Regenerate the graph
with `python tools/import_graph.py` (writes `tools/import_graph.json`).

## 1. Dependency graph (package level)

```
                     contracts (fan-in 14)   ids (9)   errors (5)
                      /   |    \      |
        features  intake  problem  search  experience  runs  codegen ...
             \      |       /        |
              funnel, pipeline, learn, evaluate, tuning  (orchestration layer)
                          |
        cli / server / webgui / report / guijobs  (presentation layer)
```

Verified properties:

- **No circular dependencies at package or module level** (Tarjan SCC over
  the full module graph: zero multi-node SCCs).
- **`contracts` is the domain core** — dataclasses + enums only, imports
  just `events`. 39 modules import it directly.
- **`ids` is pure** — stdlib only, content-derived, no inbound dependencies.
- **Layering is acyclic downward**: presentation (cli/server/webgui/report)
  → orchestration (pipeline/learn/funnel/tuning) → engine (features/search/
  problem/experience/codegen) → domain (contracts/ids/errors/events).
- `features` does **not** import `pipeline`, `search`, or `learn`.
- `intake` does **not** import `search`, `features`, or `problem`.

## 2. Findings

### Fixed during this audit

| ID | Finding | Resolution |
|----|---------|-----------|
| F-1 | `validate_input_code` missed `__import__(...)` and `open(...)` escape hatches; `validate_column_name` missed code-injection patterns | Both patched in `security/sandbox.py`; enforced by `tests/test_claims.py::TestSecurityClaims` |
| F-2 | README operator-family table listed 14 fabricated families | Replaced with the registry-derived 13 families; enforced by `test_claims.py` count tests |
| F-3 | "25+ connectors" claim not defensible as stated | Reworded to "12 data adapters covering 25+ sources (13 file extensions + 16 URI schemes)"; enforced by `TestConnectorClaims` |
| F-4 | "Deterministic IDs" claim over-broad (run IDs are uuid4 by design) | Claim scoped in docs; `TestDeterminismClaims` proves content-hash IDs stable cross-process and run IDs intentionally ephemeral |

### Accepted risks / watch items (no action now)

| ID | Finding | Rationale |
|----|---------|-----------|
| W-1 | God modules by LOC: `webgui` (1930), `pipeline.canonical` (1022), `report` (898), `server` (751) | Presentation/phase-orchestration code; cohesive per doc 10/15. Splitting before M-phase feature work would churn every PR. Revisit if any exceeds ~2500 LOC. |
| W-2 | `gate_feature_parity` validates node existence + row count only (doc 09 wants value-level row parity) | The full value-level comparison belongs in `testing/property_tests` (export/runtime equivalence, M3); gate hardening tracked separately. |
| W-3 | 5 sites use broad `except Exception: pass` (cli.py ×2, cli_dashboard, events, experiment/tracking) | **Fixed (M32)** — all sites narrowed to specific exception tuples; tracking backend failures now surface in the audit log; a propagation test pins that unexpected error types are no longer swallowed (`tests/test_w3_narrow_exceptions.py`). events.py had zero such sites (audit note was stale). |
| W-4 | Float-coercion helpers duplicated ×4 (`_to_float_or_none`, `_to_floats`, `_to_floats_safe`, `_to_float_list`) | Small, locally-typed variants; unifying risks behavior drift in fitted-pipeline coercion. Consolidate opportunistically. |
| W-5 | Global mutable state: none found at module scope. All shared state is instance-scoped behind `threading.Lock` (events bus, experience store) or process-global registries that are write-once. | — |
| W-6 | `run_in_sandbox` timeout returns a timeout error but the runaway daemon thread remains alive (burning CPU until process exit). Proven by runtime probe (M30): `threading.active_count()` stays elevated after a 1s-timeout infinite loop. Fix: cooperative cancellation or documented boundary. | **Fixed (M31)** — `PyThreadState_SetAsyncExc` interrupt unwinds the pure-Python runaway thread; locked by `tests/test_sandbox_hardening.py` (thread count returns to baseline after timeout) |
| W-7 | `SandboxPolicy.max_memory_bytes` (256 MB) is declared but never enforced — single occurrence is the definition itself. README's security claim covers *validation* (proven), not OS-level memory containment (not claimed, not built). | **Fixed (M31)** — pre-execution memory gate on declared payload size (args/kwargs via `sys.getsizeof`); oversized payloads rejected without executing. OS-level RSS limiting remains a documented out-of-scope boundary. Locked by `tests/test_sandbox_hardening.py` |

## 3. Boundary rules now enforced by tests

`tests/test_architecture.py` parses imports (including relative imports) and
asserts, on every CI run:

1. No `fiae.*` module imports a heavier layer than its own (matrix below).
2. No module in `domain`/`engine` imports the presentation layer.
3. `contracts` imports nothing from `fiae` except `events`.
4. `ids` imports nothing from `fiae`.
5. No import cycles anywhere in `src/fiae`.

Layer order (lower must not import higher):

```
0 domain:        contracts ids errors events
1 engine:        features intake problem search experience codegen
                 funnel probe evaluate orchestration model_registry
                 fitted_pipeline tuning runs experiment security
2 orchestration: pipeline learn
3 presentation:  cli cli_advanced cli_dashboard cli_pipeline cli_colors
                 server webgui report guijobs identity
```

## 4. Public API surface

`fiae.__init__` currently exports only `__version__` and `cli_colors`
(submodules are the de-facto API). `tests/test_api_stability.py` pins the
documented entry points (registry, canonical pipeline, leakage, ids,
intake factory, report, experience store, sandbox) and their signatures,
so any breaking change fails CI rather than surprising users.

## 5. Correctness findings fixed by the audit program (M3-M8)

### Engine bugs found by property-based testing (M3)

| ID | Finding | Resolution |
|----|---------|-----------|
| C-1 | `log1p(-1.0)` crashed with `math domain error`: the `x >= -1` guard passed the boundary where `log1p(-1)` is -inf | Guard corrected to `x > -1`; out-of-domain maps to missing. Enforced by Hypothesis `test_finite_inputs_give_finite_or_none` |
| C-2 | `rolling_count`, `ewma`, `expanding_mean` declared `null_policy="preserve"` but emit values at null positions (counter / forward-carry recursion) | Declarations corrected (`always_defined`, `forward_carry`); `_temporal` helper accepts per-op policy |

### Leakage detector fixes (M4)

| ID | Finding | Resolution |
|----|---------|-----------|
| C-3 | Stage A `deterministic_bijection` hard-rejected **every continuous numeric feature** (all floats distinct ⇒ trivial 1:1; the "is string" guard inspected normalized values, which are always str) | Guard now checks raw value types; false-positive rate on clean data verified 0/30 by `test_stage_a_never_rejects_clean_data` |
| C-4 | Stage B `extreme_mutual_information` flagged every near-unique numeric column (degenerate MI=1.0) | MI branch skipped for identifier-like features (distinct ≈ row count) |
| C-5 | The real Stage A/B detectors were never invoked by any runtime path (`fiae leakage` CLI/GUI use only distinct-ratio heuristics) | **Fixed (M25)**: `phase_validate` now runs `detect_deterministic` + `statistical_triage` over sampled source columns; findings surface in `ValidationResult.leakage_flags` with class/severity/action/evidence — proven by `tests/test_phase_validate_leakage.py` (11 tests incl. clean-data false-positive guard) |

### Experience store hardening (M5)

| ID | Finding | Resolution |
|----|---------|-----------|
| C-6 | `identity_hash` excluded `engine_commit` and `schema_fingerprint`: a newer engine silently **overwrote** older records | Both participate in identity; storage key is now the identity hash so versions coexist (`test_write_does_not_overwrite_across_engine_versions`) |
| C-7 | `all_cases()` crashed on a single corrupt JSON row (JSONDecodeError) | Corrupt rows are skipped (`_safe_row_to_case`); enforced by `test_corrupt_row_does_not_break_store_reads` |
| C-8 | Records with unknown JSON keys (written by a newer engine) raised TypeError on read | Unknown keys dropped on load; forward-compat enforced by test |

### Intake / reliability fixes (M6)

| ID | Finding | Resolution |
|----|---------|-----------|
| C-9 | A single CSV field >128 KB escaped as a raw `_csv.Error` (ungraceful huge-row DoS) | Field limit raised to a bounded 1 MB; oversized fields counted as malformed and scanning continues |
| C-10 | Dialect detection: for ragged CSVs a degenerate tab parse (stability 1.0) outranked the real comma delimiter, marking **all** rows malformed | Delimiters that never occur in the data are score-capped; ragged comma files now parse correctly |

### Experiment lineage (M8, MLflow decision)

The existing `ExperimentTracker` already backends MLflow/W&B/local behind
one interface (`backend: "none" \| "mlflow" \| "wandb" \| "local"`). Adding a
second tracking system would duplicate lineage, not improve it. Instead the
**recorded lineage was deepened** so every backend captures the mission
fields: engine commit (`git rev-parse`, `FIAE_ENGINE_COMMIT` override),
python version, dataset fingerprint, split fingerprint, feature portfolio
(+ proposals with params), metrics, and a portfolio-lineage JSON artifact.
`benchmarks/bench_core.py` provides the size/shape/memory benchmark suite
(1K/100K/1M tiers, narrow/wide/high-cardinality/missing-heavy shapes,
tracemalloc peak memory).
