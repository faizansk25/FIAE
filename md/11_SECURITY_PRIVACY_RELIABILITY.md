# 11 — Security, Privacy, Reliability, and Trust Boundaries

## Trust model
Treat raw user data, filenames, database/API text, LLM output, generated code, custom plugins and imported serialized artifacts as untrusted unless explicitly verified.

## Trust boundaries
```text
external source
→ source adapter
→ normalized data contract
→ deterministic FIAE core
→ optional ML backends
→ generated artifacts
→ deployment
```
Research/LLM reasoning is advisory, not evidence authority.

## Data minimization
Prefer metadata, pushed-down queries and needed columns/samples. Do not copy an entire remote dataset if a safe source-side operation is enough.

## Secrets
Secrets come from environment/secret provider; never write them into logs, generated config, reports or research prompts.

## PII
Semantic detection can flag likely sensitive columns but is imperfect. Default:
- no raw-cell logging,
- no raw dashboard preview,
- hash identifiers if linkage required,
- explicit opt-in for any raw sample display.

## Generated-code sandbox
Before generated code executes:
1. controlled working directory,
2. timeout,
3. resource limits where available,
4. network disabled unless required,
5. controlled environment,
6. stdout/stderr capture,
7. verify produced paths/artifacts.

## Plugin permissions
Every custom FeatureOperator declares input/output schema, fit requirement, target access, temporal/group access, determinism, resource estimate, version and permissions.

## Target access
```text
P0 no target
P1 target through cross-fit API only
P2 historical target up to explicit temporal cutoff
P3 evaluator-only target access
```
Arbitrary plugins never receive unrestricted target by default.

## Path/file safety
Normalize export paths, prevent traversal, use atomic manifests, verify hashes, never overwrite arbitrary host files without explicit policy.

## Unsafe deserialization
Avoid untrusted pickle-like artifacts. If unavoidable for a backend, only load artifacts with trusted producer/run provenance and verified hash; document trust level.

## Dependency security
No unattended `pip install <LLM-proposed-package>`. Optional dependencies are known adapters/groups with recorded versions.

## Denial-of-resource protections
Bound:
- maximum line/field size,
- candidate count,
- feature depth,
- pending queue,
- text length processed in fast mode,
- time,
- RAM,
- temporary disk.

## Audit integrity
Consequential records are append-oriented. Corrections reference old record; do not silently rewrite history.

## Reliability states
```text
CREATED → PLANNING → RUNNING
                  ↘ COMPLETED
                  ↘ FAILED
                  ↘ CANCELLED
                  ↘ INTERRUPTED
```
A crash cannot become COMPLETED automatically.

## Checkpoints
Checkpoint only stable boundaries:
- profile complete,
- validation frozen,
- feature stage complete,
- HPO stage complete,
- ensemble frozen,
- final evaluation complete,
- codegen verified.

## Resume
1. verify source fingerprint,
2. verify config/build compatibility,
3. verify checkpoint hash,
4. recover durable state,
5. invalidate unsafe caches,
6. resume after last complete boundary.
Source change usually means new run.

## Determinism
Record seeds but state reproducibility level. Bitwise reproducibility may not hold across hardware/backend/nondeterministic kernels.

## Overrides
Normal search/resource options may be overridden. Confirmed leakage and final-holdout isolation are hard invariants by default.

## Incident record
Capture safe error code, component/stage, resource snapshot, checkpoint, retryability and internal traceback reference. Redact sensitive values before persistence.
