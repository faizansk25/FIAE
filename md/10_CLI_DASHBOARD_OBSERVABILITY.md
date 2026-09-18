# 10 — CLI, Dashboard, Logging, Experiments, Audit, and Telemetry

## Architecture
```text
CLI ─┐
UI  ─┼→ Control Plane → Coordinator → Workers
SDK ─┘                         │
                              ↓
                           Event Bus
                 ┌────────────┼────────────┐
                 ↓            ↓            ↓
              logs       experiments     telemetry/audit
```
CLI and dashboard never implement separate ML logic.

## Run identity
Every durable operation has run_id. Nested work has stage_id, optional trial_id and feature_id.

## CLI
```text
fiae inspect SOURCE
fiae analyze SOURCE --target Y
fiae plan SOURCE --target Y --time-budget 30m --memory 8GiB
fiae run SOURCE --target Y ...
fiae status RUN
fiae logs RUN --follow
fiae features RUN
fiae models RUN
fiae leaderboard RUN
fiae pareto RUN
fiae explain RUN
fiae compare RUN_A RUN_B
fiae artifacts RUN
fiae export RUN --project out/
fiae cancel RUN
fiae resume RUN
fiae dashboard
fiae doctor
```
Commands support machine-readable `--json` where applicable.

## Dashboard pages
1. Overview: active/completed/failed, CPU/RAM/workers.
2. Run workspace: stage, budget, warnings, current best/frontier.
3. Dataset Intelligence: schema, semantics, missingness, cardinality, target, quality.
4. Leakage & Validation: split diagram, cutoffs/groups, findings.
5. Feature Funnel: generated → cheap-gate → probe → CV → portfolio → final.
6. Feature Detail: expression, lineage, exact semantics, prior, metrics, cost, decision.
7. Model Search: family, hyperparameters, fidelity, metrics, train/valid gap, resource cost.
8. Ensemble: OOF quality, diversity, marginal gain, weights, cost, stack guard.
9. Pareto: quality/latency/RAM/size/time/interpretability.
10. Timeline: run → stage → trial spans.
11. Audit: decisions and evidence.
12. Artifacts: reports, model, manifest, verified code.

## Four separate streams
### Application logs
Operational/debug events.
### Experiment records
Parameters, metrics, artifacts, input fingerprints.
### Telemetry
System performance metrics/traces.
### Audit
Durable consequential decisions and evidence.

## EventEnvelope
```json
{
  "event_id":"...",
  "timestamp":"...",
  "run_id":"...",
  "stage_id":"...",
  "trial_id":null,
  "feature_id":null,
  "level":"INFO",
  "component":"feature_search",
  "event_type":"FEATURE_REJECTED",
  "schema_version":1,
  "payload":{}
}
```

## Event types
Run: RUN_CREATED, RUN_STARTED, RUN_CANCELLED, RUN_FAILED, RUN_COMPLETED.
Data: SOURCE_OPENED, PROFILE_STARTED, PROFILE_COMPLETED, DATA_QUALITY_FINDING.
Problem: TASK_INFERRED, VALIDATION_PLAN_CREATED, LEAKAGE_FINDING.
Feature: FEATURE_PROPOSED, DUPLICATE, PRUNED, MATERIALIZED, EVALUATED, ACCEPTED, REJECTED.
Trial: QUEUED, STARTED, CHECKPOINT, PRUNED, FAILED, COMPLETED, PROMOTED.
Ensemble: CANDIDATE, MEMBER_ADDED, MEMBER_REJECTED, STACKING_DISABLED.
Codegen: STARTED, TEST_FAILED, PARITY_PASSED, EXPORT_COMPLETED.

## Privacy-safe logging
Do not persist raw names, emails, phone numbers, tokens, credentials, arbitrary text cells by default. Store schema, safe statistics, hashes/fingerprints, IDs and sanitized errors.

## Local persistence
```text
run/
├── run.json
├── events.jsonl
├── decisions.jsonl
├── metrics/
├── reports/
└── artifacts/
```
SQLite indexes searchable metadata; artifacts remain files.

## Async writer
```text
worker event → bounded queue → batched durable append → SQLite index → live subscribers
```
Critical/audit events can require synchronous durable flush.

## Crash recovery
1. discover interrupted RUNNING runs,
2. inspect append log/checkpoint,
3. never infer unfinished trial success,
4. mark INTERRUPTED or RESUMABLE,
5. verify source/config before resume.

## Experiment compatibility
Internal schema maps cleanly to run/params/metrics/artifacts/tags so an MLflow exporter can be optional rather than mandatory.

## Telemetry compatibility
Internal spans/metrics can export to OpenTelemetry. Core runs without collector.

## Metrics
Engine: active runs/trials, queue, utilization, cache hit.
Data: rows/sec, bytes/sec, profile latency.
Feature: candidates/sec, prune rate, accepted rate, bytes.
Model: trials/sec, prune rate, duration, OOM.
Quality: best metric, best utility, improvement per compute.

## Trace
```text
run
├── profile
├── task_analysis
├── leakage
├── feature_search
├── model_search
├── ensemble
├── final_evaluation
└── codegen
```

## Audit record
```text
DecisionRecord
├── subject
├── action
├── reason
├── evidence refs
├── rule refs
├── metrics/constraint snapshot
├── override policy
└── timestamp
```

## Dashboard transport
Durable store is truth. SSE is sufficient for one-way live updates; WebSocket only if bidirectional live control needs it. Reconnect uses cursor/history.

## Security
Local dashboard binds loopback by default. Remote exposure requires authentication/authorization/network protection.

## Completion view
A completed run must answer:
- what problem,
- what validation,
- what leakage,
- selected/rejected features and why,
- models tested,
- winner and why,
- final result,
- time/RAM/latency,
- verified code/artifacts,
- limitations.
