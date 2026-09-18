# 08 — Parallel Execution, Resource Scheduler, Lazy Materialization, and Cache

## Purpose
Parallelism is useful only when it lowers wall-clock time without multiplying memory, repeated I/O, or nested-thread contention.

## Work classes
```text
IO_BOUND
NATIVE_CPU_RELEASES_GIL
PYTHON_CPU_BOUND
MEMORY_HEAVY
ACCELERATOR
REMOTE
```
Routing:
- I/O: threads/async where useful.
- Native vectorized code: threads may be efficient.
- Python CPU-bound: processes.
- Memory-heavy: deliberately low concurrency.
- Accelerator: device queue.
- Remote: bounded concurrency with backpressure.

## Resource tokens
Scheduler owns CPU slots, RAM reservation, optional device slots, disk scratch quota and remote concurrency quota.
```text
estimate → reserve → execute → measure → release → update cost model
```

## Memory admission
```text
estimated_peak =
    base_data_live
  + candidate_feature_bytes
  + model_training_workspace
  + backend_overhead
  + safety_margin
```
If `estimated_peak + currently_reserved > allowed_ram`, shrink fidelity, stream, delay, or reject.

## Lazy Feature DAG
FeatureNode remains a recipe until required:
```text
base columns
→ logical DAG
→ canonicalize
→ dead-node/duplicate pruning
→ common-subexpression sharing
→ cost estimate
→ materialize survivors
```

## DAG optimization passes
1. dead node elimination,
2. duplicate expression elimination,
3. constant folding,
4. projection pushdown,
5. safe filter/predicate pushdown,
6. common subexpression elimination,
7. operator fusion,
8. fitted-state sharing only when split-safe,
9. materialization boundary placement,
10. memory estimate.

## Cache key
```text
hash(
 source_fingerprint,
 split_id,
 canonical_feature_node,
 fitted_state_fingerprint,
 implementation_build
)
```
Fold-fitted target encoders/scalers must include fold/split identity.

## Cache tiers
- L0 in-process memory,
- L1 mmap/shared local representation when safe,
- L2 local disk,
- L3 optional artifact/object store.
Evict based on size, recompute cost, recency and dependency count.

## Data locality
Preferred order:
1. metadata-only,
2. source-side query/aggregation,
3. read required columns only,
4. stream batches,
5. full copy only if necessary.

## CSV
Because CSV is not self-describing:
- robust dialect/header/type sniffing,
- multiple sample positions for seekable files,
- bounded fast mode,
- exact scan only when required.

## Parallel feature evaluation
Good:
```text
shared source/sample → independent candidate batches
```
Bad:
```text
every worker re-reads the full source
```

## Nested parallelism guard
If there are W outer processes:
```text
inner_threads ≈ max(1, floor(total_cpu / W))
```
Avoid 8 processes each launching 16 BLAS/model threads.

## Backpressure
Bound all queues.
When candidate producer reaches queue high-water mark, pause production. Store compact IDs/specifications instead of huge materialized frames in queues.

## Priority
Possible score:
```text
expected_improvement / predicted_time
- memory_risk
- failure_risk
+ exploration_bonus
```

## Cancellation
1. stop scheduling new work,
2. cooperative cancellation checks between batches/folds/checkpoints,
3. grace period,
4. terminate stubborn workers if required,
5. flush critical audit events,
6. release resource tokens,
7. mark partial artifacts incomplete.

## Timeout
```text
trial_timeout = min(
 configured_trial_timeout,
 stage_budget_remaining,
 run_budget_remaining
)
```
Timeout is a normalized TrialResult status, not a run crash.

## Retry
Retry only transient failures. Do not repeat unchanged deterministic invalid-domain, confirmed-leakage, or preflight-OOM trials.

## Async event path
```text
worker → compact EventEnvelope → bounded queue → batch writer
                                      ├→ JSONL
                                      ├→ SQLite index
                                      └→ live subscribers
```
Critical decisions can request durable flush.

## Streaming capability flags
Every operator declares:
```text
streaming_capable
requires_fit
requires_sort
requires_group_state
requires_full_matrix
```

## Cost measurement
Measure wall time, CPU time, peak RSS where possible, output bytes, temporary bytes, rows/sec, prediction latency with batch size/hardware context.

## Scheduler pseudocode
```text
while run active:
    finished = executor.poll()
    for job in finished:
        release(job.reservation)
        persist(job.result)
        cost_model.update(job.result)

    ready = graph.ready_jobs()
    for job in priority_sort(ready):
        estimate = conservative_cost(job)
        if violates_hard_limit(estimate):
            prune(job, RESOURCE_PREFLIGHT)
        elif tokens_available(estimate):
            reserve(job, estimate)
            executor.submit(job)

    if nothing running and nothing ready:
        break
    wait()
```

## Execution interface
```text
Executor.submit(job)
Executor.poll()
Executor.cancel(job)
Executor.shutdown()
```
Implement Inline, LocalThread and LocalProcess first. Distributed frameworks remain optional adapters.

## Acceptance
Parallel implementation must show:
- speedup on eligible workload,
- bounded RAM,
- no leakage/result-contract change,
- clean cancellation,
- no runaway oversubscription.
If parallel is slower for a regime, scheduler should choose serial.
