# 12 — Testing, Benchmarks, Ablations, and Research Gates

## Principle
`code executed` != `accepted design`.

## Test layers
1. unit,
2. property,
3. integration,
4. differential/reference,
5. end-to-end,
6. performance benchmark,
7. adversarial/failure injection.

## Every FeatureOperator tests
- nulls,
- dtype,
- empty/constant data,
- extreme values,
- invalid domain,
- deterministic behavior where promised,
- fit/transform separation,
- fitted-state serialization,
- fold safety if target/data fitted,
- temporal cutoff safety,
- group boundary safety.

## Shift/lag properties
### Row lag k
- row i equals source i-k within entity,
- first k rows per entity missing/default,
- changing any future row cannot alter earlier lag output.
### Time offset
- selected observation timestamp <= permitted lookup time,
- entity boundaries preserved.
### Rolling `[T-W,T)`
- current row excluded,
- future rows cannot alter historical output,
- sorting/duplicate-time policy explicit.

## Leakage adversarial suite
1. exact target copy,
2. renamed target copy,
3. target + tiny noise,
4. post-outcome status,
5. target encoding fit on all data,
6. future temporal aggregate,
7. same entity across random split,
8. feature selection before split,
9. global imputer/scaler fit before split.

Expected: hard reject where proven; risk warning where only statistical; correct fold-safe pipeline.

## Validation tests
- stratification where valid,
- no group overlap in GroupKFold-style strategy,
- chronological cutoffs for time,
- inner search never sees outer-test fold,
- final holdout never appears in selection state.

## Profiler tests
Quoted newlines/commas, alternate delimiter, BOM, empty/ragged rows, mixed types, all-null, dates, high cardinality, nonseekable input, bounded fast mode.

## Feature search
- canonical duplicates collapse,
- depth/candidate budget,
- invalid ops rejected,
- priors affect ordering not truth,
- pruning actually reduces work,
- promoted features have recorded evidence.

## HPO
- conditional space valid,
- pruner checkpoints,
- timeout normalized,
- worker failure isolated,
- hard resource preflight,
- promotion factor/schedule.

## Ensemble
- OOF one prediction per development row,
- no row predicted by model fitted on that row for OOF,
- final holdout excluded from weight fit,
- redundant candidate can be rejected,
- stacking guard can disable stack.

## Codegen
- generated import,
- tests,
- feature parity,
- prediction parity,
- manifest hashes,
- clean-process inference,
- dependency minimality.

## Observability
- run ID in durable events,
- audit durable,
- privacy fixture leaks zero prohibited raw values,
- queue backpressure,
- crash replay.

## Performance dimensions
- profile rows/sec,
- feature proposal/sec,
- materialization throughput,
- feature evaluation/sec,
- model trial throughput,
- peak RAM,
- event/dashboard overhead,
- export time.

## Quality benchmark
Across multiple datasets/regimes report:
- primary metric,
- regret versus references,
- time-to-quality,
- total compute,
- peak memory,
- failures.

## Reference challengers
When accessible:
- sklearn baseline,
- AutoGluon,
- OpenFE,
- FLAML,
- H2O Driverless AI,
- DataRobot/SageMaker with exact configuration/cost stated.
Never claim fairness if environments/budgets differ materially.

## Ablation
Disable one:
- experience retrieval,
- failure memory,
- feature search,
- coarse pruning,
- multi-fidelity,
- cost-aware ordering,
- ensembles,
- cache,
- semantic typing.
Measure quality/time/memory impact.

## “100x” rule
No marketing shortcut. Example accepted statement:
```text
baseline full-equivalent evaluations = 100000
FIAE full-equivalent evaluations     = 1000
quality delta                        >= -defined tolerance
hardware/data/seed policy            documented
```
Could instead prove 100x compute/time under an explicitly fixed quality target.

## Research gate
```text
Question
Current behavior
Primary/official research
Hypothesis
Alternatives
Minimal challenger implementation
Correctness tests
Equal-budget benchmark
Ablation
Result
Decision
Rollback trigger
```

## Gate states
```text
PROPOSED
IMPLEMENTED
CORRECTNESS_PASSED
BENCHMARKED
ACCEPTED | REJECTED
```
These are gates, not product versions.

## Statistical reporting
Across datasets/seeds: mean/median, distribution, wins/ties/losses, total compute and uncertainty where appropriate. Do not report only winning examples.

## Meta-learning benchmark
Hold out entire datasets. Compare generic search vs deterministic nearest-history vs learned priors. Track negative-transfer rate.

## Resource predictor benchmark
Track predicted vs actual time/RAM and especially catastrophic underprediction rate.

## Failure injection
Worker crash, disk full, cancel, timeout, corrupted checkpoint, source change, invalid artifact, slow event writer. Verify terminal state and recoverability.

## Completion
A run is COMPLETE only with valid problem/validation, final candidate evidence, artifact manifest, audit summary and explicit export status.
