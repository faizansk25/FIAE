# System Requirements and Hard Invariants

> **Purpose:** Turn design intent into enforceable engineering requirements.

**Status:** Normative specification for one continuously evolving FIAE codebase. These documents describe implementation gates, not user-facing versions.


## Functional requirements

### FR-001 Source abstraction
Core operates on a `DataSourceAdapter`, not direct CSV assumptions.

### FR-002 Bounded profiling
Profiling respects time/row/byte budgets and records whether statistics are exact or approximate.

### FR-003 Semantic schema
Physical dtype and semantic role are separate outputs.

### FR-004 Problem formulation
Task inference exposes confidence and reasons and allows explicit override.

### FR-005 Validation before learned features
A validation plan exists before any target-aware or learned-statistic transformation executes.

### FR-006 Leakage levels
Findings distinguish informational, review-required, and hard-reject.

### FR-007 Feature operation registry
Each operation declares accepted semantic types, output type, fit requirements, leakage class, cost estimator, transform algorithm, serialization contract, and inference requirements.

### FR-008 Feature DAG
Feature candidates are canonical DAG nodes with deterministic signatures and lineage.

### FR-009 Progressive search
Search supports fidelity changes in rows, folds, iterations, feature depth, and HPO budget.

### FR-010 Model registry
Model families declare capabilities, search spaces, resource behavior, inference properties, and early-stopping support.

### FR-011 OOF discipline
Stacking and target-aware encodings rely on out-of-fold/cross-fit predictions/statistics.

### FR-012 Constraint-aware final selection
Hard latency/RAM/time/size requirements filter candidates before preference scoring.

### FR-013 Experience memory
Every valid success, prune, and failure can become a reusable case.

### FR-014 Code generation
Generated code is derived from structured pipeline IR and verified against internal predictions.

### FR-015 Observability
All material run/stage/trial/feature/decision states are queryable.

## Non-functional requirements

### NFR-001 Local first
CPU-only single-machine operation is first-class.

### NFR-002 Low mandatory dependency count
Optional capabilities may require optional packages; core import cannot require cloud/distributed frameworks.

### NFR-003 Bounded RAM
No architecture step assumes the full dataset fits in memory.

### NFR-004 Deterministic identifiers
Dataset, split, feature, portfolio, trial, model, artifact, and run identifiers are stable under their declared inputs.

### NFR-005 Reproducibility
Record seeds, environment, dependency versions, code hash, dataset fingerprint, split fingerprint, parameters, and artifacts.

### NFR-006 Recoverability
Stage checkpoints are resumable when semantic correctness is preserved.

### NFR-007 Privacy
Raw values and secrets are not logged by default.

### NFR-008 Minimal observability overhead
High-frequency telemetry is buffered/batched and cannot dominate model runtime.

## Hard invariants

1. Never fit any learned preprocessing on final holdout.
2. Never use final holdout metrics to choose features, hyperparameters, model family, ensemble weights, calibration, threshold, or early-stopping policy.
3. Exact target copy is forbidden.
4. A feature proven unavailable at prediction time is forbidden.
5. Target encoding uses cross-fitting.
6. Target lags require explicit label-availability semantics.
7. Random row CV is forbidden for future forecasting.
8. Entity leakage is prevented when multiple rows from one entity create dependence.
9. Feature graph is acyclic.
10. Domain errors are explicit; they never silently become misleading numeric values.
11. Failed candidates do not receive valid scores.
12. Resource estimates are checked before expensive scheduling.
13. Cache keys include split/fold context for learned transforms.
14. Parallel workers cannot exceed scheduler CPU/RAM tokens.
15. Ensemble members require unbiased predictions.
16. Generated code must pass prediction-equivalence tests.
17. Audit events cannot be dropped.
18. CLI/dashboard cannot implement independent modeling logic.
19. External AutoML engines are optional challengers/adapters.
20. Benchmark claims include baseline, hardware, data, seed policy, time budget, and metric.

## Error taxonomy

```text
DATA_FORMAT_ERROR
SCHEMA_AMBIGUITY
SOURCE_CHANGED_DURING_RUN
TARGET_MISSING
TARGET_INVALID
TASK_AMBIGUOUS
SPLIT_INVALID
LEAKAGE_CONFIRMED
PREDICTION_TIME_UNAVAILABLE
FEATURE_PRECONDITION_FAILED
FEATURE_DOMAIN_ERROR
FEATURE_CARDINALITY_EXPLOSION
RESOURCE_PRECHECK_FAILED
TRIAL_TIMEOUT
TRIAL_OOM
MODEL_FIT_FAILED
METRIC_UNDEFINED
HPO_INVALID
ENSEMBLE_INVALID
CALIBRATION_INVALID
CODEGEN_MISMATCH
REPRODUCIBILITY_MISMATCH
ARTIFACT_CORRUPTION
CANCELLED
```

Each error contains:
- code;
- severity;
- component;
- run/stage/trial/feature IDs;
- recoverable flag;
- retry policy;
- human reason;
- machine-readable evidence.

## Resource hierarchy

```text
machine safety reserve
  -> user hard limits
    -> run budget
      -> stage budget
        -> worker budget
          -> trial budget
            -> feature-materialization budget
```

## Dependency tiers

### Tier 0
Python standard library.

### Tier 1
NumPy, scikit-learn, joblib; Polars optional/preferred for accelerated tabular execution.

### Tier 2 optional learners/search
LightGBM, XGBoost, CatBoost, Optuna.

### Tier 3 optional challengers
AutoGluon, H2O Driverless AI integration, DataRobot and SageMaker service adapters.

### Tier 4 optional operations
MLflow, OpenTelemetry exporters, PostgreSQL, object storage, distributed schedulers.
