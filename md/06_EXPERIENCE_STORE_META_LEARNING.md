# 06 — Experience Store, Meta-Learning, Failure Memory, and Retrieval

## Purpose
FIAE must not start every dataset from zero. The Experience Store keeps context, actions, outcomes, cost, and failure reasons from prior runs. Historical knowledge changes search order only; current-dataset experiments remain authoritative.

## Case schema
```text
Case
├── identity: case_id, dataset_fingerprint, schema_fingerprint, engine_commit
├── context
│   ├── task, target semantics, prediction-time semantics
│   ├── dataset meta-features
│   ├── group/time structure
│   ├── data-quality profile
│   └── user resource/deployment constraints
├── action
│   ├── validation plan
│   ├── feature graph / portfolio
│   ├── model family + hyperparameters
│   ├── ensemble plan
│   └── fidelity/resource budget
├── result
│   ├── primary/secondary metrics
│   ├── fold distribution and generalization gap
│   ├── calibration/threshold results
│   └── robustness
├── cost
│   ├── wall/CPU time
│   ├── peak RAM
│   ├── model bytes
│   ├── materialization bytes
│   └── inference latency + batch size
└── diagnosis
    ├── success tags
    ├── failure tags
    ├── leakage findings
    ├── instability
    └── explanation/evidence refs
```

## Dataset meta-feature families
### Shape
- log rows; log columns; row/column ratio; observed cells; file/source size estimate.
### Type composition
- numeric, integer-like, categorical, high-cardinality, boolean, datetime, text, identifier and mixed fractions.
### Missingness
- global missing ratio; column mean/max; counts above 10/30/50/90%; missingness co-occurrence summaries.
### Numeric distribution
- zero/negative fractions; skew/kurtosis summaries; robust ranges; unique ratios; outlier-rate summaries.
### Categorical structure
- cardinality quantiles; rare-level fraction; dominant-level ratio; entropy summaries; unseen-level risk.
### Target
- task; cardinality; class distribution; imbalance; entropy; regression dispersion/heavy-tail signals.
### Relationship/meta structure
- routed correlation/association summaries; redundancy cluster counts; group repetition; temporal continuity.
### Runtime/source
- seekable/compressed/local/remote; metadata availability; estimated scan/decode/materialization cost.

## Retrieval hierarchy
### R0 — hard compatibility
Reject prior cases with incompatible task, target semantics, mandatory time/group structure, or impossible deployment constraints.

### R1 — cheap buckets
Bucket by task, row scale, width, categorical/high-cardinality regime, time-series yes/no, text yes/no, severe imbalance yes/no.

### R2 — weighted meta-feature distance
```text
distance(q,c) =
    Σ family_weight[f] * robust_distance(q[f], c[f])
```
No noisy high-dimensional family may dominate the distance simply because it has more dimensions.

### R3 — learned ranking (optional)
Only after enough *distinct datasets* exist, train a ranking/embedding model to predict transfer utility. Evaluate by holding out whole datasets, never random trials from the same dataset.

## Retrieval output
```text
RetrievalPrior
├── feature-family priors
├── transformation priors
├── model-family priors
├── hyperparameter-region priors
├── split priors
├── ensemble priors
├── cost priors
├── failure priors
└── confidence / OOD score
```

## Bayesian smoothing
Never trust 1/1 historical success as 100%.
```text
posterior = (successes + alpha) / (attempts + alpha + beta)
```
Default starting prior can use alpha=1, beta=1 unless benchmarked otherwise.

## Failure taxonomy
```text
LEAKAGE_CONFIRMED
LEAKAGE_SUSPECTED
VALIDATION_INVALID
TARGET_ENCODER_NOT_CROSSFIT
TEMPORAL_LOOKAHEAD
GROUP_CONTAMINATION
OVERFIT_HIGH_VARIANCE
UNDERFIT_HIGH_BIAS
NO_INCREMENTAL_GAIN
REDUNDANT
UNSTABLE_ACROSS_FOLDS
RESOURCE_RAM
RESOURCE_TIME
RESOURCE_LATENCY
RESOURCE_MODEL_SIZE
INVALID_DOMAIN
NUMERIC_OVERFLOW
TOO_MANY_LEVELS
ENSEMBLE_REDUNDANT
ENSEMBLE_NEGATIVE_GAIN
HPO_NONCONVERGENT
GRADIENT_EXPLODING
GRADIENT_VANISHING
DATA_SHIFT
CODEGEN_MISMATCH
SERIALIZATION_FAILURE
```

## Negative-transfer guard
1. Retrieve historical priors.
2. Apply them only to search order.
3. Run a cheap current-dataset probe.
4. Compare retrieved strategy against a generic baseline.
5. Down-weight prior when it fails.
6. Never promote merely because history says it worked.

## Cost predictors
Experience can train conservative predictors for:
- trial time,
- peak RAM,
- inference latency,
- feature materialization cost,
- probability of resource failure.
Start with robust rules/simple regressors. Learned predictors become active only if they beat deterministic baselines and are conservatively calibrated.

## Case write-back
```text
trial ends
→ validate TrialResult schema
→ attach dataset/split/feature/model provenance
→ diagnose success/failure
→ freeze immutable Case
→ atomic persistence
→ update smoothed sufficient statistics
→ emit CASE_WRITTEN
```

## Duplicate identity
```text
hash(
  dataset_fingerprint,
  split_fingerprint,
  feature_graph_hash,
  model_spec_hash,
  hyperparameters,
  seed
)
```
Repeated seeds/runs can remain for variance estimation but must not be mistaken for distinct-dataset evidence.

## OOD query
If retrieval confidence is low:
1. mark UNKNOWN_REGIME,
2. rely more on generic search,
3. preserve exploration quota,
4. optionally create offline research escalation,
5. write the new result back after verification.

## Local storage
```text
.fiae/
├── experience.sqlite
├── cases/
├── profiles/
├── artifacts/
└── cache/
```
Large arrays/artifacts remain files/Parquet/object storage, not giant SQLite blobs.

## Retrieval pseudocode
```text
function retrieve_priors(query, K):
    compatible = hard_filter(case_index, query)
    if empty(compatible):
        return generic_priors(confidence=LOW)

    bucketed = cheap_bucket_rank(compatible, query)
    candidates = take_top(bucketed, R2_LIMIT)

    parallel for case in candidates:
        case.distance = family_weighted_meta_distance(query.meta, case.meta)

    nearest = top_k(candidates, K)
    priors = bayesian_aggregate(nearest)
    confidence = retrieval_confidence(nearest)

    if confidence < OOD_THRESHOLD:
        priors = blend(priors, generic_priors(), favor_generic=True)

    return priors
```

## Meta-learning validation
Dataset identity is the unit of separation:
```text
TRAIN meta-learner: datasets A..N
TEST meta-learner: completely unseen datasets
```
Additional tests: leave-domain-out, leave-scale-regime-out, chronological case accumulation.

## Acceptance criteria
The layer stays enabled only when it measurably reduces one or more of:
- candidate evaluations,
- full fits,
- time-to-quality,
- resource failures,
without unacceptable quality degradation or negative transfer.
