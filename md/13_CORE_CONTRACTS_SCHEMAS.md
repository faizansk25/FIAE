# 13 — Core Contracts, Typed Schemas, IDs, and Serialization

## Principle
Modules communicate through typed contracts; no undocumented mutable dictionaries as architecture.

## IDs
Stable opaque identifiers: run, stage, trial, feature, portfolio, decision, event, artifact, dataset fingerprint and split fingerprint.

## DataSourceSpec
```text
kind
uri_or_path
format
seekable
compression
credentials_ref
options
```

## ColumnProfile
```text
name
physical_dtype
semantic_type
null_fraction
distinct_estimate
distinct_ratio
statistics
warnings
```

## DatasetProfile
```text
dataset_fingerprint
rows_observed / estimated
columns
column_profiles
meta_features
source_cost
quality_findings
```

## ProblemDefinition
```text
task
target
positive_class
prediction_time_semantics
group_keys
time_key
horizon
metrics
ambiguities
```

## ConstraintSpec
```text
wall_time_s
memory_bytes
latency_s
latency_batch_size
model_bytes
interpretability
quality_target
worker_limit
```

## ValidationPlan
```text
strategy
folds
group_key
time_key
cutoffs
final_holdout
nested_policy
seed
split_fingerprint
```

## LeakageFinding
```text
finding_id
subject
severity
type
evidence
action
override_allowed
```

## FeatureNode
```text
feature_id
operator
inputs
params
fit_scope
target_permission
time_semantics
null_policy
cost_hint
lineage_hash
```

## FittedStateRef
```text
state_id
feature_id
split_id
artifact_ref
state_hash
```
Split identity prevents cross-fold state reuse.

## FeaturePortfolio
```text
portfolio_id
feature_ids
graph_hash
estimated_cost
verified_metrics
```

## ModelSpec
```text
family
backend
hyperparameters
seed
capabilities
resource_hints
```

## FidelitySpec
```text
row_fraction
fold_count
iterations
feature_fraction
stage
```

## TrialSpec
```text
trial_id
portfolio_id
model_spec
fidelity
constraints
timeout_s
resource_reservation
```

## TrialResult
```text
trial_id
status
metrics
fold_metrics
train_metrics
resource_measurements
warnings
artifacts
diagnosis
```

## EnsembleSpec
```text
member_trial_ids
weights
stacker
OOF refs
latency estimate
size estimate
stack-overfit guard result
```

## MetricValue
```text
name
value
direction = maximize|minimize
split
fold
aggregation
unit_optional
```

## ResourceMeasurement
```text
wall_time_s
cpu_time_s
peak_rss_bytes
disk_bytes
model_bytes
inference_seconds
inference_batch_size
hardware_context
```
Latency without batch size/hardware context is incomplete.

## DecisionRecord
```text
decision_id
type
subject
action
evidence_refs
rule_refs
metric/constraint snapshot
timestamp
```

## EventEnvelope
```text
event_id
timestamp
run_id
stage_id
trial_id?
feature_id?
level
component
event_type
schema_version
payload
```

## ArtifactRef
```text
artifact_id
kind
path_or_uri
hash
bytes
producer_run
trust_level
```

## RunManifest
```text
run_id
source_fingerprint
config_hash
engine_build
environment
selected_pipeline_hash
artifacts
completion_state
```

## PredictionContext
```text
decision_timestamp
observation_cutoff
horizon
entity_keys
allowed_source_lag
```
Temporal operators receive this explicitly.

## FIAEError
```text
code
category
retryable
safe_message
internal_cause_ref
stage_id
subject_id
```

## Canonical serialization
1. stable key order,
2. stable enum values,
3. exclude ephemeral timestamps from semantic hashes,
4. normalize permitted number representation,
5. UTF-8 encode,
6. content hash.

## Immutability boundaries
Freeze after validation plan, final feature portfolio, final model/ensemble, final-holdout opening and artifact manifest. Changing a frozen object produces a new identity/decision.

## Schema evolution
Durable records carry `schema_version`. Reader migrates known old schemas and safely rejects unknown incompatible newer schemas.
