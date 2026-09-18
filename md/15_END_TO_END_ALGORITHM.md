# 15 — End-to-End FIAE Algorithm: Canonical Execution Procedure

This is the detailed canonical execution flow. It is a specification, not a claim that every step is implemented today.

Backend code may optimize these steps only if correctness, leakage, resource and evidence contracts remain equivalent.


# PHASE — BOOTSTRAP

## Step 0001
**Action:** Create run ID and immutable initial configuration snapshot.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0002
**Action:** Capture FIAE build/commit identity and runtime environment metadata.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm entity/order/cutoff semantics; future observations must not affect past features.
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0003
**Action:** Initialize durable run directory, event stream, SQLite metadata store, resource monitor and cancellation token.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- Use conservative resource estimate and enforce hard limits before expensive execution.
- Persist correlation IDs and sanitize raw sensitive values.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0004
**Action:** Validate requested operation mode: inspect, analyze, plan, run or export.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0005
**Action:** Normalize user constraints and units; reject impossible negative/zero limits.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0006
**Action:** Initialize local worker/resource limits from available hardware and user policy.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- Use conservative resource estimate and enforce hard limits before expensive execution.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0007
**Action:** Emit RUN_CREATED and persist RunManifest draft.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Persist correlation IDs and sanitize raw sensitive values.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.


# PHASE — SOURCE INTAKE

## Step 0008
**Action:** Create a DataSourceAdapter without loading the complete source.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0009
**Action:** Inspect cheap metadata: format, bytes, seekability, compression, schema metadata and row-group/table metadata if available.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0010
**Action:** Prefer source-provided schema for self-describing formats, but verify sampled values for obvious conflicts.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0011
**Action:** For CSV, detect dialect/header/physical types using bounded robust sampling; use multiple positions when seekable.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0012
**Action:** Normalize column names and apply duplicate-name policy.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0013
**Action:** Collect bounded representative samples under both row and time budgets.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm entity/order/cutoff semantics; future observations must not affect past features.
- Use conservative resource estimate and enforce hard limits before expensive execution.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0014
**Action:** Infer physical dtypes from parser/source metadata.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0015
**Action:** Infer semantic types separately: numeric, categorical, high-cardinality, boolean, datetime, text, identifier, mixed.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm entity/order/cutoff semantics; future observations must not affect past features.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0016
**Action:** Run streaming/bounded null, distinct, numeric and categorical summary statistics.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0017
**Action:** Detect basic data-quality issues: all-null, constant, extreme missingness, suspicious identifier, malformed rows, impossible type casts.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0018
**Action:** Estimate full scan time, materialization bytes and source access cost.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm entity/order/cutoff semantics; future observations must not affect past features.
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0019
**Action:** Build DatasetProfile, schema fingerprint and dataset fingerprint.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0020
**Action:** Emit PROFILE_COMPLETED with safe statistics only.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- Persist correlation IDs and sanitize raw sensitive values.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.


# PHASE — PROBLEM DEFINITION

## Step 0021
**Action:** Verify target presence for supervised runs and exclude target from ordinary feature inputs.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0022
**Action:** Analyze target physical/semantic type, missingness, cardinality and distribution.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0023
**Action:** Infer candidate task: binary classification, multiclass, regression, time-series/forecasting or ambiguous.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm entity/order/cutoff semantics; future observations must not affect past features.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0024
**Action:** If task remains ambiguous, surface ambiguity instead of silently forcing a task.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0025
**Action:** Resolve positive class or class order when required.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0026
**Action:** Choose candidate primary/secondary metrics consistent with task and imbalance.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0027
**Action:** Resolve prediction-time semantics: what timestamp/observation cutoff is available when prediction is made.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm entity/order/cutoff semantics; future observations must not affect past features.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0028
**Action:** Resolve entity/group keys from explicit configuration; inferred groups remain suggestions until accepted by policy.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0029
**Action:** Resolve temporal key and forecast horizon where applicable.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm entity/order/cutoff semantics; future observations must not affect past features.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0030
**Action:** Create immutable ProblemDefinition and emit TASK_INFERRED.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Persist correlation IDs and sanitize raw sensitive values.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.


# PHASE — VALIDATION AND LEAKAGE

## Step 0031
**Action:** Select validation family: stratified K-fold, K-fold, group-aware, time-ordered/backtest or custom.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- Confirm entity/order/cutoff semantics; future observations must not affect past features.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0032
**Action:** Create final untouched holdout policy before any fitted preprocessing.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0033
**Action:** Generate split IDs and split fingerprint.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0034
**Action:** Check exact target-copy columns and deterministic target derivations where detectable.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0035
**Action:** Check identifier-like columns for memorization risk.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0036
**Action:** Run statistical single-feature leakage triage; classify as suspicion, not automatic proof unless rule is deterministic.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0037
**Action:** Check prediction-time availability for temporal/post-outcome features.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm entity/order/cutoff semantics; future observations must not affect past features.
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0038
**Action:** Check entity/group contamination across proposed folds.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0039
**Action:** Check precomputed target encodings/aggregates for fold leakage when metadata/lineage exists.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0040
**Action:** Create hard-reject set for confirmed leakage and warning set for suspected leakage.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0041
**Action:** Construct fold-safe fitted-transform API and prohibit fit on final holdout.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0042
**Action:** Emit VALIDATION_PLAN_CREATED and individual LEAKAGE_FINDING records.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- Persist correlation IDs and sanitize raw sensitive values.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.


# PHASE — CONSTRAINT AND ASPECT ROUTING

## Step 0043
**Action:** Normalize wall-clock, RAM, model-size, inference-latency, interpretability and quality-target constraints.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- Record exact spec, seed/fidelity, fold metrics, wall time, RAM and status.
- Use conservative resource estimate and enforce hard limits before expensive execution.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0044
**Action:** Bind inference latency to an explicit batch size and hardware context.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Use conservative resource estimate and enforce hard limits before expensive execution.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0045
**Action:** Route only relevant diagnostics from the full aspect catalog.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0046
**Action:** Disable gradient diagnostics unless gradient-trained models are active.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Record exact spec, seed/fidelity, fold metrics, wall time, RAM and status.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0047
**Action:** Disable temporal feature families unless valid ordered temporal semantics exist.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm entity/order/cutoff semantics; future observations must not affect past features.
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0048
**Action:** Disable group aggregations unless an entity/group relationship exists.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0049
**Action:** Disable target-aware transforms unless cross-fit implementation is available.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0050
**Action:** Allocate provisional budgets to profiling, feature search, model search, ensemble and final verification.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- Record exact spec, seed/fidelity, fold metrics, wall time, RAM and status.
- Use conservative resource estimate and enforce hard limits before expensive execution.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0051
**Action:** Create RunPlan; plan mode may stop here after rendering the plan.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.


# PHASE — EXPERIENCE RETRIEVAL

## Step 0052
**Action:** Build dataset/task/constraint meta-feature query vector.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0053
**Action:** Hard-filter historical cases by task and mandatory semantics.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0054
**Action:** Perform cheap scale/type bucket retrieval.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0055
**Action:** Compute weighted meta-feature distance for shortlist.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0056
**Action:** Aggregate successful-strategy priors using support-aware/Bayesian smoothing.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0057
**Action:** Aggregate failure priors and resource-failure signatures.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- Use conservative resource estimate and enforce hard limits before expensive execution.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0058
**Action:** Estimate retrieval confidence and out-of-distribution score.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0059
**Action:** If confidence is low, blend strongly toward generic priors and preserve larger exploration budget.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Use conservative resource estimate and enforce hard limits before expensive execution.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0060
**Action:** Return priors that influence ordering only; emit RETRIEVAL_COMPLETED.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Persist correlation IDs and sanitize raw sensitive values.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.


# PHASE — BASELINES

## Step 0061
**Action:** Build trivial task baseline: majority/prior or mean/median.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0062
**Action:** Build cheap linear/probabilistic baseline when schema allows.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0063
**Action:** Build cheap nonlinear tree baseline.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0064
**Action:** Evaluate baselines using the development validation plan only.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0065
**Action:** Record fold metrics, train metrics, wall time, RAM and prediction artifacts.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- Confirm entity/order/cutoff semantics; future observations must not affect past features.
- Use conservative resource estimate and enforce hard limits before expensive execution.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0066
**Action:** Create residual/OOF reference needed by incremental feature scoring.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0067
**Action:** Abort or warn on degenerate target/no-signal conditions according to policy.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.


# PHASE — FEATURE HYPOTHESIS GENERATION

## Step 0068
**Action:** Enumerate allowed unary numeric transforms for numeric columns.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0069
**Action:** Enumerate categorical transforms for categorical/high-cardinality columns.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0070
**Action:** Enumerate date decomposition only for validated datetime columns.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm entity/order/cutoff semantics; future observations must not affect past features.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0071
**Action:** Enumerate pairwise arithmetic/interactions using routed pair heuristics rather than all O(d^2) pairs blindly.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0072
**Action:** Enumerate ratios only with denominator-safety policy.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0073
**Action:** Enumerate group counts/means/sums/min/max/std/recency only when group semantics are valid.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0074
**Action:** Enumerate row lags only within entity and stable order.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm entity/order/cutoff semantics; future observations must not affect past features.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0075
**Action:** Enumerate time-offset lags only using historical observations at or before allowed lookup time.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm entity/order/cutoff semantics; future observations must not affect past features.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0076
**Action:** Enumerate rolling windows with explicit closed/open interval semantics such as [T-W,T).

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm entity/order/cutoff semantics; future observations must not affect past features.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0077
**Action:** Enumerate expanding/EWM features with current-row inclusion policy explicit.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0078
**Action:** Enumerate target encoding/WOE only through cross-fitting.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0079
**Action:** Enumerate text length/count/TF-IDF/SVD families only when text routing enables them.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0080
**Action:** Enumerate PCA/SVD/clustering only when dimensionality regime and budget justify them.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Use conservative resource estimate and enforce hard limits before expensive execution.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0081
**Action:** Merge high-prior historical FeatureNodes.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0082
**Action:** Reserve a bounded exploration quota for generic/novel candidates.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0083
**Action:** Canonicalize expressions, fold commutative operand order, simplify safe identities and deduplicate.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0084
**Action:** Enforce max feature depth, max candidates per family and global pending-candidate limit.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0085
**Action:** Estimate cost, fit scope, target permission, temporal semantics and null policy for every FeatureNode.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- Confirm entity/order/cutoff semantics; future observations must not affect past features.
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.


# PHASE — FEATURE CHEAP GATES

## Step 0086
**Action:** Reject schema/type-incompatible FeatureNodes before materialization.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0087
**Action:** Reject invalid domains: unsafe log/root/division unless protected transform semantics are defined.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0088
**Action:** Reject direct identifier features by default; retain identifier as grouping key where justified.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0089
**Action:** Reject confirmed temporal lookahead and target leakage.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- Confirm entity/order/cutoff semantics; future observations must not affect past features.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0090
**Action:** Reject candidates whose conservative memory estimate violates hard policy.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0091
**Action:** Run constant/near-constant checks.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0092
**Action:** Run cheap duplicate/hash equivalence checks on bounded sample.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0093
**Action:** Compute low-cost candidate association/variance proxy only as ranking evidence.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0094
**Action:** Use incremental/FeatureBoost-like probe or cheap residual model where implemented.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- Record exact spec, seed/fidelity, fold metrics, wall time, RAM and status.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0095
**Action:** Calculate expected incremental utility per estimated cost.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0096
**Action:** Keep top candidates plus exploration reserve and prune the rest with recorded reasons.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.


# PHASE — PROGRESSIVE FEATURE EVALUATION

## Step 0097
**Action:** Materialize first-stage candidate batches lazily on low fidelity.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0098
**Action:** For fitted transforms, fit state inside training fold only.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0099
**Action:** Transform validation rows using training-fold state.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0100
**Action:** Evaluate incremental quality against a fixed comparable base portfolio/model.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Record exact spec, seed/fidelity, fold metrics, wall time, RAM and status.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0101
**Action:** Record fold-level deltas rather than only aggregate delta.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0102
**Action:** Measure feature generation/materialization time and output bytes.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm entity/order/cutoff semantics; future observations must not affect past features.
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0103
**Action:** Measure added training and inference cost.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0104
**Action:** Promote statistically/resource-plausible candidates.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- Use conservative resource estimate and enforce hard limits before expensive execution.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0105
**Action:** Increase row/fold fidelity for survivors only.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0106
**Action:** Re-evaluate suspected unstable candidates under stricter folds.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0107
**Action:** Compute model-agnostic/permutation or ablation evidence only on sufficiently valid model and shortlisted features.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- Record exact spec, seed/fidelity, fold metrics, wall time, RAM and status.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0108
**Action:** Cluster highly correlated/redundant survivors; do not interpret low permutation importance naively under collinearity.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0109
**Action:** Produce verified FeatureCandidateResult records.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.


# PHASE — FEATURE PORTFOLIO

## Step 0110
**Action:** Initialize portfolio with validated base/raw features permitted by model backend.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- Record exact spec, seed/fidelity, fold metrics, wall time, RAM and status.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0111
**Action:** Run greedy marginal-addition search as baseline portfolio algorithm.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0112
**Action:** Run bounded beam search when interactions/complementarity make greedy insufficient.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0113
**Action:** Optionally run late evolutionary mutation/crossover only after candidate pruning.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0114
**Action:** Evaluate whole portfolio because individually weak features can interact.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0115
**Action:** Penalize redundancy, latency, RAM and instability.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Use conservative resource estimate and enforce hard limits before expensive execution.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0116
**Action:** Stop portfolio expansion when marginal utility is below threshold or budget exhausted.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Record exact spec, seed/fidelity, fold metrics, wall time, RAM and status.
- Use conservative resource estimate and enforce hard limits before expensive execution.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0117
**Action:** Freeze a small shortlist of feature portfolios for model routing.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- Record exact spec, seed/fidelity, fold metrics, wall time, RAM and status.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.


# PHASE — MODEL ROUTING AND HPO

## Step 0118
**Action:** Route eligible model families from task/schema/constraints.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Record exact spec, seed/fidelity, fold metrics, wall time, RAM and status.
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0119
**Action:** Order model families using historical priors plus generic diversity requirement.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Record exact spec, seed/fidelity, fold metrics, wall time, RAM and status.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0120
**Action:** Construct conditional hyperparameter spaces with sensible scales.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Use conservative resource estimate and enforce hard limits before expensive execution.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0121
**Action:** Assign low initial fidelity to expensive candidates.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0122
**Action:** Estimate trial time/RAM before scheduling.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm entity/order/cutoff semantics; future observations must not affect past features.
- Record exact spec, seed/fidelity, fold metrics, wall time, RAM and status.
- Use conservative resource estimate and enforce hard limits before expensive execution.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0123
**Action:** Reject impossible model/backend/resource combinations before launch.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Record exact spec, seed/fidelity, fold metrics, wall time, RAM and status.
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- Use conservative resource estimate and enforce hard limits before expensive execution.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0124
**Action:** Schedule diverse cheap trials in parallel under resource tokens.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Record exact spec, seed/fidelity, fold metrics, wall time, RAM and status.
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- Use conservative resource estimate and enforce hard limits before expensive execution.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0125
**Action:** Report intermediate metrics for iterative learners.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0126
**Action:** Prune unpromising iterative trials using selected pruner with warmup.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Record exact spec, seed/fidelity, fold metrics, wall time, RAM and status.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0127
**Action:** Use random/TPE/cost-aware optimizer according to search-space regime.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0128
**Action:** Use Successive Halving/Hyperband when resource fidelity is meaningful.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- Use conservative resource estimate and enforce hard limits before expensive execution.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0129
**Action:** Promote survivors to more rows/folds/iterations.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0130
**Action:** Diagnose overfit: train-validation gap, fold variance, stricter-validation collapse.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0131
**Action:** Diagnose underfit: poor train and validation, residual signal, capacity ceiling.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0132
**Action:** Activate gradient-norm/loss diagnostics only for neural/gradient regimes that expose them.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0133
**Action:** Adapt regularization/capacity/search bounds through recorded planner decisions.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0134
**Action:** Stop HPO when remaining budget or improvement criterion says additional search has low expected value.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Record exact spec, seed/fidelity, fold metrics, wall time, RAM and status.
- Use conservative resource estimate and enforce hard limits before expensive execution.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0135
**Action:** Create finalist TrialResults.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Record exact spec, seed/fidelity, fold metrics, wall time, RAM and status.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.


# PHASE — CALIBRATION AND DECISION POLICY

## Step 0136
**Action:** If probability quality matters, evaluate calibration using development/OOF predictions.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Record exact spec, seed/fidelity, fold metrics, wall time, RAM and status.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0137
**Action:** Fit calibrator using only development-safe data.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Record exact spec, seed/fidelity, fold metrics, wall time, RAM and status.
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0138
**Action:** Compare calibrated and uncalibrated metrics.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Record exact spec, seed/fidelity, fold metrics, wall time, RAM and status.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0139
**Action:** If operational threshold matters, optimize threshold against explicit metric/business utility on development validation.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Record exact spec, seed/fidelity, fold metrics, wall time, RAM and status.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0140
**Action:** Freeze calibration and threshold policy before final holdout.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- Record exact spec, seed/fidelity, fold metrics, wall time, RAM and status.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.


# PHASE — ENSEMBLE

## Step 0141
**Action:** Collect aligned OOF predictions from eligible finalist models.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Record exact spec, seed/fidelity, fold metrics, wall time, RAM and status.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0142
**Action:** Measure standalone competence.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0143
**Action:** Measure prediction correlation, residual/error correlation and disagreement.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0144
**Action:** Start ensemble with best feasible single model.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Record exact spec, seed/fidelity, fold metrics, wall time, RAM and status.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0145
**Action:** Test each candidate addition by OOF marginal metric gain and added latency/size/RAM.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Use conservative resource estimate and enforce hard limits before expensive execution.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0146
**Action:** Add only positive feasible marginal utility candidates.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0147
**Action:** Optimize weights using development/OOF predictions only.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0148
**Action:** If stacking is proposed, train stacker on OOF base predictions only.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0149
**Action:** Run dedicated stacked-overfit guard and disable stacking when gain fails to generalize.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0150
**Action:** Freeze best feasible ensemble and retain best single-model alternative.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Record exact spec, seed/fidelity, fold metrics, wall time, RAM and status.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.


# PHASE — PARETO AND FINAL FREEZE

## Step 0151
**Action:** Apply hard constraints to finalists.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0152
**Action:** Construct Pareto frontier across active objectives.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0153
**Action:** Remove dominated candidates.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0154
**Action:** Apply user selection policy/utility to frontier.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0155
**Action:** Freeze feature graph, preprocessing policy, model/ensemble, calibration and threshold.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- Record exact spec, seed/fidelity, fold metrics, wall time, RAM and status.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0156
**Action:** Write FINALIST_FROZEN audit event.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Persist correlation IDs and sanitize raw sensitive values.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0157
**Action:** Verify no final-holdout data has been referenced upstream.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.


# PHASE — FINAL HOLDOUT AND REFIT

## Step 0158
**Action:** Open final holdout according to the predeclared policy.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0159
**Action:** Fit candidate using allowed development data.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0160
**Action:** Evaluate once on final holdout and record metrics/uncertainty/resource cost.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- Use conservative resource estimate and enforce hard limits before expensive execution.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0161
**Action:** Compare final result against CV expectation; diagnose collapse without tuning to holdout.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0162
**Action:** Mark final holdout consumed.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm train/validation/final-holdout isolation and target-access permission.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0163
**Action:** Decide deployment refit policy on all permissible labeled data.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0164
**Action:** Fit final preprocessing/FeatureGraph/model state.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- Record exact spec, seed/fidelity, fold metrics, wall time, RAM and status.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0165
**Action:** Measure inference latency at required batch size and hardware context.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Use conservative resource estimate and enforce hard limits before expensive execution.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0166
**Action:** Hash final model and fitted-state artifacts.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Record exact spec, seed/fidelity, fold metrics, wall time, RAM and status.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.


# PHASE — CODEGEN AND VERIFICATION

## Step 0167
**Action:** Compile frozen PipelineIR into standalone project structure.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0168
**Action:** Generate schema validation and data adapter code.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0169
**Action:** Generate preprocessing and exact FeatureNode operations.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0170
**Action:** Generate model/ensemble load, predict and optional train/evaluate code.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Record exact spec, seed/fidelity, fold metrics, wall time, RAM and status.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0171
**Action:** Generate tests and dependency manifest from used components only.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Persist correlation IDs and sanitize raw sensitive values.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0172
**Action:** Generate ARCHITECTURE.md, run report and artifact manifest.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Persist correlation IDs and sanitize raw sensitive values.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0173
**Action:** Run syntax/import checks.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0174
**Action:** Run generated unit tests.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0175
**Action:** Compare generated feature outputs to FIAE runtime on frozen verification sample.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm entity/order/cutoff semantics; future observations must not affect past features.
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0176
**Action:** Compare generated predictions to FIAE runtime within defined tolerance.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Confirm entity/order/cutoff semantics; future observations must not affect past features.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0177
**Action:** Test unseen categories, nulls, invalid schema and clean-process loading.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0178
**Action:** Re-measure latency/resource constraints.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- Use conservative resource estimate and enforce hard limits before expensive execution.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0179
**Action:** If verification fails, repair compiler/IR path and repeat; never mark export verified early.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.


# PHASE — EXPERIENCE WRITEBACK AND COMPLETION

## Step 0180
**Action:** Create immutable success/failure cases for relevant trials and feature decisions.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Preserve FeatureNode lineage, fit scope, null/domain policy and cost measurement.
- Record exact spec, seed/fidelity, fold metrics, wall time, RAM and status.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0181
**Action:** Persist context, action, result, cost and diagnosis without raw sensitive values by default.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Persist correlation IDs and sanitize raw sensitive values.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0182
**Action:** Update smoothed strategy/failure statistics.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0183
**Action:** Schedule offline meta-retraining only if distinct-dataset trigger is met.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Avoid unnecessary full scans and preserve source/schema fingerprint provenance.
- Use conservative resource estimate and enforce hard limits before expensive execution.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0184
**Action:** Flush critical events/audit records.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Persist correlation IDs and sanitize raw sensitive values.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0185
**Action:** Verify artifact hashes and completion manifest.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Persist correlation IDs and sanitize raw sensitive values.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0186
**Action:** Set run COMPLETED only when completion contract passes.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.

## Step 0187
**Action:** Expose CLI/dashboard summary, final Pareto choice, artifacts and known limitations.

**Inputs:** typed durable state from prior completed steps plus explicitly permitted data/artifacts.

**Mandatory checks:**
- Fail closed on violated hard invariants and persist enough evidence to reconstruct the decision.
- If this step changes a consequential selection/rejection, write a DecisionRecord.
- If execution is expensive, measure actual cost and update the cost model.

**Output:** typed result/state with stable IDs; no hidden decision stored only in logs.

**Failure behavior:** classify the failure, persist it, apply retry policy only if retryable, and do not silently continue with corrupted state.


---

# Main control loop

```text
while run not terminal:
    state = load_durable_state()

    if cancellation_requested:
        cancel_safely()
        break

    ready = planner.ready_actions(state)

    for action in ready:
        if violates_hard_invariant(action):
            record_rejection(action)
            continue

        if requires_offline_research(action):
            pause_only_that_branch()
            create_research_question()
            continue

        reserve_resources(action)
        result = execute(action)
        verify_result_contract(result)
        release_resources(action)

        if result.failed:
            classify_and_store_failure(result)
            retry_only_if_policy_allows(result)
        else:
            persist(result)
            planner.consume(result)

    if completion_contract_satisfied(state):
        finalize_run()
```

# Research escalation loop

```text
1. Write a precise technical question from an observed limitation.
2. Record the current implementation and benchmark.
3. Search primary/official documentation and research papers first.
4. Separate source-derived facts from FIAE hypotheses.
5. List at least one plausible alternative.
6. Implement the smallest challenger behind an interface.
7. Run correctness/property tests.
8. Run equal-budget benchmark.
9. Run ablation when the component interacts with others.
10. Accept only if evidence passes the gate.
11. Store the decision and rollback trigger.
```

Research does not run inside every hot-path feature trial; online execution uses cached knowledge and empirical experiments.

# Completion contract

A run can be `COMPLETED` only when problem definition, validation, selected pipeline evidence, final-evaluation status, artifact manifest and audit summary are internally consistent. Budget-exhausted best-so-far results must be labeled as such if mandatory verification gates were not completed.
