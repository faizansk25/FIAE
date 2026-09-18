# Problem, Validation, and Leakage Intelligence

> **Purpose:** Define the prediction problem and prove the evaluation design is legitimate before learned feature search.

**Status:** Normative specification for one continuously evolving FIAE codebase. These documents describe implementation gates, not user-facing versions.

## Research basis

- [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
- [DataRobot data quality](https://docs.datarobot.com/en/docs/reference/data-ref/data-quality-ref.html)
- [H2O Driverless AI feature configuration](https://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/config_docs/features_config.html)
- [SageMaker Autopilot validation](https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-metrics-validation.html)
- [scikit-learn threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html)


## Task inference

1. Validate explicit task if supplied.
2. String/bool/categorical target -> classification candidate.
3. Numeric target with low semantic cardinality -> classification candidate.
4. Numeric target with broad continuous support -> regression candidate.
5. Time-indexed target plus future horizon -> forecasting candidate.
6. Ranking requires explicit query/group semantics.
7. No-target anomaly/unsupervised mode must be explicit.
8. Emit confidence and reasons.
9. If ambiguity is material, request clarification.

## Metric routing

### Binary classification
- ROC-AUC for ranking quality;
- average precision/PR-AUC for rare positives;
- log loss/Brier for probability quality;
- business cost/utility when supplied;
- F1/balanced accuracy only when the action objective warrants.

### Multiclass
- log loss;
- macro/micro F1;
- balanced accuracy;
- class-specific recall/precision where policy demands.

### Regression
- RMSE;
- MAE;
- R² as secondary;
- RMSLE only for compatible non-negative targets;
- MAPE guarded against zero/near-zero targets.

## Split planner

### Independent classification
Stratified K-fold.

### Independent regression
K-fold, usually shuffled with a deterministic seed if i.i.d. assumption is credible.

### Group dependence
GroupKFold/group holdout so one entity does not leak across train/validation.

### Time dependence
Ordered backtests/time splits; random row shuffling is invalid for future prediction.

### Tiny data
Use stronger CV and a smaller candidate search rather than repeatedly sacrificing rows to holdouts.

## Final holdout

Reserve before feature/model search.

Forbidden uses:
- feature selection;
- learned preprocessing;
- target encoding fit;
- HPO;
- model-family choice;
- ensemble weights;
- calibration;
- threshold tuning;
- early-stopping-policy choice.

## Progressive/nested validation

1. cheap probe split;
2. 3-fold screening;
3. 5-fold or repeated CV for survivors;
4. nested/dedicated selection verification for finalists if budget warrants;
5. untouched final holdout once.

Nested CV is reserved for finalists because it is computationally expensive, while still protecting against optimism from using the same CV procedure for tuning and evaluation.

## Leakage taxonomy

### L0 deterministic row-wise
No learned population/target state.

### L1 learned unsupervised
Scaler, imputer, PCA, frequency map. Fit on training fold only.

### L2 target-aware
Target encoding, WOE, supervised binning/selection. Strict cross-fitting.

### L3 temporal/history
Lags, rolling, history aggregates. Must obey time/entity/label availability.

### L4 prediction-time semantic risk
Potential post-event or operationally unavailable feature.

### L5 confirmed leakage
Exact target copy, future target, post-outcome field proven unavailable, validation/test target used in feature construction. Hard reject.

## Leakage detector pipeline

### Stage A deterministic checks
- exact target equality;
- inverted binary target;
- deterministic bijection;
- target embedded in strings;
- identifiers/row numbers;
- impossible timestamps.

### Stage B statistical suspicion
- near-perfect single-feature AUC/R²;
- deterministic category->target mapping;
- extreme mutual information;
- missingness nearly determines target;
- strong train/test domain-classifier performance.

Statistical suspicion is not proof. It creates `REVIEW_REQUIRED` unless semantic/temporal evidence confirms leakage.

### Stage C prediction-time semantics
For each suspicious feature determine:
- when created;
- when updated;
- whether known at scoring;
- whether computing it requires the outcome.

### Stage D graph fold-safety
Inspect learned transforms and prove each `fit` sees training partitions only.

## Cross-fit target encoding algorithm

For K folds:

```text
for fold k:
    train_idx = all folds except k
    val_idx = fold k
    learn smoothed category->target statistics on train_idx
    transform val_idx
concatenate all OOF encoded rows
learn final mapping on all development data for deployment
```

Unknown category -> global prior or configured fallback.

A training row never contributes its target to its own encoded value.

## Group aggregate safety

If aggregate learns population statistics, fit it inside the training fold.

For temporal online aggregates:
- the window/history can include only records whose information was available before the current prediction time.

## Prediction-time availability

For prediction row `i` at time `P_i`:

```text
max(availability_time(source_fact_j)) <= P_i
```

for every source fact used by the feature.

A target lag is safe only if the old label was observable before `P_i`; event time alone is not enough if labels are delayed.

## Calibration

1. produce unbiased development probabilities;
2. measure reliability/Brier/log loss;
3. fit candidate calibrator on independent/cross-validated predictions;
4. keep only if useful and within constraints.

## Decision threshold

Probability estimation and decision action are separate:

1. obtain unbiased development scores;
2. optimize threshold for metric/business utility;
3. freeze it;
4. open final holdout only after freeze.

## Distribution shift

A domain classifier can estimate whether a feature/distribution separates train and validation/test. Treat high shift as a risk/selection signal. Time-series tasks need special handling because chronological shift may be natural and informative.
