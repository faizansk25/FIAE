# 07 — Progressive Search, HPO, Over/Underfitting Diagnostics, and Ensembles

## Core rule
```text
retrieve → route → cheap probe → prune → promote → verify → ensemble only for marginal value
```

## Unified TrialSpec
```text
TrialSpec
├── trial_id
├── feature_portfolio_id
├── model family/backend/hyperparameters
├── fidelity
│   ├── row_fraction
│   ├── fold_count
│   ├── iterations/trees/epochs
│   └── feature_fraction
├── constraints
├── seed
├── timeout
└── resource reservation
```

## Fidelity
Fidelity can mean rows, folds, trees, epochs, feature count or candidate depth. Do not raise every dimension at once.

## Default search ladder
Large-data starting policy:
```text
5% → 16% → 40% → 100%
```
Medium-data starting policy:
```text
16% → 32% → 64% → 100%
```
Small data:
```text
100% rows → progressively stronger CV, fewer candidates
```
These are policies to benchmark, not universal constants.

## Promotion utility
Hard constraints first. Then:
```text
utility =
    normalized_quality
  - λ_variance * fold_variance
  - λ_gap      * generalization_gap
  - λ_time     * time
  - λ_ram      * memory
  - λ_latency  * inference_latency
  - λ_size     * model_size
  - λ_risk     * risk
```

## Successive halving
```text
current = candidates
for fidelity in fidelity_schedule:
    results = evaluate_parallel(current, fidelity)
    feasible = hard_constraint_filter(results)
    rank(feasible)
    current = top(feasible, ceil(len(feasible)/eta))
return current
```
Do not eliminate candidates based on tiny noisy differences at extremely low fidelity; use uncertainty/minimum-support guards.

## Cost-aware ordering
```text
priority =
    expected_improvement / max(predicted_seconds, epsilon)
    - memory_risk_penalty
    - failure_risk_penalty
```
Preserve an exploration quota so priors cannot suppress unknown strategies entirely.

## Model routing table
| Family | Use | Main role |
|---|---|---|
| linear/logistic | classification/regression | cheap baseline, sparse/high-dimensional |
| SGD linear | very large/sparse | cheap/streamable |
| naive Bayes | selected text/count classification | probabilistic baseline |
| decision tree | tabular | nonlinear diagnostic |
| random forest | tabular | robust/diverse baseline |
| extra trees | tabular | diversity source |
| HistGradientBoosting | tabular | strong low-dependency baseline |
| LightGBM | optional | strong fast GBM |
| XGBoost | optional | strong GBM challenger |
| CatBoost | optional | categorical-aware challenger |
| MLP | conditional | only when regime/budget justifies |

## HPO algorithm routing
### Random search
Use as reference and cold-start method for wide mixed spaces.

### TPE
Use for conditional mixed spaces after enough observations.

### Successive Halving
Use when a meaningful monotonically increasing resource exists.

### Hyperband
Use when best starting resource allocation is uncertain; multiple halving brackets explore the trade-off.

### Bayesian/GP
Use only for relatively small expensive mostly-continuous spaces; avoid as default for huge conditional trees.

### CMA-ES
Use for continuous correlated parameter subspaces when appropriate.

### Evolutionary search
Use late for joint feature-graph/model refinement after coarse pruning.

### FLAML-style cost-aware search
Use trial cost explicitly; cheaper informative trials can be preferable early.

## HPO spaces
### Linear
- regularization strength: log scale
- penalty/mixing only if supported
- class weights when routed
- max iterations/convergence tolerance

### Forest
- trees as fidelity
- depth
- min samples leaf
- max features
- bootstrap
- class weights

### Gradient boosting
- learning rate log-scale
- iterations as fidelity
- depth/leaves
- min child samples
- row/column subsampling
- L1/L2 regularization
- early stopping

## Early stopping
```text
train to checkpoint
→ report validation metric
→ report cost
→ pruner decision
→ continue or stop
```
Warmup prevents premature pruning of slow-starting trials.

## Overfitting detector
Signals:
- train much better than validation,
- large fold variance,
- degradation under stricter validation,
- importance instability,
- calibration failure,
- stacked ensemble collapse.
Response order:
1. inspect leakage/split,
2. reduce unstable features,
3. increase regularization,
4. reduce capacity,
5. increase data/fold rigor if possible,
6. stop over-tuning the same validation signal.

## Underfitting detector
Signals:
- train and validation both low,
- similar poor score across simple/complex models,
- learning curve plateaus low.
Actions:
- verify target/problem,
- increase useful feature coverage,
- reduce excessive regularization,
- increase capacity,
- test nonlinear interactions,
- verify predictive information exists.

## Gradient diagnostics — conditional
Activate only when gradients are observable/relevant.
Track:
- training/validation loss,
- gradient norm/max norm,
- update/weight norm,
- NaN/Inf,
- LR history.
Exploding: scale inputs, lower LR, clip gradient, inspect initialization/normalization.
Vanishing/no-learning: inspect activation/depth, LR, normalization/residual structure. Compare against simpler models before spending more.

## Imbalance
Metric first. Then candidate strategies:
- class weights,
- threshold optimization,
- calibrated probabilities,
- training-fold-only resampling.
Synthetic sampling never sees validation/test rows.

## Calibration
If probability quality matters:
- Brier/log-loss/calibration plots,
- fit calibrator only using development/OOF information,
- compare calibrated vs uncalibrated.

## Threshold
Probability estimation and decision policy are separate.
Possible objective:
- F1,
- precision at recall floor,
- recall at precision floor,
- expected business utility.
Threshold is frozen before final holdout.

## Ensemble eligibility
Require:
- valid validation,
- OOF predictions,
- reproducible inference,
- hard constraints satisfied,
- no confirmed leakage,
- either competence or complementary errors.

## Diversity
Measure:
- prediction correlation,
- error/residual correlation,
- classifier disagreement,
- marginal metric delta.

## Greedy ensemble
```text
ensemble = best_single
repeat:
    for candidate not in ensemble:
        evaluate addition using OOF predictions
        compute metric gain + latency/size/RAM delta
    add best feasible positive-marginal candidate
until no meaningful gain
```

## Stacking
1. generate base-model OOF predictions,
2. train stacker only on OOF matrix,
3. align folds exactly,
4. run stacked-overfit guard,
5. disable stacking if dedicated guard shows non-generalizing gain.

## Pareto
Candidate A dominates B if A is no worse on every active objective and better on at least one.
Objectives can include quality, stability, time, RAM, latency, size, complexity, interpretability.

## Final holdout
Before opening:
- features frozen,
- model/hyperparameters frozen,
- ensemble weights frozen,
- calibration frozen,
- threshold frozen.
Evaluate according to policy and do not retune to this holdout.

## Stopping
Stop when:
- budget exhausted,
- quality target reached with confidence,
- no meaningful improvement for policy-defined window,
- remaining candidates dominated,
- user cancellation.

## Mandatory audit
Each stage stores:
- entering candidates,
- pruned candidates + reasons,
- promoted candidates,
- metrics and uncertainty,
- time/RAM spent,
- remaining budget.
