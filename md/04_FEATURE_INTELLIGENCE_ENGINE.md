# Feature Intelligence Engine

> **Purpose:** Specify feature IR, hypothesis generation, pruning, incremental evaluation, portfolio search, and evidence recording.

**Status:** Normative specification for one continuously evolving FIAE codebase. These documents describe implementation gates, not user-facing versions.

## Research basis

- [OpenFE paper](https://proceedings.mlr.press/v202/zhang23ay.html)
- [H2O Driverless AI feature engineering](https://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/feature-engineering.html)
- [DataRobot modeling process](https://docs.datarobot.com/latest/en/docs/reference/pred-ai-ref/model-ref.html)
- [AutoGluon feature engineering](https://auto.gluon.ai/stable/tutorials/tabular/tabular-feature-engineering.html)
- [scikit-learn permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)


## Hypothesis principle

No feature is accepted because a rule, LLM, or prior case says it is useful. Proposal mechanisms generate hypotheses; valid experiments decide.

## FeatureNode contract

```json
{
  "feature_id": "f_<content_hash>",
  "op": "safe_ratio",
  "inputs": ["total_spend", "active_months"],
  "params": {"zero_policy": "null"},
  "output_type": "numeric",
  "depth": 1,
  "fit_required": false,
  "leakage_class": "L0",
  "availability_rule": "all_inputs_available",
  "estimated_cpu_seconds": 0.1,
  "estimated_peak_ram_mb": 16,
  "estimated_output_mb": 8,
  "lineage": ["total_spend", "active_months"]
}
```

## Canonicalization

- `a+b` and `b+a` share signature;
- `min(a,b)` and `min(b,a)` share signature;
- `x+0 -> x`;
- `x*1 -> x`;
- `abs(abs(x)) -> abs(x)`;
- exact duplicate DAG nodes collapse;
- cycles reject;
- invalid numeric domain rejects before evaluation.

## Proposal sources

1. operation registry matching semantic types;
2. profile/statistical triggers;
3. semantic hints from names and metadata;
4. historical success/failure cases;
5. development OOF residual/error analysis;
6. optional domain plugin.

## Example triggers

- positive heavy skew -> log1p;
- mixed-sign heavy tail -> signed log;
- missingness -> missing indicator;
- excess zeros -> zero indicator;
- low-cardinality category -> one-hot candidate;
- high cardinality -> frequency/hash/cross-fit target encoding;
- date -> date parts/cyclical/elapsed;
- repeated entity -> group aggregate;
- temporal order -> lag/rolling;
- free text -> text statistics/hash/TFIDF under budget.

## Feature depth

```text
depth 0: raw
depth 1: unary/direct interaction
depth 2: transform-of-transform or interaction
depth >=3: only with strong evidence/prior and enough budget
```

Complexity penalty rises with depth, dependencies, state, and inference cost.

## Feature funnel

### F0 Preconditions
Types, numeric domain, semantic compatibility, prediction-time availability.

### F1 Static resource gate
Estimate CPU, RAM, output cardinality, sparse/dense bytes, graph depth.

### F2 Cheap materialization
Small valid development sample; inspect invalid rate, NaN/inf, variance, duplicate hashes.

### F3 Incremental probe
Cheap model or residual correction on a small valid split.

### F4 Progressive evaluation
Increasing rows/folds for survivors.

### F5 Portfolio interaction
Complementarity/redundancy.

### F6 Final stability
Ablation, fold/seed stability, shift, resource measurements.

## Constraint-aware incremental utility

Hard constraints filter first. A ranking utility can then combine:

```text
+ quality gain
+ stability gain
+ complementary diversity
- training time
- inference latency
- peak RAM
- feature/model bytes
- graph complexity
- risk penalty
```

Weights are policy, not universal constants.

## OpenFE-inspired coarse-to-fine search

OpenFE's research emphasizes that the key problem is efficiently finding useful features among a huge candidate pool, and uses an incremental evaluation idea with two-stage pruning.

FIAE follows the principle:

```text
huge symbolic candidate set
-> static/domain/cost rejection
-> cheap individual probe
-> successive pruning
-> interaction-aware portfolio evaluation
-> finalist ablation/stability
```

## Redundancy handling

Use:
- exact output signature/hash;
- correlation/association clustering;
- lineage overlap;
- conditional ablation;
- grouped permutation;
- model error diversity.

Do not use permutation importance alone; correlated features can mask one another.

## Portfolio search

### Greedy forward/backward
Add the candidate with best positive marginal utility. Periodically ablate accepted engineered features and remove those whose contribution disappeared.

### Beam
Keep a small number of alternative portfolios when pairwise interactions are likely.

### Evolutionary refinement
Only after pruning:
- individual = feature portfolio + compatible model summary;
- mutation = add/drop/replace transform;
- crossover = merge non-redundant blocks;
- fitness = valid constraint-aware score;
- preserve diversity;
- stop on plateau.

H2O Driverless AI demonstrates evolutionary feature/model search; FIAE adds historical warm starts, lazy DAG representation, explicit resource accounting, and aggressive early pruning.

## Feature acceptance record

Every accepted/rejected feature stores:
- proposal sources;
- lineage;
- operation/parameters;
- precondition evidence;
- leakage class;
- availability status;
- scores by stage;
- incremental gain;
- resource cost;
- fold stability;
- redundancy/interaction notes;
- final reason code.
