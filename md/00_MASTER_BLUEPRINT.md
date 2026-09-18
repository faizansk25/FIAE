# FIAE Master Blueprint

> **Purpose:** Define the full system goal, invariants, architecture, inputs, outputs, research loop, and success measures.

**Status:** Normative specification for one continuously evolving FIAE codebase. These documents describe implementation gates, not user-facing versions.


## 1. Product definition

FIAE — Feature Intelligence & Architecture Engine — accepts a data source plus a modeling objective and produces a verified feature portfolio, preprocessing graph, model or ensemble, final evaluation, production architecture, generated code, tests, logs, and reusable experience records.

FIAE is not “AutoML plus a few extra transforms.” The intended differentiator is **feature intelligence under constraints**: use schema, statistical structure, prediction-time semantics, historical success/failure cases, progressive experimentation, and cost models to propose and verify only plausible feature/model paths.

The online runtime must be fast. Live web research is not part of ordinary training runs. Research is an engineering loop and a low-confidence escalation path.

## 2. Primary objectives

The system jointly optimizes:

1. generalization quality;
2. stability across folds/seeds;
3. calibration or business utility where relevant;
4. feature and model complexity;
5. wall-clock training time;
6. CPU/GPU usage;
7. peak memory;
8. disk/model size;
9. inference latency and throughput;
10. interpretability;
11. leakage and prediction-time risk;
12. reproducibility;
13. operational reliability.

## 3. Hard principles

1. **Executed evidence beats reasoning.** A language model, rule, or meta-learner can propose a feature; only a valid experiment can accept it.
2. **Confirmed leakage is a hard veto.**
3. **Learned transforms fit only on training partitions.**
4. **Final holdout does not participate in feature/model/HPO/ensemble/threshold selection.**
5. **Feature candidates are represented lazily as IR/DAG nodes before expensive materialization.**
6. **Prior experience controls search order, not truth.**
7. **Failed experiments are retained as negative evidence.**
8. **Progressive fidelity replaces brute force.**
9. **All expensive work has resource preflight and cancellation.**
10. **Parallelism is bounded by CPU/RAM tokens.**
11. **Minimal mandatory dependencies; heavy engines remain optional adapters.**
12. **CLI, dashboard, and Python API call the same core engine.**
13. **All material decisions emit structured events and audit records.**
14. **Generated code must be executed and compared to the internal pipeline.**
15. **“100x better” requires a benchmark definition, not a slogan.**

## 4. High-level architecture

```text
CLI / Dashboard / Python SDK
            |
            v
       Control Plane
            |
            v
       Run Orchestrator
            |
 +----------+-----------+------------------+------------------+
 |                      |                  |                  |
 v                      v                  v                  v
Data Intake        Problem/Split       Constraint        Experience
Profiler           Leakage Intel       Engine            Retrieval
 |                      |                  |                  |
 +----------------------+---------+--------+------------------+
                                  |
                                  v
                             Aspect Router
                                  |
                                  v
                        Feature Hypothesis Engine
                                  |
                                  v
                            Feature IR / DAG
                                  |
                                  v
                      Static + Cheap Feature Gates
                                  |
                                  v
                       Progressive Feature Search
                                  |
                                  v
                       Feature Portfolio Search
                                  |
                                  v
                         Model / HPO Search
                                  |
                                  v
                         Ensemble / Calibration
                                  |
                                  v
                        Pareto + Hard Constraints
                                  |
                                  v
                         Frozen Final Candidate
                                  |
                                  v
                          Final Holdout Check
                                  |
                                  v
                      Architecture + Code Generation
                                  |
                                  v
                         Execute / Verify / Audit
                                  |
                                  v
                     Experience + Artifact Writeback
```

## 5. Two-brain architecture

### Offline Research & Experience Brain

Stores:
- official research findings and engineering decisions;
- benchmark results;
- dataset meta-features;
- successful feature portfolios;
- failed feature/model trials;
- resource measurements;
- leakage and shift patterns;
- transfer-performance evidence.

### Online Fast Brain

Per new dataset:
- profile;
- define task;
- design split;
- detect leakage;
- retrieve similar cases;
- route relevant aspects;
- create feature hypotheses;
- prune cheaply;
- train progressively;
- select under constraints;
- generate and verify code.

## 6. Input contract

Minimum:
- data source;
- target unless unsupervised mode is explicit.

Strongly recommended:
- prediction timestamp;
- event time;
- entity/group key;
- horizon;
- positive class;
- business metric;
- hard resource/deployment constraints.

Example:

```yaml
target: churn
task: auto
time_column: event_time
prediction_time_column: scoring_time
entity_key: customer_id
positive_class: 1
objective:
  primary_metric: average_precision
constraints:
  wall_clock_seconds: 1800
  peak_ram_mb: 8192
  p95_latency_ms: 20
  model_size_mb: 500
  cpu_workers: 8
  interpretability: medium
search:
  enable_target_encoding: true
  enable_ensembles: true
  max_feature_depth: 2
```

## 7. Output contract

A completed run should be able to emit:

```text
run_manifest.json
data_profile.json
schema.json
problem_definition.json
validation_plan.json
leakage_report.json
aspect_plan.json
experience_prior.json
feature_graph.json
feature_lineage.json
feature_leaderboard.jsonl/parquet
model_leaderboard.jsonl/parquet
ensemble_report.json
pareto_frontier.json
final_pipeline.json
architecture.md
generated_project/
tests/
benchmark_report.json
events.jsonl
audit.jsonl
metrics.jsonl
```

## 8. Decision order

Hard constraints and validity precede quality:

```text
invalid data
-> invalid target/task
-> invalid split
-> confirmed leakage
-> prediction-time unavailable feature
-> hard resource/deployment violation
-> reproducibility/test failure
-> quality/stability comparison
-> resource efficiency
-> interpretability
-> simplicity tie-break
```

## 9. Research-development loop

Every subsystem follows:

```text
search official docs/papers
-> define hypothesis
-> design minimal implementation
-> implement
-> unit/property/edge tests
-> benchmark
-> research alternatives again
-> ablation
-> accept/reject
-> record engineering decision
-> proceed
```

A module does not pass because it “works.” It passes only when correctness, performance, and regression gates pass.

## 10. Valid meaning of “100x”

Possible measurable claims:
- 100x fewer full-data feature evaluations at statistically equivalent quality;
- 50x fewer materialized bytes;
- 20x faster time-to-strong-baseline;
- 10x fewer expensive failed trials after meta-learning.

Invalid claim:
- “accuracy is 100x better.”

## 11. Cold-start behavior

Historical cases are optional for correctness. If experience retrieval is weak:
1. use conservative feature grammar;
2. run simple baselines;
3. use cost-aware successive halving;
4. store the new case.

## 12. Unknown-regime behavior

If task/split/prediction-time semantics are genuinely ambiguous:
- do not silently guess;
- stop destructive or target-aware steps;
- request clarification or use a conservative, clearly marked assumption;
- record uncertainty.

## Research basis

- [AutoGluon feature engineering](https://auto.gluon.ai/stable/tutorials/tabular/tabular-feature-engineering.html)
- [H2O Driverless AI feature engineering](https://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/feature-engineering.html)
- [DataRobot modeling process](https://docs.datarobot.com/latest/en/docs/reference/pred-ai-ref/model-ref.html)
- [SageMaker Autopilot modes](https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-model-support-validation.html)
- [OpenFE paper](https://proceedings.mlr.press/v202/zhang23ay.html)
- [FLAML research](https://www.microsoft.com/en-us/research/articles/flaml-a-fast-and-lightweight-automl-library/)
- [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
