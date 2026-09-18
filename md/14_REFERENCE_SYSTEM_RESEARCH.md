# 14 — Reference-System Research Map and FIAE Adaptation Decisions

## Rule
This document separates external/reference behavior from FIAE proposals. It is not a claim that FIAE already outperforms any system.

## AutoGluon
Observed reference ideas:
- type-aware tabular feature handling,
- strong bagging/stacking,
- explicit inference-limit constraints.
FIAE adaptation:
- optional challenger/backend,
- richer generic feature discovery before model stage,
- explicit multi-objective resource accounting,
- no required Ray/heavy framework in core.

Source:
https://auto.gluon.ai/stable/tutorials/tabular/tabular-feature-engineering.html
https://auto.gluon.ai/stable/tutorials/tabular/tabular-indepth.html

## H2O Driverless AI
Observed:
- automatic transformations/interactions,
- evolutionary feature creation/selection,
- joint search of feature transforms and model params,
- rich transformer catalog including target, time, aggregation and interactions,
- leakage/shift diagnostics.
FIAE adaptation:
- history-guided warm start,
- lazy Feature DAG,
- aggressive progressive pruning before evolution,
- explicit latency/RAM/model-size objectives,
- failure memory.

Source:
https://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/feature-engineering.html
https://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/transformations.html
https://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/modeling.html

## DataRobot
Observed:
- staged Autopilot resource use; for sufficiently large data full Autopilot uses 16%, 32%, 64% stages,
- broad blueprints/transformations/data-quality checks.
FIAE adaptation:
- generalize fidelity beyond fixed sample fractions,
- candidate/failure evidence stored permanently,
- feature DAG and explicit cost.

Source:
https://docs.datarobot.com/en/docs/reference/pred-ai-ref/model-ref.html

## SageMaker Autopilot
Observed:
- ensembling uses AutoGluon,
- HPO uses Bayesian or multi-fidelity depending on mode/data size,
- OOF-based cross-validation for ensembling,
- candidate-definition notebooks expose preprocessing, algorithms and HPO ranges.
FIAE adaptation:
- candidate transparency in machine-readable DecisionRecords/IR,
- multi-fidelity independent of AWS/cloud,
- routing based on richer meta/constraints, not one file-size threshold.

Source:
https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-model-support-validation.html
https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-metrics-validation.html
https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-candidate-generation-notebook.html

## OpenFE
Observed:
- automated feature generation bottleneck is evaluating/selecting useful candidates from an enormous pool,
- FeatureBoost incremental evaluation,
- two-stage coarse-to-fine pruning.
FIAE adaptation:
- use incremental/cheap evidence as a pruning principle,
- expand utility to cost, leakage, stability, latency, memory,
- portfolio and history-aware search.

Source:
https://proceedings.mlr.press/v202/zhang23ay.html

## FLAML
Observed:
- search order optimized around both cost and model quality,
- cheap trials tend to precede expensive trials.
FIAE adaptation:
- expected improvement per cost plus explicit memory/failure risk.

Source:
https://www.microsoft.com/en-us/research/articles/flaml-a-fast-and-lightweight-automl-library/

## Optuna
Observed:
- pruning algorithms include Median, Successive Halving, Hyperband and others,
- multi-thread/process/node optimization patterns.
FIAE adaptation:
- optional HPO backend; scheduler interface remains FIAE-owned.

Source:
https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html
https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html

## scikit-learn
Observed:
- split before fitted preprocessing; train-only fitting prevents leakage,
- nested CV estimates generalization of tuning/selection procedure,
- successive-halving search increases resources for survivors,
- permutation importance is useful but correlated features can mask importance.
FIAE adaptation:
- hard fold-safety contract,
- strong validation only on shortlisted candidates,
- importance never used alone.

Source:
https://scikit-learn.org/stable/common_pitfalls.html
https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html
https://scikit-learn.org/stable/modules/permutation_importance.html

## Polars
Observed:
- lazy query optimizer supports predicate/projection pushdown, common subplan elimination and other optimizations,
- lazy execution supports streaming/larger-than-memory patterns.
FIAE adaptation:
- logical Feature DAG before materialization,
- optional accelerated execution adapter.

Source:
https://docs.pola.rs/user-guide/lazy/optimizations/
https://docs.pola.rs/user-guide/lazy/using/

## DuckDB
Observed:
- CSV sniffer detects dialect/types/header,
- default type-detection sample is 20,480 rows,
- seekable regular files can be sampled at different positions.
FIAE adaptation:
- avoid head-only inference,
- optional robust CSV accelerator/sniffer.

Source:
https://duckdb.org/docs/current/data/csv/auto_detection

## MLflow
Observed:
- tracking organizes runs with params, metrics, artifacts and metadata.
FIAE adaptation:
- internal schema maps to it; optional exporter only.

Source:
https://mlflow.org/docs/latest/ml/tracking

## OpenTelemetry
Observed:
- Python supports traces/metrics/logs with documented maturity.
FIAE adaptation:
- internal telemetry abstraction + optional OTel exporter.

Source:
https://opentelemetry.io/docs/languages/python/

## Cross-system synthesis
Mature systems support these principles:
1. type-aware preprocessing,
2. strict validation,
3. model portfolios,
4. progressive resource allocation,
5. feature transformations,
6. ensembles,
7. inspection/reporting,
8. compute efficiency.

## FIAE differentiation target
Must be proven, not assumed:
- reusable success + failure case memory,
- semantic lazy Feature DAG,
- history-guided feature/model/resource search,
- deterministic leakage veto,
- minimal local core,
- verified production code export,
- full audit/observability.

## What must be benchmarked
- retrieval reduces work without negative transfer,
- pruning reduces feature evaluations without unacceptable quality loss,
- scheduler improves time/RAM,
- failure memory changes future search beneficially,
- codegen reproduces runtime,
- dashboard/logging overhead stays bounded.
