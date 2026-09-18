# FIAE Information Graph — Working Memory

> **Purpose:** Compressed navigation tree of the 16 normative specification documents.
> This is a memory aid, not a replacement for the specs. When in doubt, the numbered
> documents are authoritative.
>
> **Status:** Working document for the FIAE implementation effort.

## Layer 0 — Absolute invariants (docs 00 §3, 01) — apply to EVERY subsystem

1. Executed evidence beats reasoning; LLMs/rules/meta-learners only propose.
2. Confirmed leakage is a hard veto; exact target copy forbidden.
3. Learned transforms fit only on training partitions; never on final holdout.
4. Final holdout used exactly once, after all freezing; never for selection of anything.
5. Feature candidates are lazy IR/DAG nodes before materialization; graph is acyclic.
6. Prior experience controls search ORDER only, never truth; failures kept as negative evidence.
7. Progressive fidelity replaces brute force.
8. All expensive work has resource preflight + cancellation.
9. Parallelism bounded by CPU/RAM tokens.
10. Minimal mandatory dependencies; heavy engines are optional adapters (Tiers 0–4).
11. CLI/dashboard/SDK call ONE core engine; never implement independent ML logic.
12. All material decisions emit structured events + audit records; audit events never dropped.
13. Generated code must be executed and match internal predictions.
14. "100x" claims require benchmark definitions, never slogans.
15. More: fold-context cache keys; unbiased ensemble members; no random-row CV for future
    forecasting; entity-group leakage prevention; explicit domain errors; failed candidates
    get no valid scores; reproducibility records (seeds/env/versions/hashes); benchmark
    claims include baseline/hardware/data/seed policy/budget/metric.

Decision order: invalid data → invalid target/task → invalid split → confirmed leakage →
prediction-time unavailable → hard resource violation → reproducibility/test failure →
quality/stability → resource efficiency → interpretability → simplicity tie-break.

## Layer 1 — Typed contracts (doc 13) — the vocabulary of all modules

- **IDs:** run/stage/trial/feature/portfolio/decision/event/artifact; dataset + split
  fingerprints. Stable, opaque, deterministic under declared inputs (NFR-004).
- **Data:** DataSourceSpec, ColumnProfile, DatasetProfile.
- **Problem:** ProblemDefinition, ConstraintSpec, ValidationPlan, PredictionContext.
- **Leakage:** LeakageFinding (informational / review-required / hard-reject).
- **Features:** FeatureNode, FittedStateRef (split-scoped), FeaturePortfolio.
- **Trials:** ModelSpec, FidelitySpec, TrialSpec, TrialResult, EnsembleSpec.
- **Evidence:** MetricValue, ResourceMeasurement (latency requires batch size + hardware
  context), DecisionRecord, EventEnvelope, ArtifactRef, RunManifest, FIAEError.
- **Canonical serialization:** stable key order, stable enums, exclude ephemeral
  timestamps, normalized numbers, UTF-8, content hash.
- **Freeze boundaries:** validation plan → final portfolio → final model/ensemble →
  final-holdout opening → artifact manifest. Changing frozen object = new identity.
- **Schema evolution:** durable records carry schema_version; migrate known, reject unknown.
- **Non-functional:** local-first CPU; low deps; bounded RAM; determinism honesty;
  recoverability (resumable stage checkpoints); privacy (no raw values logged by default);
  observability overhead buffered/batched.

## Layer 3 — Subsystem deep specs (implemented behind the spine)

- **S1 Data intake (doc 02):** DataSourceAdapter protocol (source_id, schema_hint,
  estimates, sample/scan, seek/pushdown flags, fingerprint_material); 10-rule CSV
  algorithm; SamplePlan = head+middle+tail+pseudo-random blocks; physical vs semantic
  types; identifier evidence (distinct ratio, patterns, name hints — never alone);
  text-vs-categorical evidence; Welford streaming stats; missingness taxonomy (null /
  empty / whitespace / tokens / suspicious sentinels); exact-vs-KMV/HLL cardinality;
  quality checks; dataset meta-features; fingerprint ≠ path; FAST/STANDARD/EXACT modes;
  aspect routing; abort conditions (zero rows, constant target, dup columns, low parser
  confidence, source changed, malformed rows).
- **S2 Problem/validation/leakage (doc 03):** 9-rule task inference; metric routing
  tables (binary/multiclass/regression); split planner (stratified/K/group/time/tiny);
  holdout forbidden-uses list; progressive validation ladder (probe → 3-fold → 5-fold →
  nested finalists → untouched holdout); leakage classes L0–L5; 4-stage detector pipeline;
  cross-fit target encoding algorithm (OOF + final dev map + prior fallback); group
  aggregate safety; availability inequality max(availability_time) ≤ P_i; label-delay
  semantics; calibration then threshold order (freeze before holdout); domain-classifier
  shift detection (careful for time series).
- **S3 Feature engine (doc 04):** FeatureNode JSON contract; canonicalization (commutative
  fold, x+0→x, x*1→x, abs∘abs→abs, dedupe, cycle reject, invalid domain reject); 6
  proposal sources; trigger table; depth policy (0=raw, 1=unary/direct, 2=transform-of-
  transform, ≥3 only with strong evidence) + complexity penalty; funnel F0 preconditions →
  F1 static resource → F2 cheap materialization → F3 incremental probe → F4 progressive →
  F5 portfolio interaction → F6 final stability; constraint-aware utility formula;
  OpenFE coarse-to-fine; redundancy toolkit (signature hash, association clustering,
  lineage overlap, conditional ablation, grouped permutation — never permutation alone);
  portfolio search (greedy fwd/bwd + ablation, beam, evolutionary late); acceptance
  record fields.
- **S4 Transformation catalog (doc 05):** 95 operators, each with a 13-field contract
  (inputs, output, purpose, preconditions, fit state, transform, null/invalid policy,
  leakage class, cost shape, trigger, rejection, validation, inference state, tests).
  Families: numeric unary 1–20; interactions 21–32; categorical 33–43; datetime 44–57;
  temporal 58–72; group aggregates 73–80; text 81–89; dimensionality/cluster 90–93;
  model-informed 94–95. `shift_lead_rows` is L5 (diagnostic only, never predictive).

- **S4a Exact shift/lag/rolling procedure (doc 05):** reject k≤0; require ordering key
  (file order not meaningful); group partition; stable sort with tie-break;
  AMBIGUOUS_ORDER on duplicate times without tie semantics; first-k rows per group
  missing; NO backfill from later rows; restore original row order; record
  group/order/tie/lag/boundary in lineage; time-aware validation; label-availability
  proof for target lags; history state at inference; group boundary tests; property test
  (output at T never depends on source time >T). Time-offset lag: latest record with
  timestamp ≤ T−delta, never nearest-absolute if future possible. Rolling [T-W,T) with
  INSERT-AFTER-EMIT ordering (current row inserted only after emitting its feature).
  Target lags only if A(Y_t) ≤ P_i; unknown label delay ⇒ review-required, disabled by
  default. Temporal tests: first-k missing, no cross-group contamination, shuffled-input
  equality, duplicate-time policy, future never referenced, batch≡online, label-delay
  safety, boundary inclusion, no backfill.
- **S5 Experience store (doc 06):** Case schema (identity/context/action/result/cost/
  diagnosis); meta-feature families (shape, types, missingness, numeric dist, categorical
  structure, target, relationships, runtime); retrieval R0 hard-compat → R1 cheap buckets
  → R2 family-weighted distance → R3 learned ranking (only with many distinct datasets);
  RetrievalPrior output; Bayesian smoothing posterior=(s+α)/(n+α+β), α=β=1 default;
  26-code failure taxonomy; negative-transfer guard (priors affect order only + cheap
  probe + down-weight on failure); cost predictors (conservative, active only if they
  beat baselines); case write-back pipeline (validate → provenance → diagnose → freeze →
  atomic persist → update stats → CASE_WRITTEN); duplicate identity hash; OOD handling;
  `.fiae/` layout (experience.sqlite, cases/, profiles/, artifacts/, cache/);
  meta-learning validation = held-out whole datasets; acceptance criteria (reduces
  evaluations/fits/time-to-quality/resource failures without negative transfer).
- **S6 Search/HPO/ensembles (doc 07):** TrialSpec; fidelity dims (rows, folds,
  iterations, features, depth); ladders 5→16→40→100% / 16→32→64→100% / small-data 100%;
  promotion utility formula (quality − variance − gap − time − ram − latency − size −
  risk); successive halving with eta + uncertainty/minimum-support guards; cost-aware
  priority (EI/cost − memory risk − failure risk) + exploration quota; model routing
  table (linear, SGD, NB, tree, RF, extra trees, HistGB, LightGBM/XGB/CatBoost optional,
  MLP conditional); HPO routing (random/TPE/SH/Hyperband/GP/CMA-ES/evolutionary/
  FLAML-style); HPO spaces per family; early stopping with warmup; overfit detector
  signals + response order (check leakage/split → reduce unstable features → regularize →
  reduce capacity → more rigor → stop over-tuning); underfit detector + actions; gradient
  diagnostics conditional; imbalance (metric first, then weights/threshold/calibration/
  fold-only resampling; synthetic sampling never sees validation/test); calibration;
  threshold frozen pre-holdout; ensemble eligibility + diversity measures + greedy
  marginal add + stacking (OOF-only + stacked-overfit guard can disable); Pareto
  dominance; freeze before holdout; stopping rules; mandatory stage audit (entering/
  pruned/promoted/metrics/cost/remaining budget).

- **S7 Scheduler (doc 08):** work classes (IO_BOUND, NATIVE_CPU_RELEASES_GIL,
  PYTHON_CPU_BOUND, MEMORY_HEAVY, ACCELERATOR, REMOTE) with routing; token lifecycle
  estimate→reserve→execute→measure→release→update; memory admission formula; lazy
  feature DAG + 10 optimization passes (dead node, dedupe, constant fold,
  projection/predicate pushdown, CSE, fusion, split-safe state sharing, materialization
  boundaries, memory estimate); cache key (source fingerprint, split_id, canonical node,
  fitted-state fingerprint, build); cache tiers L0 mem → L1 mmap → L2 disk → L3 object
  store; data locality order; shared-source parallel evaluation; nested-parallelism guard
  inner_threads ≈ max(1, total/W); bounded queues + backpressure (IDs not frames);
  priority score; cancellation protocol (stop scheduling → cooperative checks → grace →
  terminate → flush audit → release tokens → mark incomplete); timeout =
  min(configured, stage, run) as status; retry transient only; async event path with
  durable flush; streaming capability flags; cost measurement; scheduler loop; executors
  Inline/LocalThread/LocalProcess; acceptance incl. serial fallback when parallel is
  slower.
- **S8 Codegen (doc 09):** compile frozen IR only, no rediscovery; PipelineIR; project
  layout (README, ARCHITECTURE.md, pyproject, config, src/contracts|data|preprocessing|
  features|model|train|evaluate|predict, tests/, artifacts/manifest.json); 22 compiler
  steps; feature parity (identity, dtype, alignment, null mask, categoricals, numeric
  tolerance); prediction parity (class mapping, prob vector, threshold, regression
  tolerance, temporal cutoff/horizon); manifest fields; dependency minimization;
  ARCHITECTURE.md required sections; verify loop (repair compiler/IR, never ad-hoc
  export patch); failure codes UNSUPPORTED_IR_NODE, IMPORT_FAILURE, TEST_FAILURE,
  FEATURE_PARITY_FAILURE, PREDICTION_PARITY_FAILURE, ARTIFACT_HASH_FAILURE,
  RESOURCE_CONSTRAINT_FAILURE; reproduction commands; LLM advisory-only.
- **S9 CLI/dashboard/observability (doc 10):** 17 CLI commands (inspect, analyze, plan,
  run, status, logs, features, models, leaderboard, pareto, explain, compare, artifacts,
  export, cancel, resume, dashboard, doctor) + `--json`; 12 dashboard pages; 4 separate
  streams (app logs, experiment records, telemetry, audit); EventEnvelope JSON; event-type
  catalog (run/data/problem/feature/trial/ensemble/codegen); privacy-safe logging; run/
  dir layout (run.json, events.jsonl, decisions.jsonl, metrics/, reports/, artifacts/);
  SQLite index + files; async batched writer; crash recovery (never infer unfinished
  trial success); MLflow/OTel optional exporters; metrics catalog; trace spans;
  DecisionRecord; SSE-first transport with cursor reconnect; loopback binding; completion
  view questions.
- **S10 Security/reliability (doc 11):** trust model (raw data, filenames, DB text, LLM
  output, generated code, plugins, serialized artifacts = untrusted); trust boundary
  chain; data minimization; secrets from env only; PII rules (no raw-cell logging, hash
  ids, opt-in raw display); generated-code sandbox (controlled dir, timeout, resource
  limits, network off, controlled env, capture, verify paths); plugin permission
  declaration; target access P0–P3; path/file safety; unsafe deserialization rules;
  dependency security (no unattended LLM pip installs); DoS bounds (line/field size,
  candidate count, depth, queue, text length, time, RAM, disk); append-only audit with
  correction references; reliability states CREATED→PLANNING→RUNNING→{COMPLETED|FAILED|
  CANCELLED|INTERRUPTED} (crash never becomes COMPLETED); checkpoint boundaries; resume
  verification (source fingerprint, config/build, checkpoint hash, cache invalidation);
  determinism honesty; overrides (leakage veto + holdout isolation are hard); incident
  record with redaction.
- **S11 Testing/benchmarks (doc 12):** 7 test layers (unit, property, integration,
  differential, e2e, perf benchmark, adversarial/failure injection); per-operator test
  matrix (nulls, dtype, empty/constant, extremes, invalid domain, determinism,
  fit/transform separation, serialization, fold safety, temporal cutoff, group boundary);
  shift/lag properties; 9-scenario leakage adversarial suite (exact copy, renamed copy,
  copy+noise, post-outcome status, non-crossfit target encoding, future aggregate,
  same-entity random split, pre-split selection, pre-split global scaler); validation
  tests; profiler tests; feature-search/HPO/ensemble/codegen/observability tests;
  performance dimensions; quality benchmark fields; reference challengers + fairness
  rule; ablation list; "100x" rule with example; research gate template + gate states
  (PROPOSED → IMPLEMENTED → CORRECTNESS_PASSED → BENCHMARKED → ACCEPTED|REJECTED);
  statistical reporting (mean/median, distribution, wins/ties/losses, compute,
  uncertainty — never only winning examples); meta-learning benchmark (held-out
  datasets); resource-predictor benchmark (catastrophic underprediction rate); failure
  injection; completion definition.

## Layer 4 — Reference knowledge (doc 14) — proposals to benchmark, never assumed wins

- AutoGluon: type-aware features, bagging/stacking, inference-limit constraints → optional
  challenger backend, no Ray/heavy core.
- H2O Driverless AI: evolutionary FE, transformer catalog, joint feature/model search →
  history-guided warm start, lazy DAG, aggressive pruning, explicit resource objectives,
  failure memory.
- DataRobot: 16%/32%/64% Autopilot stages, blueprints, data-quality checks → generalized
  fidelity, permanent candidate/failure evidence, DAG + cost.
- SageMaker Autopilot: AutoGluon ensembling, Bayesian/multi-fidelity HPO, candidate
  notebooks → machine-readable DecisionRecords/IR, cloud-independent multi-fidelity.
- OpenFE: incremental evaluation, two-stage pruning for huge candidate pools → cheap
  evidence pruning + cost/leakage/stability/latency/memory utility.
- FLAML: cost-aware search order (cheap trials early) → EI-per-cost + memory/failure risk.
- Optuna: Median/SH/Hyperband pruners, distributed patterns → optional HPO backend,
  FIAE-owned scheduler interface.
- scikit-learn: split before fitted preprocessing; nested CV; successive halving;
  permutation importance caveats (correlated masking) → fold-safety contract, strong
  validation only on shortlist, importance never alone.
- Polars: lazy optimizer (pushdown, CSE), streaming → logical Feature DAG, optional
  accelerated adapter.
- DuckDB: CSV sniffer (20,480-row default, positional sampling on seekable files) →
  avoid head-only inference, optional robust CSV accelerator.
- MLflow / OpenTelemetry: run/params/metrics/artifacts; traces/metrics/logs → optional
  exporters only.

## Layer 5 — Cross-cutting mechanics (doc 01)

- **Error taxonomy (24 codes):** DATA_FORMAT_ERROR, SCHEMA_AMBIGUITY,
  SOURCE_CHANGED_DURING_RUN, TARGET_MISSING, TARGET_INVALID, TASK_AMBIGUOUS,
  SPLIT_INVALID, LEAKAGE_CONFIRMED, PREDICTION_TIME_UNAVAILABLE,
  FEATURE_PRECONDITION_FAILED, FEATURE_DOMAIN_ERROR, FEATURE_CARDINALITY_EXPLOSION,
  RESOURCE_PRECHECK_FAILED, TRIAL_TIMEOUT, TRIAL_OOM, MODEL_FIT_FAILED,
  METRIC_UNDEFINED, HPO_INVALID, ENSEMBLE_INVALID, CALIBRATION_INVALID, CODEGEN_MISMATCH,
  REPRODUCIBILITY_MISMATCH, ARTIFACT_CORRUPTION, CANCELLED. Each carries code, severity,
  component, run/stage/trial/feature IDs, recoverable flag, retry policy, human reason,
  machine-readable evidence.
- **Resource hierarchy:** machine safety reserve → user hard limits → run budget → stage
  budget → worker budget → trial budget → feature-materialization budget.
- **Dependency tiers:** T0 stdlib; T1 NumPy/sklearn/joblib (Polars optional-preferred);
  T2 optional learners/search (LightGBM, XGBoost, CatBoost, Optuna); T3 optional
  challengers (AutoGluon, H2O DAI, DataRobot, SageMaker adapters); T4 optional ops
  (MLflow, OTel, PostgreSQL, object storage, distributed schedulers).

## Layer 6 — Definition of done (docs 00 §7, 10, 12, 15)

- **Output contract (17 artifacts):** run_manifest.json, data_profile.json, schema.json,
  problem_definition.json, validation_plan.json, leakage_report.json, aspect_plan.json,
  experience_prior.json, feature_graph.json, feature_lineage.json,
  feature_leaderboard.jsonl/parquet, model_leaderboard.jsonl/parquet,
  ensemble_report.json, pareto_frontier.json, final_pipeline.json, architecture.md,
  generated_project/, tests/, benchmark_report.json, events.jsonl, audit.jsonl,
  metrics.jsonl.
- **Completion contract:** problem definition + validation + selected pipeline evidence +
  final-evaluation status + artifact manifest + audit summary internally consistent;
  budget-exhausted best-so-far labeled as such.
- **Research loop per subsystem:** search official docs/papers → hypothesis → minimal
  implementation → tests → benchmark → alternatives → ablation → accept/reject → record
  decision. Gates, not versions.

## Build plan (dependency-safe milestones, each gated per doc 12/00 §9)

| # | Milestone | Delivers | Docs |
|---|---|---|---|
| M0 | Repo skeleton + contracts | package layout, pyproject, typed contracts, error taxonomy, IDs/fingerprints, event bus + JSONL/SQLite, run dir, run state machine | 13, 01, 10 |
| M1 | Data intake | adapters (CSV first), sampling, dialect/dtype/semantic inference, streaming stats, quality checks, fingerprints, `fiae inspect/analyze` | 02 |
| M2 | Problem + validation + leakage | task inference, metric routing, split planner, holdout policy, detector stages A–D, cross-fit encoding, findings | 03 |
| M3 | Feature IR + first operators | FeatureNode, canonicalization, DAG, op registry, catalog slice incl. lag/rolling, temporal property tests | 04, 05 |
| M4 | Progressive feature search | funnel F0–F3, cheap gates, probes, portfolio greedy/beam, redundancy clustering, leaderboard | 04, 07 |
| M5 | Scheduler + trials | token scheduler, executors, lazy materialization + DAG passes + cache, TrialSpec/Result, cost model | 08 |
| M6 | Model routing + HPO + ensembles | model registry, fidelity ladder, SH/Hyperband, pruners, diagnostics, calibration/threshold, ensemble + stack guard, Pareto, freeze, holdout + refit | 07 |
| M7 | Experience store | cases, write-back, retrieval R0–R2, smoothing, negative-transfer guard, `.fiae/` store | 06 |
| M8 | Codegen + verification | PipelineIR compiler, generated project, parity tests, sandbox, manifest, `fiae export` | 09 |
| M9 | CLI/dashboard/audit completion | remaining CLI, dashboard (SSE), DecisionRecords, crash recovery, privacy fixtures | 10, 11 |
| M10 | Testing/benchmark harness | 7-layer suite, leakage adversarial suite, ablation runner, benchmark reporting, research gates | 12 |

## Progress log (update after every completed milestone)

### M0 DONE — Repo skeleton + contracts (docs 13, 01, 10)
- `src/fiae/`: `contracts.py` (typed schemas: DataSourceSpec, ColumnProfile, DatasetProfile, ProblemDefinition, ConstraintSpec, ValidationPlan, LeakageFinding/Class/Severity, FeatureNode, FittedStateRef, FeaturePortfolio, ModelSpec, FidelitySpec, TrialSpec/Result, EnsembleSpec, MetricValue, ResourceMeasurement, DecisionRecord, EventEnvelope, ArtifactRef, RunManifest, PredictionContext, FIAEError, FitScope, TargetPermission), `errors.py` (error taxonomy + FIAEError), `ids.py` (stable IDs, content_hash, canonical serialization, dataset/split fingerprints), `events.py` (EventEnvelope, event types, async batched writer, JSONL + SQLite, durable flush for audit), `runs.py` (run dir layout, state machine CREATED→PLANNING→RUNNING→COMPLETED/FAILED/CANCELLED/INTERRUPTED), `cli.py` + `__main__.py` (fiae CLI), `pyproject.toml`, `conftest.py`.
- Python 3.12 at `C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe`; run tests with it (`-m pytest tests/`); PYTHONPATH not needed (conftest inserts src). No numpy/sklearn yet in core — stdlib only so far.

### M1 DONE — Data intake (doc 02)
- `src/fiae/intake/`: `base.py` (DataSourceAdapter protocol, SamplePlan, RowBatch), `csv_source.py` (BOM/encoding, delimiter sniffing incl. quoted delimiters/newlines, header inference, multi-region sampling for seekable files, malformed-row accounting), `stats.py` (Welford streaming numerics, min/max/finite/zero fractions, KMV-style bounded cardinality sketch, exact-vs-approx flags), `typing_engine.py` (physical candidates elimination + semantic roles incl. identifier evidence, text-vs-categorical), `profiler.py` (bounded FAST/STANDARD/EXACT modes, missingness tokens, quality findings, meta-features, dataset+schema fingerprints, ProfileResult), CLI `inspect`/`analyze` wired.

### M2 DONE — Problem + validation + leakage (doc 03)
- `src/fiae/problem/`: `task.py` (task inference with confidence+reasons, explicit override, metric routing incl. guarded MAPE/RMSLE, positive-class resolution, ambiguity surfacing instead of silent guess), `splits.py` (stratified/K-fold/GroupKFold/time-ordered backtest split planner, final-holdout policy reserved before any fitting, split fingerprint, deterministic seeds, tiny-data policy), `leakage.py` (Stage A deterministic: exact copy, boolean-renamed copy, identifier-like bijection with distinct-ratio>=0.9 gate, constant-residue target-embedded strings; Stage B statistical triage: rank AUC, mutual information, missingness bias — suspicion only, REVIEW_REQUIRED; Stage C prediction-time availability + confirm_unavailable hard reject + target-history label-availability rule; Stage D fold-safety report over FeatureNode fit scopes; CrossFitTargetEncoder: seeded fold assignment, group-aware folds, m-estimate smoothing, OOF fit_transform + fit_final/transform, unknown→prior), `calibration.py` (brier/log_loss, BinnedCalibrator with PAV monotone projection + transform, optimize_threshold f1/precision_floor/recall_floor/utility with fallback warning, freeze_decision_policy fingerprint, distribution_shift_probe).
- Key lesson: a low-cardinality categorical that perfectly maps to target is Stage B suspicion (doc 03), NOT Stage A hard reject — only near-unique identifier-like values are bijection proof.
- Tests: 128 passing (`python -m pytest tests/`).

### M3 DONE — Feature IR + first operators (docs 04, 05)
- `src/fiae/features/`: `registry.py` (FeatureOperator 13-field contract: name, family, arity, input/output types, purpose, preconditions, fit_scope, null_policy, leakage_class, target_permission, cost_shape, trigger, rejection, validation, inference_requirement, transform, mandatory_tests, commutative, domain_check, time_semantics), `canonical.py` (canonical_inputs commutative fold, feature_signature, signature_hash = content hash, make_feature_node with deterministic feature_id, drop_identity_inputs x+0->x / x*1->x), `dag.py` (FeatureDAG: add_or_create, duplicate collapse, cycle detection via graph cycle check, topological_order, roots/leaves), `ops_numeric.py` (27 L0 unary/interaction ops: identity, log1p+domain, signed_log1p, sqrt+domain, cbrt, square/cube overflow guards, abs, sign, reciprocal, exp_clip, zero/missing/finite indicators, sum/difference/product/safe_ratio+domain/relative_difference/min/max/mean/harmonic/geometric/euclidean/absolute_difference), `ops_datetime.py` (parse helper + year/month/quarter/day_of_week/day_of_month/hour/is_weekend/is_month_start/is_month_end), `ops_temporal.py` (lag/lead/diff/pct_change + rolling_mean/sum/std/max/min/count with no-future backward windows; time_semantics="sequential").
- 46 operators registered across numeric / numeric interaction / datetime / temporal families.
- Tests: `test_features.py` with 89 tests (registry 13-field contract, canonicalization, DAG acyclic + duplicate collapse, all transform semantics, domain checks).

### M4 → deferred (see NEXT) — Progressive feature search (docs 04, 07)

### M7 DONE — Experience Store + Meta-Learning + Retrieval (doc 06)
- `src/fiae/experience/`: `case.py` (FailureTag 25-code taxonomy; CaseContext/ Action/ Result/ Cost/ Diagnosis; CaseRecord with is_success + identity_hash from dataset+split+feature+model+hyperparms+seed), `metafeatures.py` (MetaFeatureFamily: shape/type_composition/missingness/numeric_distribution/categorical_structure/target/relationship/runtime; DatasetMetaFeatures normalized families; extract_meta_features; meta_feature_distance weighted by family), `priors.py` (FeatureFamilyPrior/ ModelFamilyPrior with Bayesian update; RetrievalPrior: feature/transformation/model/hyperparameter/split/ensemble/cost/failure priors + confidence/ood_score/source_count; bayesian_posterior), `store.py` (ExperienceStore SQLite-backed: write_case with identity dedup, get_case, get_by_identity, count, all_cases, find_by_dataset, clear), `retrieval.py` (R0 hard filter by task/time/group; R1 cheap bucket ranking; R2 weighted meta-feature distance; RetrievalPrior aggregation via Bayesian smoothing; OOD confidence guard blending to generic; RetrievalQuery; retrieve_priors convenience).
- Key lesson (doc 06): identity hash excludes case_id — repeated identical runs dedup for variance estimation but never count as distinct-dataset evidence.
- Tests: `test_experience.py` with 19 tests (store CRUD round-trip, dedup, meta-feature distance, Bayesian priors, retrieval confidence / hard filter / family priors).

### Tests total: 236 passing (`python -m pytest tests/`).

### NEXT — M4: Progressive Feature Search (docs 04, 07)
- Wire the FeatureDAG + operator registry into: feature hypothesis generation (proposal sources + triggers), cheap gates F0–F2 (schema/type/domain/identifier, static resource, constant/duplicate hash), progressive evaluation F3–F6 with fold-safe fit, portfolio search (greedy/beam), and the FIAE runtime loop. Then M5 scheduler, M6 search/HPO/ensembles, M8 codegen, M9 CLI/dashboard, M10 test harness.





