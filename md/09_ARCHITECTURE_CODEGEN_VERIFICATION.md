# 09 — Architecture Synthesis, Code Generation, Reproduction, and Verification

## Rule
Codegen compiles frozen evidence-backed IR. It does not rediscover the pipeline from prose.

## Input
- DataContract
- ProblemDefinition
- ValidationPlan
- frozen FeatureGraph
- selected ModelSpec / EnsembleSpec
- calibration/threshold state
- schema contract
- deployment constraints

## Pipeline IR
```text
PipelineIR
├── source adapter
├── input schema
├── preprocessing graph
├── feature graph
├── model/ensemble graph
├── postprocessing
├── prediction contract
└── monitoring contract
```
Each node has stable ID, inputs, params, fit state, output schema, seed if stochastic, and lineage.

## Generated project
```text
project/
├── README.md
├── ARCHITECTURE.md
├── pyproject.toml
├── config/
├── src/
│   ├── contracts.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
│   ├── test_contracts.py
│   ├── test_features.py
│   ├── test_prediction_equivalence.py
│   └── test_edge_cases.py
└── artifacts/manifest.json
```

## Compiler steps
1. validate frozen IR,
2. topologically sort,
3. deduplicate common nodes,
4. partition fit-time/transform-time state,
5. select implementation backend,
6. emit input contract,
7. emit preprocessing,
8. emit feature functions,
9. emit training/loading,
10. emit prediction/postprocessing,
11. emit evaluation,
12. emit tests,
13. emit ARCHITECTURE.md,
14. emit dependencies from used nodes only,
15. emit artifact manifest,
16. syntax/import check,
17. run tests,
18. feature parity test,
19. prediction parity test,
20. clean-process smoke test,
21. latency/resource test,
22. package only after mandatory gates pass.

## Feature parity
For frozen verification sample compare FIAE-runtime vs exported project:
- feature identities,
- output dtype,
- row alignment,
- null mask,
- categorical values,
- numeric tolerance.

## Prediction parity
Classification:
- class mapping,
- probability vector,
- threshold behavior.
Regression:
- numeric tolerance.
Temporal:
- cutoff/horizon identity.

## Manifest
```text
artifact type
artifact SHA/hash
producer run
engine commit
library versions
feature graph hash
training dataset fingerprint
selected pipeline hash
trust level
```

## Dependency minimization
If final pipeline does not use a library, exported inference should not depend on it. Never export all FIAE just because the search engine used it.

## Generated ARCHITECTURE.md
Must state:
- task/objective,
- input schema,
- prediction-time assumptions,
- split strategy,
- accepted features + lineage,
- rejected critical leakage features,
- model/ensemble,
- calibration/threshold,
- metrics,
- resource profile,
- known limitations,
- monitoring/retraining signals.

## Verification loop
```text
generate
→ import/static check
→ tests
→ feature parity
→ prediction parity
→ benchmark
→ if fail: repair compiler/IR logic
→ regenerate affected output
```
Do not make ad-hoc patches to the export that are not represented in canonical compiler logic.

## LLM role
Allowed: propose code, documentation, optimization ideas.
Not authoritative: metric result, feature acceptance, leakage, benchmark, test result.

## Failure codes
```text
UNSUPPORTED_IR_NODE
IMPORT_FAILURE
TEST_FAILURE
FEATURE_PARITY_FAILURE
PREDICTION_PARITY_FAILURE
ARTIFACT_HASH_FAILURE
RESOURCE_CONSTRAINT_FAILURE
```

## Reproduction
```bash
fiae export RUN --project out/
cd out
python -m pytest
python -m src.evaluate --manifest artifacts/manifest.json
```
A non-reproducible run is not a finished engineering artifact.
