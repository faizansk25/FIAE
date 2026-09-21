<div align="center">

<img src="assets/fiae-logo.png" alt="FIAE" width="414"/>

# FIAE

### Feature Intelligence & Architecture Engine

**Automated feature engineering with safety-first design, experience memory, and verified pipeline export.**

[![CI](https://github.com/faizansk25/FIAE/actions/workflows/ci.yml/badge.svg)](https://github.com/faizansk25/FIAE/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Source-Available](https://img.shields.io/badge/license-source--available-orange.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-860%20passing-brightgreen.svg)](#testing)
[![Operators](https://img.shields.io/badge/operators-95-purple.svg)](#operator-catalog)
[![security](https://img.shields.io/badge/scanned%20by-gitleaks-informational.svg)](.github/workflows/ci.yml)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

---

*FIAE does not guess. It probes, validates, and proves every feature is safe, stable, and complementary before it enters your model.*

</div>

---

## Table of Contents

- [Why FIAE](#why-fiae)
- [Key Differentiators](#key-differentiators)
- [Features](#features)
- [Quick Start](#quick-start)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [CLI Reference](#cli-reference)
- [Data Sources](#data-sources)
- [Core Concepts](#core-concepts)
  - [6-Class Leakage Taxonomy](#6-class-leakage-taxonomy)
  - [Multi-Stage Funnel](#multi-stage-funnel)
  - [Experience Memory](#experience-memory)
  - [Fitted-State Pipeline](#fitted-state-pipeline)
- [Operator Catalog](#operator-catalog)
- [Model Registry](#model-registry)
- [Design Documents](#design-documents)
- [Development](#development)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Citation](#citation)
   - [License](#license)
   - [Support FIAE](#support-fiae)

> **New here?** Skim [Why FIAE](#why-fiae), then run the 60-second end-to-end
> example — no credentials, no GPU: [`examples/churn/run_api.py`](examples/churn/run_api.py).
> Contributing? Start with [CONTRIBUTING.md](CONTRIBUTING.md) — it contains the
> exact three commands CI runs, so you can validate a PR before opening it.

---

## Why FIAE

Feature engineering is the highest-leverage step in machine learning — and the most dangerous. A single leaked feature, a silent data-snooping bug, or a redundant transformation can silently corrupt an entire model pipeline.

Existing tools focus on **coverage** (how many features can I generate?) or **speed** (how fast can I search?). FIAE is the first tool to focus on **correctness** — proving that every generated feature is:

1. **Leakage-free** — validated against a 6-class leakage taxonomy (L0–L5)
2. **Stable** — passes multi-seed cross-validation (F6 gate)
3. **Complementary** — not redundant with existing features (F5 gate)
4. **Verified** — exported pipeline passes 11 automated verification gates (including operator-coverage and row-level feature parity)
5. **Reproducible** — deterministic IDs, content hashes, and snapshot isolation

FIAE is built for **production ML systems where correctness matters more than speed** — healthcare, finance, autonomous systems, and any domain where a silent bug has real consequences.

### Proven end-to-end (v0.0.2)

On a 400-row synthetic churn dataset, the full canonical 10-phase pipeline — intake → profile → validate → split → generate → funnel → train 4 model families with 5-fold CV → ensemble → evaluate → verified export — completes in ~5 seconds:

| Stage | Measured result |
|---|---|
| Funnel | 17 proposals → 15 pass F2 materialization → 4-feature portfolio (F5-deconflicted) |
| Model selection (HPO) | 4 families trained with real 5-fold CV; random forest promoted at **ROC-AUC 0.713** — beating the 0.691 raw-input baseline |
| Evaluation | fold-std 0.059, generalization-gap estimate 0.030, calibration valid |
| Export | **11/11 verification gates pass**, including row-level feature parity between runtime and exported code |
| Determinism | md5 feature bucketing + seeded CV; identical inputs → identical pipeline IDs, hashes, and exports |

Run it yourself in under a minute: `python examples/churn/run_api.py` (no credentials, no GPU, CPU-only).

---

## Key Differentiators

| Capability | FIAE | Featuretools | AutoFeat | getML | Auto-sklearn | FLAML |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **6-class leakage taxonomy** (L0–L5) | ✅ | ❌ | ❌ | ❌ | ⚠️ | ❌ |
| **Cross-dataset experience memory** | ✅ | ❌ | ❌ | ❌ | ⚠️ | ❌ |
| **Complementarity-aware selection** (F5) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Verified pipeline export** (11 gates incl. row-level parity) | ✅ | ❌ | ❌ | ⚠️ | ❌ | ❌ |
| **Runtime security enforcement** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Fitted-state pipeline** (fit/transform) | ✅ | ❌ | ❌ | ❌ | ⚠️ | ❌ |
| **Deterministic exports** (md5 feature bucketing, seeded CV) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Deterministic content-hash IDs** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Core: zero third-party deps** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **95 typed operators** | ✅ | ~50 | ~30 | ~20 | — | — |
| **25+ data source connectors** (files, URLs, cloud, SQL, warehouses, APIs) | ✅ | ⚠️ | ❌ | ⚠️ | ❌ | ❌ |
| **Progress indicators with ETA** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |

**FIAE does not try to be the fastest or the most feature-rich. It is the safest and most principled.**

---

## Features

### 🌐 Universal Data Source Support
Load data from anywhere — local files, URLs, cloud storage, databases, or DataFrames:

```bash
# Local file
fiae inspect data.csv

# URL (with download progress bar)
fiae inspect https://raw.githubusercontent.com/user/repo/data.csv

# Cloud storage
fiae inspect s3://bucket/data.csv
fiae inspect gs://bucket/data.csv

# Database
fiae inspect postgresql://user:pass@host/db
```

### 📊 Progress Indicators
Real-time progress bars with ETA for downloads and data processing:

```
  ↓ Downloading: [████████████████████████████░░] 85.3% 12.5/14.7 MB 2.3 MB/s ETA: 1s
  ⚙ Processing: [████████████████████████████░░] 85.3% 3700/4350 rows 12500 rows/s ETA: 0s
```

### 🔒 Safety-First Design
- 6-class leakage taxonomy (L0–L5)
- Multi-stage funnel with policy-driven gates
- Complementarity-aware feature selection
- Verified pipeline export (11 gates)

### 🧠 Experience Memory
Cross-dataset experience store with R0–R3 retrieval for intelligent feature hints.

### 📦 Verified Pipeline Export
Export to standalone sklearn projects with 11 automated verification gates — including row-level feature parity between the fitted runtime and the exported code.

---

## Quick Start

```bash
# Install
pip install -e ".[dev,tier1]"

# Profile a dataset
fiae inspect data.csv

# BIG Data: hard row limit + memory-tuning batch size
fiae inspect big.csv --max-rows 1000000 --chunk-size 512

# Analyze with target
fiae analyze data.csv --target is_returned

# Run the full feature intelligence pipeline
fiae learn data.csv --target is_returned

# Run with JSON output (for automation)
fiae learn data.csv --target is_returned --json
```

**Example output:**

```
source:      csv|/data/ecommerce.csv
target:      is_returned (binary_classification, confidence=1.0)
rows:        5,322 observed
columns:     20
metrics:     roc_auc, log_loss, brier
proposals:   1,805 generated, 200 unique
funnel:      200 passed F2
portfolio:   8 features selected
  log1p(price): fold_stability=0.8721 [STABLE]
  sqrt(weight): fold_stability=0.8403 [STABLE]
  ...
time:        1.62s
```

---

## Getting Started

### Starting FIAE

Run `fiae` with no arguments to see the branded splash screen and all available commands:

```bash
fiae
```

This displays the FIAE wordmark and full command list. You can also show just the brand banner:

```bash
fiae banner              # Full splash with tagline box
fiae banner --no-tagline # Wordmark only
fiae --version           # Version info
```

### First Steps

```bash
# 1. Profile a dataset
fiae inspect data.csv

# 2. Show the FIAE brand
fiae banner

# 3. Run the full pipeline
fiae learn data.csv --target is_returned
```

---

## Installation

### From source (recommended)

```bash
git clone https://github.com/faizansk25/FIAE.git
cd FIAE
pip install -e ".[dev]"
```

### Dependency tiers

FIAE follows a strict dependency policy — **core imports require zero third-party packages** (NFR-002). Optional capabilities are gated behind dependency tiers:

| Tier | Packages | Purpose |
|---|---|---|
| **Core** | *(none — stdlib only)* | Intake, profiling, type inference, leakage detection, funnel, probe, codegen |
| **Tier 1** | `numpy`, `scikit-learn`, `joblib`, `polars` | Model training, HPO, sklearn operators, progressive CV |
| **Tier 2** | `lightgbm`, `xgboost`, `catboost`, `optuna` | Gradient boosting model families, advanced HPO |
| **Dev** | `pytest>=7.0` | Test suite |

> **Dependency policy:** Core imports require zero third-party packages (NFR-002). Every heavy capability is behind an optional tier gate.

```bash
# Core only (zero dependencies)
pip install -e .

# With ML capabilities
pip install -e ".[tier1]"

# With URL/cloud/database support
pip install -e ".[tier1,requests]"

# Full stack
pip install -e ".[tier1,tier2]"
```

---

## CLI Reference

### `fiae inspect SOURCE`

Profile a data source — detect encoding, delimiter, header, physical/semantic types, and quality findings.

```bash
fiae inspect data.csv                    # Human-readable output
fiae inspect data.csv --json             # Machine-readable JSON
fiae inspect data.csv --mode standard    # Larger sample (FAST|STANDARD|EXACT)
fiae inspect big.csv --max-rows 1000000  # Hard row limit for large datasets
fiae inspect big.csv --chunk-size 512    # Smaller batches = lower memory use
fiae inspect wide.csv --scheduler threads  # Parallel per-column accumulation

# Experiment tracking (audit trail + optional MLflow/W&B export)
fiae learn data.csv --target y --track local            # Local run log + audit.jsonl
fiae learn data.csv --target y --track mlflow           # Also export to MLflow
fiae learn data.csv --target y --track wandb            # Also export to Weights & Biases
fiae learn data.csv --target y --track local --run-name exp1 --experiment myproj

# URL with progress bar
fiae inspect https://raw.githubusercontent.com/user/repo/data.csv
```

### `fiae analyze SOURCE --target Y`

Profile source and verify target column — infer task type (classification/regression) and route metrics.

```bash
fiae analyze data.csv --target is_returned
fiae analyze data.csv --target is_returned --json
```

### `fiae learn SOURCE --target Y`

Run the full feature intelligence pipeline — generate, validate, and select features.

```bash
fiae learn data.csv --target is_returned
fiae learn data.csv --target is_returned --max-features 12
fiae learn data.csv --target is_returned --mode exact
fiae learn data.csv --target is_returned --json

# URL source
fiae learn https://raw.githubusercontent.com/user/repo/data.csv --target label
```

### `fiae validate SOURCE`

Validate data source schema and detect quality issues.

```bash
fiae validate data.csv
fiae validate https://raw.githubusercontent.com/user/repo/data.csv
```

### `fiae leakage SOURCE --target Y`

Detect feature leakage using statistical and deterministic tests.

```bash
fiae leakage data.csv --target is_returned
```

### `fiae pipeline SOURCE --target Y`

Run full pipeline with Pipeline IR export.

```bash
fiae pipeline data.csv --target is_returned --output my_pipeline/
```

### `fiae export IR_PATH --project OUT/`

Export Pipeline IR to standalone sklearn project.

```bash
fiae export my_pipeline/pipeline_ir.json --project my_project/
```

### `fiae optimize SOURCE --target Y`

Run HPO + ensemble selection.

```bash
fiae optimize data.csv --target is_returned
```

### Other commands

```bash
fiae banner              # Print the FIAE wordmark
fiae experience show     # Display experience store contents
fiae status run_path     # Show run status
fiae portfolio report    # Display feature portfolio
fiae report report       # Generate pipeline report
fiae benchmarks          # Run benchmark suite
fiae codegen             # Code generation & operator tests
```

### `fiae connect` — Universal Source Connector

One command for all 25+ supported sources: auto-detects the adapter, verifies
the connection, and profiles a sample — so you know a source works before
running a pipeline on it.

```bash
fiae connect data.csv                       # local file
fiae connect s3://bucket/data.csv           # cloud storage
fiae connect postgresql://u:p@host/db --table orders
fiae connect bigquery://proj/ds --table events
fiae connect snowflake://u:p@acct/db/s --table orders
fiae connect redshift://u:p@host:5439/db --table sales
fiae connect https://api.example.com/users --header "Authorization: Bearer tok"
fiae connect --list                         # show all supported sources
```

Example output:

```
fiae 0.0.1 -- Feature Intelligence & Architecture Engine
==========================================================
Connecting to postgresql://u:p@host/db

adapter:  SqlAdapter
type:  sql
source_id:  sql|postgresql|orders
pushdown:  supported

--- Verification Profile ---

status:   PASS  connected and readable
rows observed:  10,000
columns:  12
fingerprint:  ds_7403400d2858037a5b426...

semantic types:  count, currency_like, identifier, low_cardinality_categorical

time:  0.41s
```

### `fiae serve` — REST API + Web Dashboard

Zero-dependency HTTP server (Python stdlib only). Binds to `127.0.0.1` by
default — the dashboard and API never expose your data to the network
unless you explicitly pass `--host 0.0.0.0`.

```bash
fiae serve                       # http://127.0.0.1:8420
fiae serve --port 9000           # custom port
fiae serve --runs-dir runs/      # serve a different runs directory
```

**Endpoints**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | HTML dashboard (runs + jobs, dark purple theme) |
| GET | `/api/health` | Liveness + version |
| GET | `/api/runs` | List tracked runs (`--track local` runs) |
| GET | `/api/runs/<name>` | Run detail: params, metrics, audit count |
| POST | `/api/learn` | Launch async learn job (JSON body) |
| GET | `/api/jobs` | List jobs |
| GET | `/api/jobs/<id>` | Job status / result |

**Example — launch a learn job via the API:**

```bash
curl -X POST http://127.0.0.1:8420/api/learn \
  -H "Content-Type: application/json" \
  -d '{"source": "data.csv", "target": "y", "max_rows": 100000}'

# then poll
curl http://127.0.0.1:8420/api/jobs/<job_id>
```

Security: path traversal in run names is rejected, only JSON artifacts
inside the runs root are served, and job bodies are size-capped at 1 MB.

#### Concurrency & Scale

The server is built for many simultaneous clients (doc 08):

- **Bounded worker pool** — learn jobs run on fixed worker threads
  (`--max-workers`, default 4), never one-thread-per-request
- **Backpressure** — when the pending-job queue is full (`--max-queue`,
  default 100), the server returns **429 Too Many Requests** with a retry
  hint instead of degrading or exhausting memory
- **Per-client rate limiting** — sliding-window limiter
  (`--rate-limit`, default 50 req/s per client IP)
- **Bounded memory** — completed jobs are pruned beyond 1,000 entries
- **Deep TCP backlog** (128) — burst connects are held, not dropped

Verified by load test: **200 simultaneous clients × 5 jobs = 1,000
submissions in 2.3s — 0 errors, 0 failed jobs, backpressure engaged exactly
at queue capacity.**

```bash
# Scale workers for a busy team server
fiae serve --max-workers 16 --max-queue 500 --rate-limit 200
```

Live pool stats are exposed at `GET /api/health` and on the dashboard:
workers, busy count, queued jobs, and job counts by state.


---

## Data Sources

FIAE supports multiple data source types with auto-detection:

### Local Files
```bash
fiae inspect data.csv
fiae inspect data.tsv
fiae inspect data.json
fiae inspect data.parquet
```

### URLs (HTTP/HTTPS)
```bash
# Direct file URLs
fiae inspect https://raw.githubusercontent.com/user/repo/data.csv

# Shows download progress with ETA
# ↓ Downloading: [████████████████████████████░░] 85.3% 12.5/14.7 MB 2.3 MB/s ETA: 1s

# Shows helpful error for HTML pages (e.g., GitHub blob URLs)
fiae inspect https://github.com/user/repo/blob/main/data.csv
# ERROR: URL returned HTML page instead of raw data. Use: https://raw.githubusercontent.com/...
```

### Cloud Storage
```bash
fiae inspect s3://bucket/data.csv          # AWS S3
fiae inspect gs://bucket/data.csv          # Google Cloud Storage
fiae inspect az://container/data.csv       # Azure Blob
```

Cloud objects are **streamed** in 256 KB chunks — CSV/JSONL files of any size
are processed with constant memory (never fully loaded into RAM), with a
download progress bar showing MB/s and ETA. Parquet is buffered in memory
because the format requires random access.

### Docker / Kubernetes
```bash
docker build -t fiae:latest .
docker run --rm -v ${PWD}:/data fiae:latest inspect /data/data.csv
docker run --rm fiae:latest inspect s3://bucket/data.csv
```
The image ships with cloud SDKs (boto3, google-cloud-storage,
azure-storage-blob) and runs as a non-root user, so it drops directly into
Kubernetes Jobs/CronJobs for scheduled profiling pipelines.

### Databases
```bash
fiae inspect sqlite:///path/to/db.sqlite
fiae inspect postgresql://user:pass@host/db
fiae inspect mysql://user:pass@host/db
```

### Data Warehouses
```bash
fiae inspect bigquery://project/dataset --table events
fiae inspect snowflake://user:pass@account/db/schema --table orders
fiae inspect redshift://user:pass@host:5439/db --table sales
```

Warehouse scans stream in batches with constant memory, support
predicate/projection pushdown, and use thread-local connections so a single
adapter instance can be scanned concurrently from multiple workers.

### DataFrames
```python
from fiae.intake import auto_adapter
import pandas as pd

df = pd.read_csv("data.csv")
adapter = auto_adapter(df)
```

---

## Core Concepts

### 6-Class Leakage Taxonomy

FIAE classifies feature leakage into 6 classes (L0–L5):

| Class | Description | Action |
|---|---|---|
| **L0** | No leakage | ✅ Accept |
| **L1** | Target in feature names | ❌ Reject |
| **L2** | Target-derived features | ❌ Reject |
| **L3** | Future information | ❌ Reject |
| **L4** | Data snooping | ❌ Reject |
| **L5** | Suspicious correlation | ⚠️ Review |

### Multi-Stage Funnel

Features pass through multiple validation gates:

- **F0**: Leakage detection
- **F1**: Stability check
- **F2**: Complementarity test
- **F4**: Incremental gain
- **F6**: Cross-validation stability

### Experience Memory

Cross-dataset experience store with R0–R3 retrieval:
- **R0**: Exact match
- **R1**: Similar dataset
- **R2**: Similar features
- **R3**: Domain hints

### Fitted-State Pipeline

FIAE exports pipelines with fit/transform lifecycle:
- `fit()` — learn transformations from training data
- `transform()` — apply to new data
- `fit_transform()` — fit and transform in one step

---

## Operator Catalog

95 typed feature operators across 13 families:

| Family | Count | Examples |
|---|---|---|
| **Numeric** | 12 | log1p, sqrt, square, reciprocal, zscore, minmax |
| **Categorical** | 10 | onehot, label, frequency, target_encoding, ordinal |
| **Datetime** | 8 | year, month, day, hour, weekday, is_weekend |
| **Text** | 10 | length, word_count, char_count, uppercase_count |
| **Group** | 8 | group_mean, group_sum, group_count, group_std |
| **Temporal** | 7 | lag, rolling_mean, rolling_std, expanding_mean |
| **Model-Informed** | 8 | pca, svd, kmeans, cluster_distance |
| **Interactions** | 6 | multiply, divide, add, subtract |
| **Binning** | 5 | uniform, quantile, kmeans_binning |
| **Missing** | 4 | is_missing, missing_count, missing_rate |
| **Rolling** | 5 | rolling_min, rolling_max, rolling_median |
| **Cumulative** | 4 | cumsum, cummax, cummin, cumcount |
| **String** | 6 | contains, startswith, endswith, regex_match |
| **Custom** | 2 | custom_transform, custom_aggregation |

---

## Model Registry

12 model families supported:

| Family | Models |
|---|---|
| **Linear** | LogisticRegression, Ridge, Lasso, ElasticNet, SGD |
| **Tree** | DecisionTree, ExtraTree |
| **Ensemble** | RandomForest, ExtraTrees, GradientBoosting |
| **Boosting** | AdaBoost, Bagging |
| **SVM** | SVC, LinearSVC |
| **Naive Bayes** | GaussianNB, MultinomialNB, BernoulliNB |
| **Nearest Neighbors** | KNeighbors |
| **Neural Network** | MLP |
| **Discriminant Analysis** | LinearDiscriminant, QuadraticDiscriminant |
| **LightGBM** | LGBMClassifier, LGBMRegressor [tier2] |
| **XGBoost** | XGBClassifier, XGBRegressor [tier2] |
| **CatBoost** | CatBoostClassifier, CatBoostRegressor [tier2] |

---

## Design Documents

FIAE is built on 16 design documents (md/00–15):

| Document | Title |
|---|---|
| `md/00_MASTER_BLUEPRINT.md` | Master blueprint & 15 hard principles |
| `md/01_ARCHITECTURE.md` | System architecture |
| `md/02_INTAKE.md` | Data intake & profiling |
| `md/03_LEAKAGE.md` | 6-class leakage taxonomy |
| `md/04_FUNNEL.md` | Multi-stage funnel |
| `md/05_OPERATORS.md` | Operator catalog |
| `md/06_EXPERIENCE.md` | Experience memory |
| `md/07_HPO.md` | HPO & model training |
| `md/08_PARALLEL.md` | Parallel execution |
| `md/09_PIPELINE.md` | Pipeline IR & export |
| `md/10_CLI.md` | CLI design |
| `md/11_SECURITY.md` | Security sandbox |
| `md/12_TESTING.md` | Testing strategy |
| `md/13_ROADMAP.md` | Roadmap |
| `md/14_DESIGN_DECISIONS.md` | Design decisions |
| `md/15_CANONICAL_PIPELINE.md` | 10-phase canonical pipeline |

---

## Development

### Running tests

```bash
# All tests
python -m pytest tests/

# Specific test file
python -m pytest tests/test_cli.py

# With coverage
python -m pytest tests/ --cov=fiae --cov-report=html
```

### Code style

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/fiae/
```

### Continuous integration

Every push and pull request runs [.github/workflows/ci.yml](.github/workflows/ci.yml):

| Job | What it does |
|---|---|
| **lint** | `ruff check src/fiae tests` — the enforced lint baseline (0 findings) |
| **test** | Full suite **plus an end-to-end example smoke**, on a 3 OS × 3 Python matrix (Ubuntu/Windows/macOS × 3.10/3.11/3.12), with JUnit reports and per-matrix summaries |
| **coverage** | Suite run under `pytest-cov`; `coverage.xml` published as a run artifact |

Tests run against the source tree (the core is stdlib-only; no Cython build in CI). Packaging is exercised by the [release workflow](.github/workflows/release.yml) on tags: every wheel is **verified before publishing** — installed into a clean environment, import-path-checked (`site-packages`), full suite + example smoke run against the installed package, and only then pushed to PyPI. The CI commands also run locally:

```bash
ruff check src/fiae tests
python -m pytest tests/
python examples/churn/run_api.py
```

### End-to-end example

`examples/churn/run_api.py` runs the canonical 10-phase pipeline on a
generated synthetic churn dataset (no downloads, no credentials, CPU-only):

```bash
python examples/churn/run_api.py
```

---

## Testing

| Category | Files | Count | Coverage |
|---|---|---|---|
| **Intake** | `test_intake_*.py` | ~50 | Profiling, CSV, JSON, Parquet |
| **Features** | `test_features.py`, `test_operators.py` | ~100 | All 95 operators, contracts |
| **Funnel** | `test_funnel.py`, `test_gates.py` | ~40 | F0–F6 gates, policies |
| **Models** | `test_models.py`, `test_hpo.py` | ~60 | All 12 families, HPO algorithms |
| **Pipeline** | `test_learn.py`, `test_m*.py` | ~80 | End-to-end pipeline, fitted pipeline, all milestones |
| **Real Data** | `test_real_data_full_pipeline.py` | 17 | 20-type dataset, 100K rows, full pipeline |
| **Contracts & Events** | `test_contracts.py`, `test_events.py` | ~30 | Typed contracts, event bus, deterministic IDs |
| **CLI** | `test_cli.py` | ~20 | All CLI commands, JSON output |
| **Security** | `test_security*.py` | 15 | Resource limits, input validation, audit logging |
| **Misc** | `test_m*.py` | 200+ | Milestone integration tests (M3–M13) |
| **Total** | **35 files** | **852** | |

---

## Project Structure

```
fiae/
├── assets/                     # Brand assets
│   └── fiae-logo.png           # FIAE logo
├── md/                         # 16 design documents (00–15) + navigation
├── src/fiae/                   # Source package
│   ├── __init__.py             # Package init
│   ├── __main__.py             # CLI entry point
│   ├── cli.py                  # CLI commands
│   ├── cli_advanced.py         # Advanced CLI commands
│   ├── cli_colors.py           # CLI color system
│   ├── cli_dashboard.py        # Dashboard commands
│   ├── cli_pipeline.py         # Pipeline commands
│   ├── codegen/                # Code generation
│   ├── contracts.py            # Typed contracts
│   ├── errors.py               # Error handling
│   ├── evaluate.py             # Evaluation
│   ├── events.py               # Event bus
│   ├── experience/             # Experience memory
│   ├── features/               # Feature operators
│   ├── funnel.py               # Multi-stage funnel
│   ├── identity.py             # CLI visual identity
│   ├── ids.py                  # Deterministic IDs
│   ├── intake/                 # Data intake & adapters
│   │   ├── adapter_api.py      # API/URL adapter
│   │   ├── adapter_base.py     # Base adapter
│   │   ├── adapter_cloud.py    # Cloud storage adapter
│   │   ├── adapter_dataframe.py # DataFrame adapter
│   │   ├── adapter_factory.py  # Auto-detection factory
│   │   ├── adapter_file.py     # File adapters
│   │   ├── adapter_sql.py      # SQL adapter
│   │   ├── csv_source.py       # CSV adapter
│   │   ├── profiler.py         # Data profiling
│   │   └── stats.py            # Statistics
│   ├── learn.py                # Feature intelligence pipeline
│   ├── model_registry.py       # Model registry
│   ├── orchestration/          # Orchestration
│   ├── pipeline/               # Pipeline IR
│   ├── problem/                # Problem definition
│   ├── research/               # Research
│   ├── runs.py                 # Run lifecycle
│   ├── search/                 # Feature search
│   ├── security/               # Security
│   └── testing/                # Testing
├── tests/                      # Test suite (693 tests)
├── pyproject.toml              # Build config & dependencies
├── conftest.py                 # Test infrastructure
├── README.md                   # This file
└── .gitignore
```

---

## Roadmap

### Completed

- [x] Data intake with streaming profiling (doc 02)
- [x] 6-class leakage taxonomy (L0–L5) (doc 03)
- [x] 95 typed feature operators across 13 families (doc 05)
- [x] Multi-stage funnel with policy-driven gates (doc 04)
- [x] Experience store with R0–R3 retrieval (doc 06)
- [x] HPO algorithms: random, TPE, successive halving, Hyperband (doc 07)
- [x] Real sklearn model training with 12 model families (doc 07)
- [x] Thread pool parallel executor with backpressure (doc 08)
- [x] Pipeline IR with 10 verification gates (doc 09)
- [x] Standalone sklearn project export (doc 09)
- [x] Security sandbox with resource enforcement (doc 11)
- [x] Audit logging with JSONL persistence (doc 11)
- [x] Fitted-state pipeline with fit/transform lifecycle (doc 04)
- [x] 10-phase canonical pipeline end-to-end (doc 15)
- [x] URL/cloud data source support with progress indicators
- [x] Warehouse adapters: BigQuery, Snowflake, Redshift
- [x] Thread-safe SQL/warehouse adapters (per-thread connections)
- [x] Bounded-concurrency REST server: worker pool, 429 backpressure,
      per-client rate limiting, load-tested with 200 simultaneous clients
- [x] `fiae connect` universal source connector
- [x] 860 passing tests
- [x] Real-data validation on 20-type 100K-row dataset

### Planned

- [ ] Multi-table / relational feature engineering
- [ ] Time-series specialized operators
- [ ] Rust/C++ acceleration for hot paths
- [x] Integration with MLflow, Weights & Biases (`--track mlflow|wandb|local`)
- [x] Web-based dashboard for run monitoring (`fiae serve`)
- [x] REST API for remote pipeline execution (`fiae serve`, stdlib-only)
- [ ] Auto-tuning of funnel policies from experience
- [ ] Distributed execution across multiple machines

---

## Contributing

FIAE follows strict design principles. Before contributing:

1. **Read the design documents** — changes must be normatively grounded
2. **Zero third-party deps in core** — heavy capabilities go in optional tiers
3. **Every operator must have typed contracts** — input types, output type, leakage class, fit scope
4. **All transforms must be deterministic** — same input → same output, always
5. **Tests are mandatory** — no operator ships without passing its contract tests

See `md/00_MASTER_BLUEPRINT.md` for the 15 hard principles.

Full setup instructions, the pre-PR command loop, and the PR policy
(design-doc reference + risk label + test evidence) live in
[CONTRIBUTING.md](CONTRIBUTING.md). Bug reports and feature requests have
templates; security issues go through [SECURITY.md](SECURITY.md) privately.

---

## Citation

If you use FIAE in your research, please cite:

```bibtex
@software{fiae2026,
  title  = {FIAE: Feature Intelligence \& Architecture Engine},
  year   = {2026},
  url    = {https://github.com/faizansk25/FIAE},
  note   = {Automated feature engineering with safety-first design}
}
```

---

## License

FIAE uses a **source-available** license. The source code is available for viewing,
studying, and personal use, but redistribution and derivative works are restricted.
See [LICENSE](LICENSE) for details.

## Support FIAE

FIAE is free to use. If it helps you, please consider supporting its development:

[![GitHub Sponsors](https://img.shields.io/badge/Sponsor-GitHub%20Sponsors-EA4AAA.svg?style=for-the-badge&logo=githubsponsors)](https://github.com/sponsors/fiae)
[![Ko-fi](https://img.shields.io/badge/Donate-Ko--fi-FF5E5B.svg?style=for-the-badge&logo=ko-fi)](https://ko-fi.com/fiae)

See [DONATIONS.md](DONATIONS.md) for details on sponsorship tiers and how funds are used.

---

<div align="center">

**Built with safety, speed, and trust.**

*FIAE — Feature Intelligence & Architecture Engine*

</div>