# Is FIAE for everyone? An honest assessment

Short answer: **the engine is generic — tabular feature engineering is tabular feature engineering**. The guarantees (leakage taxonomy, funnel evidence, verified export) apply to every dataset it ingests. But *generic* is not the same as *finished for every vertical*. Here is exactly where FIAE stands today.

## What is genuinely universal today

| Capability | Works for |
|---|---|
| CSV / TSV / JSON / Parquet files, HTTP(S) URLs, S3, GCS, PostgreSQL/MySQL/SQLite, warehouses, REST APIs, in-memory DataFrames | Any organization with tabular data — which is effectively everyone |
| 95 typed operators (numeric, categorical, datetime, text, temporal, group, interactions, clustering) | Any tabular domain: churn, fraud, credit risk, demand forecasting, predictive maintenance, HR analytics, healthcare records, marketing attribution |
| 6-class leakage taxonomy + funnel evidence (F0–F6) | Domain-independent by construction — the gates reason about data structure, not subject matter |
| Verified standalone export (11 gates, row-level parity) | Any deployment target that runs Python: batch jobs, microservices, Airflow tasks, SageMaker/Vertex containers |
| Deterministic runs (seeded CV, content-hash IDs, md5 bucketing) | Regulated environments that require reproducibility (audit trail included) |
| REST API + dashboard (`fiae serve`), Python API, CLI | Teams of any size — from a single analyst to an ML platform group |

## What works, with honest caveats

- **Deep learning / GPU training.** FIAE engineers and verifies features; it does not train neural networks. Its output (a verified feature pipeline) is an excellent *feeder* into PyTorch/TensorFlow, but model training itself is sklearn-family today.
- **Streaming data.** Intake is batch-oriented (with sampling and row budgets). Kafka/Flink-style continuous ingestion is on the roadmap, not in the box.
- **Computer vision, audio, graphs.** Out of scope by design. FIAE is a *feature* engine for structured data, not a representation-learning engine.
- **LLM embeddings as operators.** Text operators today are statistical (lengths, hash bucketing, character/word stats). Plugging external embedding models is a natural extension point, not a current feature.
- **Petabyte scale.** The profiler streams and samples, and the core has zero heavy dependencies, but FIAE is architected for single-node workloads (GBs), not distributed clusters. Export the pipeline and run it on your cluster instead.

## Who should adopt it now

1. **ML teams in regulated domains** (finance, healthcare, insurance) — the per-feature evidence trail (every gate verdict recorded, immutable decision IDs) is the product.
2. **Platform teams** tired of hand-written, silently-drifting feature code — verified export means the fitted pipeline and the deployed pipeline provably match.
3. **Consultancies and agencies** shipping many client models on tabular data — one engine, any source, deterministic deliverables.
4. **Researchers/educators** who want leakage-safe, reproducible baselines without writing feature code.

## Who should wait (for now)

- Teams whose primary need is distributed training over multi-TB data.
- Teams needing image/audio/graph pipelines.
- Teams requiring a mature plugin marketplace — FIAE's operator registry is code-extensible today, but the ecosystem is young.

## The one-sentence answer

FIAE is **domain-agnostic and deployment-agnostic for tabular ML** — "generic for all" is true wherever the job is *turning columns into trustworthy features*; it is not (and does not pretend to be) a deep-learning, streaming, or distributed-training framework.

*Claims in this document reflect the state of the code at v0.0.2: 852 passing tests, 95 operators / 13 families, 11 verification gates, 25+ source connectors.*
