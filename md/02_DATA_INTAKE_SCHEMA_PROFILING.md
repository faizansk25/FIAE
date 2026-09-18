# Data Intake, Schema, Profiling, and Fingerprinting

> **Purpose:** Specify how FIAE understands data before feature/model search.

**Status:** Normative specification for one continuously evolving FIAE codebase. These documents describe implementation gates, not user-facing versions.

## Research basis

- [DuckDB CSV detection](https://duckdb.org/docs/current/data/csv/auto_detection)
- [Polars lazy](https://docs.pola.rs/user-guide/lazy/using/)
- [Polars streaming](https://docs.pola.rs/user-guide/concepts/streaming/)
- [DataRobot data quality](https://docs.datarobot.com/en/docs/reference/data-ref/data-quality-ref.html)
- [AutoGluon feature engineering](https://auto.gluon.ai/stable/tutorials/tabular/tabular-feature-engineering.html)


## Source adapter contract

```python
class DataSourceAdapter(Protocol):
    def source_id(self) -> str: ...
    def schema_hint(self) -> SchemaHint | None: ...
    def estimate_rows(self) -> int | None: ...
    def estimate_bytes(self) -> int | None: ...
    def sample(self, plan: SamplePlan) -> Iterable[RowBatch]: ...
    def scan(self, projection=None, predicate=None) -> Iterable[RowBatch]: ...
    def supports_seek_sampling(self) -> bool: ...
    def supports_pushdown(self) -> bool: ...
    def fingerprint_material(self) -> bytes: ...
```

## CSV parsing algorithm

1. Read bounded initial bytes.
2. Detect BOM/encoding; prefer UTF-8.
3. Evaluate delimiter candidates comma, tab, semicolon, pipe, plus override.
4. Respect quoted delimiters and quoted newlines.
5. Select dialect with stable row width and plausible column count.
6. Infer header only with adequate confidence; user declaration wins.
7. If seekable and parser supports safe record localization, sample multiple file regions.
8. If compressed/non-seekable, record head-biased limitation.
9. Parse with strict malformed-row accounting.
10. Abort or lower confidence if malformed fraction exceeds policy.

DuckDB is a reference because its current CSV sniffer uses a default 20,480-row sample and, for regular files, can sample different locations rather than only the beginning.

## Sampling plan

A deterministic representative fast sample can combine:

```text
head block
+ middle block(s)
+ tail block
+ deterministic pseudo-random blocks
```

## Physical type inference

Per column maintain compatible candidates:
- null;
- boolean;
- integer;
- float/decimal;
- date;
- timestamp;
- string.

Each observed non-null value eliminates incompatible candidates. Ambiguous mixed columns remain string/mixed rather than unsafe coercion.

## Semantic type inference

Possible roles:
- continuous numeric;
- count;
- ordinal;
- low-cardinality categorical;
- high-cardinality categorical;
- boolean;
- date/datetime;
- free text;
- identifier;
- entity key;
- geography-like code;
- percentage-like;
- currency-like;
- URL/email-like sensitive string;
- mixed/unknown.

### Identifier evidence
- distinct ratio near one;
- UUID/GUID/fixed token pattern;
- name hints such as `_id`, `uuid`, `key`, `record`;
- low repeat count;
- lack of magnitude semantics.

Name hints are never hard evidence alone.

### Text vs categorical evidence
- distinct ratio;
- length distribution;
- token count;
- whitespace frequency;
- vocabulary repetition;
- cardinality.

## Streaming numeric statistics

Welford:

```text
n = n + 1
delta = x - mean
mean = mean + delta/n
delta2 = x - mean
M2 = M2 + delta*delta2
variance = M2/(n-1)
```

Also track min/max, finite/NaN/inf, zero/negative fractions, optional quantiles, skew/heavy-tail/outlier proxies.

## Missingness

Separate true null, empty string, whitespace, configured missing tokens, and suspicious numeric sentinels. Suspicious sentinels are warnings until confirmed.

## Cardinality

Small columns: exact set up to cap.

Large columns: bounded KMV/HLL-style sketch.

Store estimate, exact/approx flag, sketch configuration, and uncertainty metadata.

## Data-quality checks

- constant/near-constant;
- duplicate columns;
- duplicate row estimate;
- excessive missingness;
- disguised missing values;
- excess zeros;
- malformed multi-category values;
- likely identifiers;
- target leakage suspicion;
- train/validation shift;
- temporal irregularities if time series.

DataRobot's current docs include target leakage among baseline data-quality checks and add time-series checks such as pre-derived lags and irregular time steps.

## Dataset meta-features

Examples:
- log rows / log columns;
- row-to-column ratio;
- source bytes;
- semantic-type fractions;
- missingness mean/max;
- cardinality mean/max;
- target entropy/imbalance/variance;
- sparse/dense estimate;
- repeated-entity ratio;
- temporal flags;
- text fraction;
- dimensionality risk;
- duplicate estimate.

## Fingerprint

Combine:
1. stable source version/content material;
2. canonical schema;
3. parser config;
4. bounded content hashes or native version ID;
5. stable source metadata.

Path alone is not identity.

## Profiling modes

### FAST
Strict row/time/byte bounds; approximation allowed.

### STANDARD
Larger representative scan.

### EXACT
Full scan where needed and affordable.

## Aspect routing examples

```text
datetime -> temporal/calendar
text -> text stats/hash/TFIDF
high-cardinality -> frequency/hash/cross-fit target encoding
imbalance -> PR metrics/weights/thresholding
small-n high-p -> conservative interaction depth
large-n -> subsampled probes/streaming
identifier -> direct exclusion, entity/group reasoning
```

## Acceptance failures

Stop or clarify if:
- zero rows;
- target missing/constant;
- duplicate columns unresolved;
- parser confidence too low;
- source changes mid-run;
- malformed rows exceed tolerance.
