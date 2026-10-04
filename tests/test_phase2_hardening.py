"""Phase-2 hardening tests: mission gaps left after the M1-M8 sweep.

- export/runtime equivalence (generated code == fitted pipeline, end-to-end)
- schema preservation (row count/dtypes contract across transformations)
- Parquet fuzzing (truncated/garbage files rejected, valid files parse)
- SSRF guard on the API adapter (loopback/private/metadata targets rejected)
- JSON/NDJSON over-nesting guard (no raw RecursionError)
- artifact-hash lineage (SHA-256 recorded in the run audit trail)
"""

import json
import os

import pytest

from fiae.codegen.pipeline_ir import (
    build_ir_from_proposals,
    generate_python_code,
)
from fiae.errors import FIAEError
from fiae.experiment.tracking import ExperimentTracker, TrackingConfig
from fiae.fitted_pipeline import FittedPipeline
from fiae.intake.adapter_api import ApiAdapter
from fiae.intake.adapter_file import (
    MAX_JSON_DEPTH,
    JsonAdapter,
    NdjsonAdapter,
)
from fiae.search.triggers import FeatureProposal


# ---------------------------------------------------------------------------
# Export / runtime equivalence
# ---------------------------------------------------------------------------


class TestExportRuntimeEquivalence:
    def _roundtrip(self, data, props):
        runtime = FittedPipeline().fit(props, data)
        ir = build_ir_from_proposals(props)
        code = generate_python_code(ir)
        compile(code, "<export>", "exec")  # must be valid Python
        namespace = {}
        exec(compile(code, "<export>", "exec"), namespace)
        exported = namespace["apply_pipeline"](data)
        return runtime, ir, exported

    def test_generated_code_matches_fitted_runtime(self):
        data = {"x": [1.0, 4.0, 9.0, 16.0], "y": [2.0, 3.0, 5.0, 7.0]}
        props = [
            FeatureProposal(op="sqrt", inputs=["raw:x"]),
            FeatureProposal(op="log1p", inputs=["raw:y"]),
            FeatureProposal(op="product", inputs=["raw:x", "raw:y"]),
        ]
        runtime, ir, exported = self._roundtrip(data, props)
        for node, prop in zip(ir.topological_order(), props):
            rt = runtime[prop.op + "(" + "_".join(
                i[4:] for i in prop.inputs) + ")"]
            ex = exported[node.node_id]
            assert len(rt) == len(ex)
            assert all(abs(a - b) < 1e-9 for a, b in zip(rt, ex)), (
                f"export/runtime mismatch for {node.node_id} ({prop.op})"
            )

    def test_missing_rows_preserved_in_export(self):
        """Missing inputs surface as None in the exported code (never as
        silent zeros -- that would corrupt downstream metrics)."""
        data = {"x": [1.0, None, 4.0], "y": [1.0, 2.0, 3.0]}
        props = [FeatureProposal(op="sqrt", inputs=["raw:x"])]
        _, _, exported = self._roundtrip(data, props)
        assert exported["n_0"][1] is None
        assert exported["n_0"][0] == 1.0

    def test_exported_code_is_self_contained(self):
        """The export must run without importing FIAE: generate, execute in
        an isolated namespace with fiae removed from sys.modules."""
        import builtins

        data = {"x": [1.0, 2.0]}
        props = [FeatureProposal(op="sqrt", inputs=["raw:x"])]
        _, ir, _ = self._roundtrip(data, props)
        code = generate_python_code(ir)
        assert "import fia" not in code  # no FIAE imports in the source

        namespace = {}
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "fiae" or name.startswith("fiae."):
                raise ImportError("exported code must be self-contained")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = guarded_import
        try:
            exec(compile(code, "<export>", "exec"), namespace)
            out = namespace["apply_pipeline"]({"x": [4.0]})
        finally:
            builtins.__import__ = real_import
        assert out["n_0"] == [2.0]


class TestSchemaPreservation:
    def test_row_count_preserved_through_pipeline(self):
        """Every transform preserves the row count exactly (no reordering,
        no drop, no duplication) even with nulls."""
        data = {
            "x": [1.0, None, 3.0, -4.0, 5.0, None, 7.0],
            "c": ["a", "b", "a", "b", "a", "b", "a"],
        }
        props = [
            FeatureProposal(op="sqrt", inputs=["raw:x"]),
            FeatureProposal(op="standardize", inputs=["raw:x"]),
            FeatureProposal(op="one_hot", inputs=["raw:c"]),
        ]
        out = FittedPipeline().fit(props, data)
        for name, values in out.items():
            assert len(values) == 7, f"{name} changed row count"

    def test_column_set_stable_between_fit_and_transform(self):
        data = {"x": [1.0, 2.0, 3.0, 4.0], "c": ["a", "b", "a", "b"]}
        props = [
            FeatureProposal(op="standardize", inputs=["raw:x"]),
            FeatureProposal(op="one_hot", inputs=["raw:c"]),
        ]
        pipe = FittedPipeline()
        fit_out = pipe.fit(props, data)
        transform_out = pipe.transform(props, data)
        assert set(fit_out) == set(transform_out)


# ---------------------------------------------------------------------------
# Parquet fuzzing
# ---------------------------------------------------------------------------


class TestParquetFuzzing:
    """Only the happy path needs pyarrow.

    This module used to end with a module-level
    ``pytest.importorskip("pyarrow")``. pyarrow is not in the dev or tier1
    extras, so on CI that line deleted **all 24 tests in this file** at
    collection time - no error, no failure, just 21 fewer tests than the
    badge claimed, none of them reporting anything. The guard is now
    per-test, where a missing optional dependency is visible as one honest
    skip. `tests/test_collection_integrity.py` keeps that from recurring.
    """

    def test_valid_parquet_reads(self, tmp_path):
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")

        p = tmp_path / "ok.parquet"
        pq.write_table(pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]}), p)
        from fiae.intake.adapter_file import ParquetAdapter

        batches = list(ParquetAdapter(str(p)).scan())
        assert sum(len(b.columns["a"]) for b in batches) == 3

    def test_truncated_parquet_rejected(self, tmp_path):

        p = tmp_path / "trunc.parquet"
        p.write_bytes(b"PAR1" + b"\x00" * 100)
        from fiae.intake.adapter_file import ParquetAdapter

        with pytest.raises(Exception):
            list(ParquetAdapter(str(p)).scan())

    def test_garbage_parquet_rejected(self, tmp_path):
        p = tmp_path / "garbage.parquet"
        p.write_bytes(os.urandom(2048))
        from fiae.intake.adapter_file import ParquetAdapter

        with pytest.raises(Exception):
            list(ParquetAdapter(str(p)).scan())

    def test_parquet_never_accepted_from_factory_when_missing(self, tmp_path):
        from fiae.errors import FIAEError
        from fiae.intake.adapter_factory import auto_adapter

        with pytest.raises(FIAEError):
            auto_adapter(str(tmp_path / "nonexistent.parquet"))


# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------


class TestSSRFGuard:
    @pytest.mark.parametrize("url", [
        "http://127.0.0.1:8080/data",
        "http://10.0.0.5/internal",
        "http://192.168.1.1/x",
        "http://172.16.0.9/x",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://localhost/admin",
        "http://[::1]/x",
        "http://metadata.google.internal/computeMetadata/v1/",
        "file:///etc/passwd",
    ])
    def test_private_and_metadata_targets_rejected(self, url):
        with pytest.raises(FIAEError):
            ApiAdapter(url)

    def test_public_url_accepted(self):
        a = ApiAdapter("https://api.example.com/users")
        assert a.source_id() == "api|https://api.example.com/users"


# ---------------------------------------------------------------------------
# JSON / NDJSON over-nesting guard
# ---------------------------------------------------------------------------


class TestJsonDepthGuard:
    def test_over_nested_json_rejected_with_clear_error(self, tmp_path):
        p = tmp_path / "deep.json"
        p.write_text('{"a": ' + "[" * 2000 + "]" * 2000 + "}")
        with pytest.raises(ValueError, match="nesting depth"):
            list(JsonAdapter(str(p)).scan())

    def test_brackets_inside_a_string_are_not_nesting(self, tmp_path):
        """The depth scan is lexical, so it has to respect string state.

        A 500-bracket *string value* is one level deep, not 500. Counting
        the characters would reject perfectly ordinary data.
        """
        p = tmp_path / "brackets.json"
        p.write_text('{"a": "' + "[" * 500 + '"}')
        batches = list(JsonAdapter(str(p)).scan())
        assert sum(len(b.columns["a"]) for b in batches) == 1

    def test_payload_exactly_at_the_ceiling_is_accepted(self, tmp_path):
        """The limit is a ceiling, not an off-by-one trap.

        Depth MAX_JSON_DEPTH must parse; MAX_JSON_DEPTH + 1 must not.
        Without this pair the constant could drift by one in either
        direction and nothing would notice.
        """
        # The outer object counts as one level, so MAX_JSON_DEPTH - 1 arrays
        # land exactly on the ceiling.
        inner = "1"
        for _ in range(MAX_JSON_DEPTH - 1):
            inner = "[" + inner + "]"

        at_limit = tmp_path / "at.json"
        at_limit.write_text('{"a": ' + inner + "}")
        assert sum(len(b.columns["a"]) for b in JsonAdapter(str(at_limit)).scan()) == 1

        over = tmp_path / "over.json"
        over.write_text('{"a": ' + "[" + inner + "]}")
        with pytest.raises(ValueError, match="nesting depth"):
            list(JsonAdapter(str(over)).scan())

    def test_ndjson_hostile_lines_skipped_not_fatal(self, tmp_path):
        p = tmp_path / "mix.ndjson"
        p.write_text(
            '{"a":"1"}\n{"a": ' + "[" * 2000 + "]" * 2000 + "}\n" + '{"a":"3"}\n'
        )
        batches = list(NdjsonAdapter(str(p)).scan())
        n = sum(len(b.columns.get("a", [])) for b in batches)
        assert n == 2  # hostile over-nested line skipped, good rows survive

    def test_normal_json_still_reads(self, tmp_path):
        p = tmp_path / "ok.json"
        p.write_text(json.dumps([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]))
        batches = list(JsonAdapter(str(p)).scan())
        assert sum(len(b.columns["a"]) for b in batches) == 2


# ---------------------------------------------------------------------------
# Artifact hash lineage
# ---------------------------------------------------------------------------


class TestArtifactHashLineage:
    def test_artifact_hash_recorded_in_run_log(self, tmp_path):
        config = TrackingConfig(backend="local", output_dir=str(tmp_path))
        tracker = ExperimentTracker(config)
        tracker.start({"seed": 1})

        artifact = tmp_path / "lineage.json"
        artifact.write_text('{"portfolio": []}')
        tracker.log_artifact(str(artifact))
        tracker.finish()

        # audit trail lives at <output_dir>/<run_id>/audit.jsonl
        run_id = tracker.run_id
        audit_log = tmp_path / run_id / "audit.jsonl"
        assert audit_log.exists(), "local audit trail missing"
        entries = [
            json.loads(line)
            for line in audit_log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        artifact_entries = [
            e for e in entries if e.get("action") == "artifact"
        ]
        assert artifact_entries, "artifact event missing from run log"
        recorded = artifact_entries[-1].get("details", {}).get("sha256")
        assert recorded, "artifact hash not recorded"
        # hash matches the file content
        import hashlib

        expected = hashlib.sha256(artifact.read_bytes()).hexdigest()
        assert recorded == expected

    def test_log_artifact_ignores_missing_file(self, tmp_path):
        config = TrackingConfig(backend="local", output_dir=str(tmp_path))
        tracker = ExperimentTracker(config)
        tracker.start({"seed": 1})
        tracker.log_artifact(str(tmp_path / "does_not_exist.json"))  # no raise
        tracker.finish()
