"""M40.5 regression guards for two bugs that a passing suite had hidden.

1. ``writeback.write_case`` constructed every ``Case*`` dataclass with keyword
   arguments that do not exist on those dataclasses, called a non-existent
   ``store.write``, and omitted two required ``CaseRecord`` identity fields.
   Every call raised ``TypeError``. The only test covering it was written as::

       try:
           case_id = write_case(...)
           assert case_id.startswith("case_")
       except Exception:
           pass  # store API may differ

   so the failure could never surface. That blanket ``except`` has been
   removed; these tests assert the write-back path really works.

2. ``cli_pipeline`` called ``C.warning_header()`` on the colour module, which
   has no such attribute. The error-reporting path for parity failures
   therefore raised ``AttributeError`` and hid the error it was meant to
   report.
"""

import dataclasses
import os
import tempfile

import pytest

from fiae import cli_colors as C
from fiae.contracts import (
    ColumnProfile, DatasetProfile, Direction, FeatureNode, MetricValue,
    SemanticType,
)
from fiae.experience.case import (
    CaseAction, CaseContext, CaseCost, CaseDiagnosis, CaseRecord, CaseResult,
)
from fiae.experience.store import ExperienceStore, StoreConfig
from fiae.experience.writeback import write_case


def _profile() -> DatasetProfile:
    return DatasetProfile(
        dataset_fingerprint="fp_test",
        rows_observed=100,
        columns=[ColumnProfile("x", "float", SemanticType.CONTINUOUS_NUMERIC, 0.0, 10, 0.1)],
    )


@pytest.fixture()
def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        st = ExperienceStore(StoreConfig(db_path=os.path.join(tmpdir, "exp.db")))
        try:
            yield st
        finally:
            st.close()


class TestWriteBackWorks:
    def test_write_case_returns_a_case_id(self, store):
        features = [FeatureNode(feature_id="f_test", operator="sqrt", inputs=["raw:x"])]
        metrics = [MetricValue("rmse", 0.5, Direction.MINIMIZE, "cv")]

        case_id = write_case(store, _profile(), features, metrics, [], "regression")

        assert case_id.startswith("case_")

    def test_case_is_actually_persisted(self, store):
        """The write must reach SQLite, not merely return an id."""
        features = [FeatureNode(feature_id="f_test", operator="sqrt", inputs=["raw:x"])]
        metrics = [MetricValue("roc_auc", 0.9, Direction.MAXIMIZE, "cv")]

        write_case(store, _profile(), features, metrics, [], "binary_classification")

        rows = store._conn.execute("SELECT COUNT(*) FROM cases").fetchone()
        assert rows[0] == 1

    def test_accepts_an_unknown_task_label(self, store):
        """Task strings outside the enum must degrade to AUTO, not crash."""
        write_case(store, _profile(), [], [], [], "not-a-real-task")

    def test_handles_empty_metrics(self, store):
        write_case(store, _profile(), [], [], [], "regression")

    def test_resources_feed_the_cost_record(self, store):
        from fiae.contracts import ResourceMeasurement

        res = ResourceMeasurement(wall_time_s=1.5, cpu_time_s=0.75, peak_rss_bytes=2 * 1024 * 1024)
        write_case(store, _profile(), [], [], [res], "regression")


class TestWriteCaseUsesRealDataclassFields:
    """Pin the kwargs to fields that actually exist.

    The original bug was a wholesale mismatch; these assert against
    ``dataclasses.fields`` so a future rename cannot pass silently.
    """

    def test_every_case_component_is_a_dataclass_with_fields(self):
        for cls in (CaseContext, CaseAction, CaseResult, CaseCost, CaseDiagnosis, CaseRecord):
            assert dataclasses.is_dataclass(cls)
            assert [f.name for f in dataclasses.fields(cls)]

    def test_record_identity_fields_are_required(self):
        required = {
            f.name for f in dataclasses.fields(CaseRecord)
            if f.default is dataclasses.MISSING
        }
        assert {"case_id", "dataset_fingerprint", "schema_fingerprint"} <= required


class TestCliColourReferencesExist:
    """`C.warning_header()` did not exist; the report path raised instead."""

    def test_warning_header_does_not_exist_so_the_bug_cannot_recur_silently(self):
        assert not hasattr(C, "warning_header")

    def test_error_header_used_by_the_parity_path_exists_and_returns_str(self):
        assert callable(C.error_header)
        assert isinstance(C.error_header(force=False), str)

    def test_cli_pipeline_references_only_real_colour_attributes(self):
        """Static check: every ``C.<name>`` used in cli_pipeline must exist."""
        import ast
        from pathlib import Path

        src = (Path(__file__).resolve().parents[1]
               / "src" / "fiae" / "cli_pipeline.py").read_text("utf-8")
        used = {
            node.attr for node in ast.walk(ast.parse(src))
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "C"
        }
        assert used, "expected cli_pipeline to use the colour module"
        missing = sorted(n for n in used if not hasattr(C, n))
        assert not missing, f"cli_pipeline references non-existent colour attrs: {missing}"

