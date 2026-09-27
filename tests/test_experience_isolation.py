"""Experience store isolation, versioning, and reliability tests (M5).

Cross-dataset memory must not:
- leak labels into stored or retrieved records
- contaminate evaluation across task types
- use incompatible schema assumptions (schema fingerprint separation)
- create irreproducible behavior (deterministic identity, no silent
  overwrites across engine versions)

Plus: record versioning (engine_commit participates in identity),
forward/backward compatibility of stored JSON, and corrupt-row tolerance.
"""

import json
import sqlite3

import pytest

from fiae.contracts import Task, TrialStatus
from fiae.experience.case import (
    CaseAction,
    CaseContext,
    CaseDiagnosis,
    CaseRecord,
    CaseResult,
)
from fiae.experience.retrieval import retrieve_priors
from fiae.experience.store import ExperienceStore, StoreConfig


@pytest.fixture()
def store(tmp_path):
    return ExperienceStore(StoreConfig(db_path=str(tmp_path / "exp.db")))


def _case(case_id, fp="fp", schema="sc", engine="commit_A", task=Task.BINARY,
          rows=100, model="logistic", metric=0.85, seed=None,
          status=TrialStatus.COMPLETED):
    return CaseRecord(
        case_id=case_id,
        dataset_fingerprint=fp,
        schema_fingerprint=schema,
        engine_commit=engine,
        context=CaseContext(task=task, dataset_meta_features={"n_rows": rows}),
        action=CaseAction(model_family=model),
        result=CaseResult(primary_metric_value=metric),
        seed=seed,
        diagnosis=CaseDiagnosis(status=status),
    )


class TestCrossDatasetIsolation:
    def test_records_stay_scoped_to_their_dataset(self, store):
        store.write_case(_case("a1", fp="fp_dataset_A"))
        store.write_case(_case("b1", fp="fp_dataset_B"))
        assert len(store.find_by_dataset("fp_dataset_A")) == 1
        assert len(store.find_by_dataset("fp_dataset_B")) == 1
        assert store.find_by_dataset("fp_unknown") == []

    def test_no_label_leakage_into_priors(self, store):
        """The prior must aggregate success statistics only -- never expose
        raw per-row target data from past datasets."""
        store.write_case(_case("a1", metric=0.9))
        prior = retrieve_priors(store, "binary_classification", 100, 5)
        dumped = json.dumps(prior.__dict__, default=str)
        # hyperparameter priors absent -> nothing row-level stored
        assert prior.hyperparameter_priors == {} or all(
            isinstance(v, dict) for v in prior.hyperparameter_priors.values()
        )
        assert "target" not in dumped.lower() or "target_semantics" in dumped

    def test_task_isolation_r0_hard_filter(self, store):
        """A regression failure must not contaminate binary priors."""
        store.write_case(_case(
            "reg1", task=Task.REGRESSION, model="ridge", metric=0.05,
            status=TrialStatus.FAILED,
        ))
        prior = retrieve_priors(store, "binary_classification", 100, 5)
        assert "ridge" not in prior.model_family_priors
        assert prior.source_count == 0

    def test_failed_case_not_scored_as_success(self, store):
        store.write_case(_case(
            "fail1", model="logistic", metric=0.3, status=TrialStatus.FAILED,
        ))
        prior = retrieve_priors(store, "binary_classification", 100, 5)
        mp = prior.model_family_priors["logistic"]
        assert mp.successes == 0 and mp.attempts == 1

    def test_prior_is_deterministic_for_same_store_state(self, store):
        for i in range(5):
            store.write_case(_case(f"c{i}", metric=0.7 + i * 0.01))
        p1 = retrieve_priors(store, "binary_classification", 100, 5)
        p2 = retrieve_priors(store, "binary_classification", 100, 5)
        assert p1.model_family_priors == p2.model_family_priors
        assert p1.confidence == p2.confidence


class TestSchemaCompatibility:
    def test_same_dataset_different_schema_is_separate_knowledge(self, store):
        """Cases recorded under schema v1 must not be presented as evidence
        for a dataset that has since changed shape (schema fingerprint)."""
        store.write_case(_case("v1", schema="SCHEMA_V1", metric=0.8))
        store.write_case(_case("v2", schema="SCHEMA_V2", metric=0.8))
        assert store.get_by_identity(_case("v1", schema="SCHEMA_V1").identity_hash())
        assert store.get_by_identity(_case("v2", schema="SCHEMA_V2").identity_hash())

    def test_identity_includes_engine_commit(self):
        """Different engine versions produce different identities -- a newer
        engine must not silently overwrite an older record (versioning)."""
        r_old = _case("x", engine="commit_AAA", metric=0.7)
        r_new = _case("x", engine="commit_BBB", metric=0.9)
        assert r_old.identity_hash() != r_new.identity_hash()

    def test_identity_includes_schema_fingerprint(self):
        assert _case("x", schema="S1").identity_hash() != _case("x", schema="S2").identity_hash()

    def test_write_does_not_overwrite_across_engine_versions(self, store):
        store.write_case(_case("x", engine="commit_AAA", metric=0.7))
        store.write_case(_case("x", engine="commit_BBB", metric=0.9))
        # both versions remain retrievable by identity
        id_old = _case("x", engine="commit_AAA").identity_hash()
        id_new = _case("x", engine="commit_BBB").identity_hash()
        old = store.get_by_identity(id_old)
        new = store.get_by_identity(id_new)
        assert old is not None and old.result.primary_metric_value == 0.7
        assert new is not None and new.result.primary_metric_value == 0.9

    def test_same_identity_dedupes(self, store):
        """Identical (dataset, split, graph, model, params, seed, version)
        records collapse to one row -- no double counting in priors."""
        store.write_case(_case("dup", engine="commit_A", metric=0.8))
        store.write_case(_case("dup", engine="commit_A", metric=0.8))
        assert store.count() == 1


class TestRecordReliability:
    def test_forward_compat_unknown_json_keys_ignored(self, store):
        """Records written by a *newer* engine (extra JSON fields) must stay
        readable -- schema drift must not brick the history."""
        store.write_case(_case("fc", engine="commit_A"))
        conn = sqlite3.connect(store.config.db_path)
        row = conn.execute("SELECT action_json FROM cases").fetchone()
        data = json.loads(row[0])
        data["future_field_from_v2"] = 42
        conn.execute("UPDATE cases SET action_json=?", (json.dumps(data),))
        conn.commit()
        conn.close()
        got = store.get_case("fc")
        assert got is not None and got.action.model_family == "logistic"

    def test_corrupt_row_does_not_break_store_reads(self, store):
        store.write_case(_case("good1", fp="ds_1"))
        store.write_case(_case("good2", fp="ds_2"))
        # simulate disk-level corruption / old-format row
        conn = sqlite3.connect(store.config.db_path)
        conn.execute(
            "INSERT INTO cases (case_id, dataset_fingerprint, schema_fingerprint,"
            " engine_commit, identity_hash, context_json, action_json,"
            " result_json, cost_json, diagnosis_json, metadata_json)"
            " VALUES ('bad','fp','sc','dev','h',"
            " 'NOT JSON','NOT JSON','NOT JSON','NOT JSON','NOT JSON','NOT JSON')"
        )
        conn.commit()
        conn.close()
        cases = store.all_cases()
        ids = {c.case_id for c in cases}
        assert {"good1", "good2"} <= ids
        assert "bad" not in ids
        assert store.get_case("bad") is None

    def test_roundtrip_preserves_record(self, store):
        rec = _case("rt", rows=500, model="gbm", metric=0.91, seed=7)
        store.write_case(rec)
        got = store.get_case("rt")
        assert got.dataset_fingerprint == rec.dataset_fingerprint
        assert got.context.task == rec.context.task
        assert got.action.model_family == "gbm"
        assert got.result.primary_metric_value == 0.91
        assert got.seed == 7
        assert got.is_success()

    def test_persisted_across_connections(self, tmp_path):
        db_path = str(tmp_path / "persist.db")
        s1 = ExperienceStore(StoreConfig(db_path=db_path))
        s1.write_case(_case("p1"))
        s1.close()
        s2 = ExperienceStore(StoreConfig(db_path=db_path))
        assert s2.get_case("p1") is not None
        assert s2.count() == 1
