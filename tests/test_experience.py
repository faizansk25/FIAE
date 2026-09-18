"""Tests for the Experience Store and Retrieval system (doc 06)."""

import os
import tempfile

import pytest

from fiae.contracts import Task, TrialStatus
from fiae.experience.case import (
    CaseAction,
    CaseContext,
    CaseCost,
    CaseDiagnosis,
    CaseRecord,
    CaseResult,
    FailureTag,
)
from fiae.experience.metafeatures import (
    DatasetMetaFeatures,
    MetaFeatureFamily,
    meta_feature_distance,
)
from fiae.experience.priors import bayesian_posterior
from fiae.experience.retrieval import (
    RetrievalEngine,
    RetrievalQuery,
    retrieve_priors,
)
from fiae.experience.store import ExperienceStore, StoreConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ExperienceStore(StoreConfig(db_path=os.path.join(tmpdir, "test.db")))
        yield store
        store.close()


def _make_case(
    case_id: str = "case-001",
    task: Task = Task.BINARY,
    success: bool = True,
    feature_portfolio=None,
    model_family: str = "rf",
) -> CaseRecord:
    ctx = CaseContext(task=task)
    action = CaseAction(
        feature_portfolio=feature_portfolio or ["num_transform"],
        model_family=model_family,
    )
    result = CaseResult(
        primary_metric_value=0.85 if success else 0.45,
        primary_metric_name="auc",
    )
    diagnosis = CaseDiagnosis(
        status=TrialStatus.COMPLETED if success else TrialStatus.FAILED,
        failure_tags=[] if success else [FailureTag.UNDERFIT_HIGH_BIAS],
    )
    cost = CaseCost(wall_time_s=10.0, cpu_time_s=5.0)
    return CaseRecord(
        case_id=case_id,
        dataset_fingerprint="fp-test-dataset",
        schema_fingerprint="schema-test",
        context=ctx,
        action=action,
        result=result,
        diagnosis=diagnosis,
        cost=cost,
    )


def _meta(rows: int, cats: int, fp: str) -> DatasetMetaFeatures:
    """Build a minimal DatasetMetaFeatures vector."""
    return DatasetMetaFeatures(
        dataset_fingerprint=fp,
        families={
            MetaFeatureFamily.SHAPE: {
                "log_rows": max(0.0, min(1.0, rows / 1000000.0)),
                "log_columns": 0.1,
            },
            MetaFeatureFamily.TYPE_COMPOSITION: {
                "numeric_fraction": 0.5,
                "categorical_fraction": cats / 100.0,
            },
        },
    )


# ---------------------------------------------------------------------------
# CaseRecord
# ---------------------------------------------------------------------------
class TestCaseRecord:
    def test_success_true(self):
        assert _make_case(success=True).is_success() is True

    def test_success_false(self):
        c = _make_case(success=False, feature_portfolio=[], model_family="")
        assert c.is_success() is False

    def test_identity_hash_deterministic(self):
        assert _make_case().identity_hash() == _make_case().identity_hash()


# ---------------------------------------------------------------------------
# ExperienceStore
# ---------------------------------------------------------------------------
class TestExperienceStore:
    def test_write_and_get(self, tmp_store):
        case = _make_case()
        ident = tmp_store.write_case(case)
        assert ident == case.identity_hash()
        got = tmp_store.get_case(case.case_id)
        assert got is not None
        assert got.case_id == case.case_id
        assert got.context.task == case.context.task

    def test_duplicate_replaces(self, tmp_store):
        tmp_store.write_case(_make_case(case_id="c"))
        tmp_store.write_case(_make_case(case_id="c"))
        assert tmp_store.count() == 1

    def test_all_cases_and_count(self, tmp_store):
        assert tmp_store.count() == 0
        # Distinct model families so identity hashes differ (case_id is not
        # part of the identity hash per doc 06).
        tmp_store.write_case(_make_case(case_id="a", model_family="rf"))
        tmp_store.write_case(_make_case(case_id="b", model_family="lgb"))
        assert tmp_store.count() == 2
        assert len(tmp_store.all_cases()) == 2

    def test_find_by_dataset(self, tmp_store):
        tmp_store.write_case(_make_case(case_id="a"))
        assert len(tmp_store.find_by_dataset("fp-test-dataset")) == 1
        assert len(tmp_store.find_by_dataset("nope")) == 0

    def test_clear(self, tmp_store):
        tmp_store.write_case(_make_case(case_id="a"))
        tmp_store.clear()
        assert tmp_store.count() == 0


# ---------------------------------------------------------------------------
# Meta-features
# ---------------------------------------------------------------------------
class TestMetaFeatures:
    def test_distance_identical_is_zero(self):
        m1 = _meta(100, 10, "fp1")
        m2 = _meta(100, 10, "fp1")
        assert meta_feature_distance(m1, m2) == pytest.approx(0.0)

    def test_distance_differs(self):
        m1 = _meta(1000, 10, "fp1")
        m2 = _meta(5, 90, "fp2")
        assert meta_feature_distance(m1, m2) > 0.0
# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------
class TestPriors:
    def test_bayesian_basic(self):
        m = bayesian_posterior(8, 10, 1.0, 1.0)
        assert 0.0 < m < 1.0

    def test_bayesian_no_data(self):
        assert bayesian_posterior(0, 0, 1.0, 1.0) == pytest.approx(0.5)

    def test_bayesian_all_success(self):
        assert bayesian_posterior(10, 10, 1.0, 1.0) > 0.8

    def test_bayesian_all_failure(self):
        assert bayesian_posterior(0, 10, 1.0, 1.0) < 0.2


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
class TestRetrieval:
    def _populate(self, store, n_success=5, n_fail=2):
        for i in range(n_success):
            store.write_case(_make_case(case_id=f"s{i}", success=True,
                                        feature_portfolio=["num_transform"],
                                        model_family="rf"))
        for i in range(n_fail):
            store.write_case(_make_case(case_id=f"f{i}", success=False,
                                        feature_portfolio=[],
                                        model_family=""))

    def test_no_cases_returns_low_confidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ExperienceStore(StoreConfig(db_path=os.path.join(tmpdir, "e.db")))
            try:
                prior = retrieve_priors(store, RetrievalQuery(task=Task.BINARY))
                assert prior.confidence == 0.0
            finally:
                store.close()

    def test_retrieve_returns_priors(self, tmp_store):
        self._populate(tmp_store)
        prior = retrieve_priors(tmp_store, RetrievalQuery(task=Task.BINARY))
        assert prior.source_count > 0
        assert prior.confidence >= 0.0

    def test_hard_filter_rejects_mismatched_task(self, tmp_store):
        self._populate(tmp_store)
        prior = retrieve_priors(tmp_store, RetrievalQuery(task=Task.MULTICLASS))
        assert prior.confidence == 0.0

    def test_feature_family_priors_populated(self, tmp_store):
        self._populate(tmp_store, n_success=10, n_fail=0)
        prior = retrieve_priors(tmp_store, RetrievalQuery(task=Task.BINARY))
        assert "num_transform" in prior.feature_family_priors
        fp = prior.feature_family_priors["num_transform"]
        assert fp.successes > 0
        assert fp.attempts > 0

    def test_engine_direct(self, tmp_store):
        self._populate(tmp_store)
        engine = RetrievalEngine(tmp_store)
        assert engine is not None
