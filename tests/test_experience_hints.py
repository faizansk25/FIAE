"""Tests for proposal source 4: experience-guided priors."""


from fiae.contracts import ColumnProfile, DatasetProfile, SemanticType, Task, TrialStatus
from fiae.experience import (
    CaseAction,
    CaseContext,
    CaseCost,
    CaseDiagnosis,
    CaseRecord,
    CaseResult,
    DatasetMetaFeatures,
)

from fiae.search.experience_hints import PriorPolicy, propose_from_experience


def _profile(n_num=4):
    cols = [
        ColumnProfile(
            name=f"num{i}", physical_dtype="float64",
            semantic_type=SemanticType.CONTINUOUS_NUMERIC,
            null_fraction=0.0, distinct_estimate=100, distinct_ratio=1.0,
        )
        for i in range(n_num)
    ]
    return DatasetProfile(
        dataset_fingerprint="d", rows_observed=100, columns=cols,
        rows_estimated=100,
    )


def _store_with_case(family="numeric", successes=9, attempts=10):
    """Seed a fresh in-memory store with ``successes``/``attempts`` outcomes.

    All cases share one dataset fingerprint and empty meta-features so the
    R2 distance to the query is 0 and retrieval confidence clears the OOD
    threshold while the family posterior stays at the requested rate.
    """
    from fiae.experience.store import ExperienceStore, StoreConfig

    store = ExperienceStore(StoreConfig(db_path=":memory:"))
    for i in range(successes):
        store.write_case(_case(f"ok-{i}", family, fp="fp-ok"))
    for i in range(attempts - successes):
        store.write_case(_case(f"bad-{i}", family, success=False, fp="fp-bad"))
    for i in range(max(0, 10 - attempts)):
        store.write_case(_case("fill", family, fp=f"fp-fill-{i}"))
    return store


def _case(case_id, family, success=True, fp="fp"):
    ctx = CaseContext(
        task=Task.REGRESSION,
        dataset_meta_features=DatasetMetaFeatures(dataset_fingerprint=fp),
    )
    return CaseRecord(
        case_id=case_id, dataset_fingerprint=f"fp-{case_id}",
        schema_fingerprint="s", engine_commit="c", context=ctx,
        action=CaseAction(feature_portfolio=[family], model_family="ridge"),
        result=CaseResult(primary_metric_name="rmse", primary_metric_value=1.0),
        cost=CaseCost(),
        diagnosis=CaseDiagnosis(
            status=TrialStatus.COMPLETED if success else TrialStatus.FAILED
        ),
    )


def test_proposes_for_successful_family():
    store = _store_with_case("numeric", 9, 10)
    out = propose_from_experience(_profile(), store, Task.REGRESSION)
    assert out, "expected proposals from a successful family"
    assert all(p.source == "experience:prior" for p in out)
    ops = {p.op for p in out}
    assert ops <= {"log1p", "square"}
    # one proposal per (column, op) pair, capped at 3 columns
    assert len(out) == 2 * min(3, 4)


def test_empty_store_yields_nothing():
    from fiae.experience.store import ExperienceStore, StoreConfig

    out = propose_from_experience(
        _profile(), ExperienceStore(StoreConfig(db_path=":memory:")), Task.REGRESSION
    )
    assert out == []


def test_low_confidence_yields_nothing():
    # single case = low confidence
    store = _store_with_case("numeric", 9, 10)
    pol = PriorPolicy(min_confidence=0.99)
    out = propose_from_experience(_profile(), store, Task.REGRESSION, pol)
    assert out == []


def test_low_posterior_family_skipped():
    store = _store_with_case("numeric", 2, 10)  # 0.2 posterior
    out = propose_from_experience(_profile(), store, Task.REGRESSION)
    assert out == []


def test_max_proposals_cap():
    store = _store_with_case("numeric", 10, 10)
    pol = PriorPolicy(max_proposals=5)
    out = propose_from_experience(_profile(), store, Task.REGRESSION, pol)
    assert len(out) == 5


def test_datetime_family_targets_datetime_columns():
    from fiae.search.experience_hints import _FAMILY_SEMANTICS
    prof = _profile()
    dt = ColumnProfile(
        name="ts", physical_dtype="datetime64",
        semantic_type=SemanticType.DATETIME, null_fraction=0.0,
        distinct_estimate=100, distinct_ratio=1.0,
    )
    prof.columns.append(dt)
    store = _store_with_case("datetime", 9, 10)
    out = propose_from_experience(prof, store, Task.REGRESSION)
    dt_ops = {p.op for p in out if p.inputs == ["raw:ts"]}
    assert dt_ops == {"day_of_week", "month"}
    assert _FAMILY_SEMANTICS["datetime"] == (SemanticType.DATETIME,)
