"""M40 regression tests: model-correctness cluster.

External audit findings, each verified against the code first:

1. A requested 5-fold binary CV was silently run as 2 folds -- the fold
   count was capped by the number of *classes*, not by the limiting unit.
2. A generic CV fallback swapped ROC-AUC for accuracy on classification
   and reported ``-model.score()`` (i.e. -R^2) as "negative MSE".
3. Split planners bounded folds by row count: stratification ignored the
   minority class, group CV could emit folds with an empty validation set.
4. Every score was published as ``MetricValue(name="quality")``, so a
   ROC-AUC and a negated MSE were directly comparable.
"""

import random

import pytest

sklearn = pytest.importorskip("sklearn")

from fiae.contracts import Direction, TrialStatus
from fiae.orchestration.model_training import (
    TrialRunner,
    TrialSpec,
    _evaluate_cv,
    _get_sklearn_model,
    _resolve_scorer,
    primary_metric,
)
from fiae.problem.splits import group_kfold_indexes, stratified_kfold_indexes


def _binary_data(n=200):
    rng = random.Random(0)
    X = [[rng.random(), rng.random()] for _ in range(n)]
    y = [float(i % 2) for i in range(n)]
    return X, y


def _spy_cross_val(monkeypatch):
    """Capture the CV object actually handed to sklearn."""
    import sklearn.model_selection as sm

    seen: dict = {}
    real = sm.cross_val_score

    def spy(model, X, y, cv=None, scoring=None, **kw):
        seen["n_splits"] = getattr(cv, "n_splits", None)
        seen["scoring"] = scoring
        return real(model, X, y, cv=cv, scoring=scoring, **kw)

    monkeypatch.setattr(sm, "cross_val_score", spy)
    return seen


class TestRequestedFoldsAreHonoured:
    def test_binary_runs_the_requested_five_folds(self, monkeypatch):
        X, y = _binary_data()
        seen = _spy_cross_val(monkeypatch)
        model = _get_sklearn_model("random_forest", {"task": "classification"})
        _evaluate_cv(model, X, y, "classification", n_folds=5)
        # Pre-M40: min(n_folds, max(2, len(set(y))), ...) == 2 for every
        # binary task, so the requested 5 folds never happened.
        assert seen["n_splits"] == 5
        assert seen["scoring"] == "roc_auc"

    def test_folds_are_bounded_by_the_minority_class_not_the_class_count(self,
                                                                       monkeypatch):
        # 6 minority rows can support the 5 folds requested, so all 5 run.
        # Pre-M40 the class *count* (2) capped this at 2 regardless.
        y = [0.0] * 6 + [1.0] * 94
        X = [[float(i), float(i % 7)] for i in range(100)]
        seen = _spy_cross_val(monkeypatch)
        model = _get_sklearn_model("random_forest", {"task": "classification"})
        _evaluate_cv(model, X, y, "classification", n_folds=5)
        assert seen["n_splits"] == 5

    def test_tiny_minority_class_reduces_folds_rather_than_breaking(self,
                                                                  monkeypatch):
        y = [0.0] * 3 + [1.0] * 97
        X = [[float(i), float(i % 5)] for i in range(100)]
        seen = _spy_cross_val(monkeypatch)
        model = _get_sklearn_model("random_forest", {"task": "classification"})
        _evaluate_cv(model, X, y, "classification", n_folds=5)
        assert seen["n_splits"] == 3


class TestNoSilentMetricSubstitution:
    def test_multiclass_has_no_implicit_accuracy_fallback(self):
        # Pre-M40: any non-binary target silently switched to accuracy.
        assert _resolve_scorer("classification", [0.0, 1.0, 2.0] * 20) is None

    def test_multiclass_trial_fails_instead_of_reporting_accuracy(self):
        X = [[float(i), float(i % 3)] for i in range(120)]
        y = [float(i % 3) for i in range(120)]
        result = _evaluate_cv(
            _get_sklearn_model("random_forest", {"task": "classification"}),
            X, y, "classification", n_folds=5)
        assert str(result["scoring"]).startswith("failed:")

    def test_no_train_test_fallback_metric(self, monkeypatch):
        import sklearn.model_selection as sm

        def boom(*a, **kw):
            raise RuntimeError("scorer exploded")

        monkeypatch.setattr(sm, "cross_val_score", boom)
        X, y = _binary_data(60)
        result = _evaluate_cv(
            _get_sklearn_model("random_forest", {"task": "classification"}),
            X, y, "classification", n_folds=5)
        # Pre-M40 this returned an accuracy from a train/test split, or a
        # -R^2 labelled "negative MSE", under the name "fallback".
        assert str(result["scoring"]).startswith("failed:")
        assert result["scoring"] != "fallback"

    def test_regression_failure_is_not_reported_as_negative_mse(self,
                                                                monkeypatch):
        import sklearn.model_selection as sm

        def boom(*a, **kw):
            raise RuntimeError("scorer exploded")

        monkeypatch.setattr(sm, "cross_val_score", boom)
        X, y = _binary_data(60)
        result = _evaluate_cv(
            _get_sklearn_model("ridge", {"task": "regression"}),
            X, y, "regression", n_folds=5)
        assert "neg_mean_squared_error" not in str(result["scoring"]) or \
            str(result["scoring"]).startswith("failed:")


class TestMetricIdentityTravels:
    def test_binary_metric_is_named_roc_auc(self):
        X, y = _binary_data()
        trial = TrialRunner().run_trial(
            TrialSpec(model_family="random_forest", fold_count=5),
            X, y, "classification")
        assert trial.status is TrialStatus.COMPLETED
        m = primary_metric(trial)
        assert m.name == "roc_auc"
        assert m.direction is Direction.MAXIMIZE
        assert m.aggregation == "mean"

    def test_regression_metric_is_named_and_minimized(self):
        rng = random.Random(1)
        X = [[rng.random(), rng.random()] for _ in range(120)]
        y = [rng.random() * 10 for _ in range(120)]
        trial = TrialRunner().run_trial(
            TrialSpec(model_family="ridge", fold_count=5), X, y, "regression")
        m = primary_metric(trial)
        assert m.name == "neg_mean_squared_error"
        # Pre-M40 every score was declared MAXIMIZE, which inverts ranking
        # for any negated metric.
        assert m.direction is Direction.MINIMIZE

    def test_no_metric_is_ever_named_quality(self):
        X, y = _binary_data()
        trial = TrialRunner().run_trial(
            TrialSpec(model_family="random_forest"), X, y, "classification")
        names = {m.name for m in trial.metrics}
        assert "quality" not in names
        assert names == {"roc_auc", "cv_std"}


class TestFoldPlannersRespectTheirLimitingUnit:
    def test_stratified_folds_bounded_by_minority_class(self):
        y = ["a"] * 3 + ["b"] * 97
        folds = stratified_kfold_indexes(100, y, 5, seed=0)
        assert len(folds) == 3
        for _, val in folds:
            kinds = {y[i] for i in val}
            assert kinds == {"a", "b"}, "a fold lost a class entirely"

    def test_group_folds_never_emit_an_empty_validation_set(self):
        group_ids = ["g0"] * 40 + ["g1"] * 40 + ["g2"] * 20
        folds = group_kfold_indexes(100, group_ids, 5, seed=0)
        # Pre-M40: k stayed 5, so folds 3 and 4 had no validation rows.
        assert len(folds) == 3
        for _, val in folds:
            assert val, "group CV produced an empty validation fold"

    def test_group_folds_keep_groups_disjoint(self):
        group_ids = [f"g{i % 4}" for i in range(80)]
        for tr, val in group_kfold_indexes(80, group_ids, 4, seed=1):
            assert not ({group_ids[i] for i in tr}
                        & {group_ids[i] for i in val})
