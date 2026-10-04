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

from fiae.contracts import Direction, MetricValue, Task, TrialStatus
from fiae.orchestration.model_training import (
    TrialRunner,
    TrialSpec,
    _evaluate_cv,
    _get_sklearn_model,
    _resolve_spec,
    is_better,
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
        assert _resolve_spec("classification", [0.0, 1.0, 2.0] * 20) is None

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

    def test_regression_metric_is_a_positive_loss_named_mse(self):
        rng = random.Random(1)
        X = [[rng.random(), rng.random()] for _ in range(120)]
        y = [rng.random() * 10 for _ in range(120)]
        trial = TrialRunner().run_trial(
            TrialSpec(model_family="ridge", fold_count=5), X, y, "regression")
        m = primary_metric(trial)
        # M40: FIAE publishes the loss itself, not sklearn's negated
        # scorer. MSE is positive and minimized.
        assert m.name == "mse"
        assert m.direction is Direction.MINIMIZE
        assert m.value >= 0.0

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


class TestDirectionSemantics:
    """M40 item 6: sklearn's neg_* convention must not leak into FIAE."""

    def test_mse_a_beats_mse_b(self):
        a = MetricValue(name="mse", value=0.1, direction=Direction.MINIMIZE,
                        split="cv")
        b = MetricValue(name="mse", value=1.0, direction=Direction.MINIMIZE,
                        split="cv")
        # A small MSE is the better model even though 0.1 < 1.0.
        assert is_better(a, b)
        assert not is_better(b, a)

    def test_maximize_metric_still_ranks_normally(self):
        a = MetricValue(name="roc_auc", value=0.81, direction=Direction.MAXIMIZE,
                        split="cv")
        b = MetricValue(name="roc_auc", value=0.62, direction=Direction.MAXIMIZE,
                        split="cv")
        assert is_better(a, b)
        assert not is_better(b, a)

    def test_different_identities_are_never_compared(self):
        a = MetricValue(name="roc_auc", value=0.9, direction=Direction.MAXIMIZE,
                        split="cv")
        b = MetricValue(name="mse", value=0.01, direction=Direction.MINIMIZE,
                        split="cv")
        assert not is_better(a, b)
        assert not is_better(b, a)

    def test_conflicting_direction_for_one_name_is_refused(self):
        a = MetricValue(name="mse", value=0.1, direction=Direction.MINIMIZE,
                        split="cv")
        b = MetricValue(name="mse", value=0.2, direction=Direction.MAXIMIZE,
                        split="cv")
        with pytest.raises(ValueError, match="conflicting"):
            is_better(a, b)


class TestDegenerateFoldCountsFailExplicitly:
    def test_single_minority_row_fails_instead_of_forcing_two_folds(self):
        y = [0.0, 1.0, 0.0]
        X = [[1.0], [2.0], [3.0]]
        result = _evaluate_cv(
            _get_sklearn_model("random_forest", {"task": "classification"}),
            X, y, "classification", n_folds=5)
        # max(2, 1) would hand sklearn a stratified split it cannot make.
        assert result["scoring"].startswith(
            "failed:insufficient_class_support_for_cv")
        assert result["folds"] == []

    def test_single_row_regression_fails_instead_of_forcing_two_folds(self):
        result = _evaluate_cv(
            _get_sklearn_model("ridge", {"task": "regression"}),
            [[1.0]], [0.5], "regression", n_folds=5)
        assert result["scoring"].startswith("failed:insufficient_rows_for_cv")

    def test_one_group_split_is_rejected_not_coerced(self):
        # max(2, ...) would have produced folds where validation and train
        # are the same rows.
        with pytest.raises(ValueError, match="at least 2 distinct groups"):
            group_kfold_indexes(20, ["only"] * 20, 5, seed=0)

    def test_singleton_class_split_is_rejected_not_coerced(self):
        with pytest.raises(ValueError, match="at least 2 members"):
            stratified_kfold_indexes(10, ["a"] * 9 + ["b"], 5, seed=0)


class TestFoldEvidenceIsReal:
    def test_fold_metrics_carry_distinct_per_fold_values(self):
        X, y = _binary_data(150)
        trial = TrialRunner().run_trial(
            TrialSpec(model_family="random_forest", fold_count=5),
            X, y, "classification")
        assert len(trial.fold_metrics) == 5
        values = [m.value for m in trial.fold_metrics]
        assert len(set(values)) > 1, (
            "fold metrics are copies of the aggregate, not real folds")
        mean = primary_metric(trial).value
        assert abs(sum(values) / len(values) - mean) < 1e-9
        assert [m.fold for m in trial.fold_metrics] == [0, 1, 2, 3, 4]


class TestEnsembleHonesty:
    def _trial(self, tid, name="roc_auc", direction=Direction.MAXIMIZE,
               folds=None, status=TrialStatus.COMPLETED):
        from fiae.contracts import MetricValue as MV
        from fiae.contracts import TrialResult
        return TrialResult(
            trial_id=tid, status=status,
            metrics=[MV(name=name, value=0.7, direction=direction,
                        split="cv_mean")],
            fold_metrics=[MV(name=name, value=v, direction=direction,
                             split="cv", fold=i)
                          for i, v in enumerate(folds or [0.6, 0.7, 0.8])],
        )

    def test_failed_trial_never_enters_an_ensemble(self):
        from fiae.orchestration.ensemble import (
            EnsemblePolicy, build_ensemble, check_eligibility,
        )

        policy = EnsemblePolicy()
        trials = [self._trial("ok"), self._trial("bad", status=TrialStatus.FAILED)]
        assert [t.trial_id for t in check_eligibility(trials, policy)] == ["ok"]
        spec = build_ensemble(trials, policy)
        assert spec is None or "bad" not in spec.member_trial_ids

    def test_ineligible_trials_produce_no_fabricated_weights(self):
        from fiae.pipeline.canonical import HOResult, phase_ensemble
        # Two COMPLETED trials with no fold evidence: nothing is eligible.
        hpo = HOResult(best_model="random_forest", best_metric="roc_auc",
                       trials=[self._trial("a"), self._trial("b")])
        for t in hpo.trials:
            t.fold_metrics = []
        result = phase_ensemble(None, hpo)
        assert result.members == 0
        assert result.weights == []
        assert result.marginal_value is False
        assert any("no eligible ensemble" in e for e in result.errors), (
            "silently fabricating an equal-weight ensemble")

    def test_metric_mismatch_is_rejected(self):
        from fiae.orchestration.ensemble import EnsemblePolicy, check_eligibility

        trials = [self._trial("a", name="roc_auc"),
                  self._trial("b", name="mse", direction=Direction.MINIMIZE)]
        assert [t.trial_id
                for t in check_eligibility(trials, EnsemblePolicy())] == ["a"]


class TestMulticlassProbeIsActuallyOneVsRest:
    """Item 3: the old probe solved one ridge against the raw class index
    and compared that single prediction vector with every OVR target."""

    def test_each_class_gets_its_own_fit(self, monkeypatch):
        from fiae import probe

        fits: list[list[float]] = []
        real = probe._solve_normal_equations

        def spy(a, b, lam=1e-6):
            fits.append(list(b))
            return real(a, b, lam)

        monkeypatch.setattr(probe, "_solve_normal_equations", spy)
        probe._cv_metric(
            [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]],
            [0.0, 1.0, 2.0, 0.0],
            [([0, 1], [2, 3])], Task.MULTICLASS, 1e-6,
        )
        # One solve per class per fold (3 classes, 1 fold), not one per fold.
        assert len(fits) == 3
        # Each fit gets its own binary target -- the old code passed the
        # raw class index once and reused the result three times.
        assert fits == [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]

    def test_one_row_gets_a_distinct_vector_per_class(self):
        from fiae.probe import _multiclass_ovr_scores

        classes = [0, 1, 2]
        y = [0.0, 1.0, 2.0, 2.0, 1.0, 0.0]
        scores = _multiclass_ovr_scores(
            [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]], y, [0, 1, 2], [3, 4, 5],
            classes, 1e-6,
        )
        assert len(scores) == 3
        for row in scores:
            # A shared prediction vector cannot do this; these are per-class.
            assert len({round(v, 12) for v in row}) == len(classes)
            assert all(0.0 <= v <= 1.0 for v in row)
            assert abs(sum(row) - 1.0) < 1e-9

    def test_each_row_scores_its_own_class_highest(self):
        from fiae.probe import _multiclass_ovr_scores

        # Three clusters, all present in the training fold, linearly
        # separable one-vs-rest. (A single 1-D ordered band is *not*:
        # one hyperplane per class cannot carve three intervals, which is
        # a real limit of a linear probe and not something OVR can fix.)
        jitter = [-0.5, -0.2, 0.1, 0.3, 0.5]
        y = [0.0] * 5 + [1.0] * 5 + [2.0] * 5
        x1 = jitter + [10.0 + v for v in jitter] + jitter
        x2 = jitter + jitter + [10.0 + v for v in jitter]
        scores = _multiclass_ovr_scores(
            [x1, x2], y, [0, 1, 6, 7, 12, 13], [3, 9, 14], [0, 1, 2], 1e-6,
        )
        assert [max(range(3), key=lambda j: row[j]) for row in scores] == [0, 1, 2]

    def test_degenerate_probe_reports_uniform_not_a_crash(self):
        """All-zero targets leave nothing to distribute; uniform is the
        no-information answer."""
        from fiae.probe import _multiclass_ovr_scores

        scores = _multiclass_ovr_scores(
            [[1.0, 1.0]], [0.0, 1.0], [0, 1], [0], [0, 1], 1e-6,
        )
        assert scores == [[0.5, 0.5]]

    def test_measure_is_not_called_brier(self):
        """Clipped linear-probe output is not a calibrated probability, so
        the name must not claim a proper scoring rule it has not earned."""
        from fiae.probe import probe_metric_name

        assert probe_metric_name(Task.MULTICLASS) == "multiclass_ovr_sq_error"
        assert probe_metric_name(Task.REGRESSION) == "rmse"
        assert probe_metric_name(Task.BINARY) == "brier"

    def test_f3_records_which_measure_it_used(self):
        from fiae.probe import f3_incremental_probe
        from fiae.search.records import FeatureAcceptanceRecord

        rng = random.Random(0)
        x = [rng.random() for _ in range(400)]
        y = [0 if v < 0.33 else (1 if v < 0.66 else 2) for v in x]
        rec = FeatureAcceptanceRecord(feature_id="c", operator="c")
        v = f3_incremental_probe(rec, [], x, y, Task.MULTICLASS)
        assert v.metrics["metric_name"] == "multiclass_ovr_sq_error"

    def test_f3_f4_f6_all_name_their_measure(self):
        from fiae.evaluate import f4_progressive_eval, f6_final_stability
        from fiae.probe import f3_incremental_probe
        from fiae.search.records import FeatureAcceptanceRecord

        rng = random.Random(1)
        n = 400
        x = [rng.random() for _ in range(n)]
        y = [0 if v < 0.33 else (1 if v < 0.66 else 2) for v in x]

        rec = FeatureAcceptanceRecord(feature_id="c", operator="c")
        v3 = f3_incremental_probe(rec, [], x, y, Task.MULTICLASS)
        assert v3.metrics["metric_name"] == "multiclass_ovr_sq_error"

        rec4 = FeatureAcceptanceRecord(feature_id="c", operator="c")
        v4 = f4_progressive_eval(rec4, [], x, y, Task.MULTICLASS)
        assert v4.metrics["metric_name"] == "multiclass_ovr_sq_error"

        rec6 = FeatureAcceptanceRecord(feature_id="c", operator="c")
        v6 = f6_final_stability(rec6, [], x, y, Task.MULTICLASS)
        assert v6.metrics["metric_name"] == "multiclass_ovr_sq_error"

    def test_informative_multiclass_feature_is_still_accepted(self):
        from fiae.probe import f3_incremental_probe
        from fiae.search.records import FeatureAcceptanceRecord

        rng = random.Random(0)
        n = 600
        x = [rng.random() for _ in range(n)]
        y = [0 if v < 0.33 else (1 if v < 0.66 else 2) for v in x]
        rec = FeatureAcceptanceRecord(feature_id="c", operator="c")
        v = f3_incremental_probe(rec, [], x, y, Task.MULTICLASS)
        assert v.passed, "informative feature rejected under corrected OVR"
        assert rec.incremental_gain and rec.incremental_gain > 0.01

    def test_pure_noise_still_rejected(self):
        from fiae.probe import f3_incremental_probe
        from fiae.search.records import FeatureAcceptanceRecord

        rng = random.Random(7)
        n = 400
        y = [float(rng.randrange(3)) for _ in range(n)]
        noise = [rng.random() for _ in range(n)]
        rec = FeatureAcceptanceRecord(feature_id="c", operator="c")
        v = f3_incremental_probe(rec, [], noise, y, Task.MULTICLASS)
        assert not v.passed
