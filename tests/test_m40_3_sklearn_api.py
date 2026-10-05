"""M40.3: scikit-learn interoperability for FIAE feature engineering.

Two things are pinned here, and the second one is the one that matters:

1. ``SklearnFeatureTransformer`` composes inside a real
   ``sklearn.pipeline.Pipeline`` -- fit, transform, ``cross_val_score``,
   ``clone``, ``get_feature_names_out``.
2. It keeps working with **scikit-learn absent** (NFR-002: core imports are
   stdlib-only). An earlier draft inherited ``BaseEstimator`` and
   ``TransformerMixin`` unconditionally, which collapsed to a duplicate
   ``object`` base and broke the stdlib-only promise outright. That failure
   is reproduced here as a guard.
"""

import builtins
import random
import sys

import pytest

from fiae.errors import ErrorCode, FIAEError
from fiae.search.triggers import FeatureProposal
from fiae.sklearn_api import SklearnFeatureTransformer, _to_columnar


@pytest.fixture()
def _sklearn():
    """scikit-learn is optional; only the composition tests need it.

    Deliberately not a module-level importorskip: that deletes every test in
    the file on an environment without sklearn, including the stdlib-only
    ones below which are the whole point of this module.
    """
    return pytest.importorskip("sklearn")


@pytest.fixture()
def frame():
    """A small e-commerce-shaped DataFrame with a learnable target."""
    pd = pytest.importorskip("pandas")
    rng = random.Random(3)
    n = 400
    df = pd.DataFrame({
        "age": [rng.randint(18, 70) for _ in range(n)],
        "tenure": [rng.randint(1, 72) for _ in range(n)],
        "spend": [round(max(0.1, rng.random() * 50), 2) for _ in range(n)],
    })
    df["is_returned"] = [
        1 if (t > 60 or s < 10) else 0 for t, s in zip(df["tenure"], df["spend"])
    ]
    return df


@pytest.fixture()
def proposals():
    return [
        FeatureProposal(op="sqrt", inputs=["raw:spend"], params={}),
        FeatureProposal(op="log1p", inputs=["raw:spend"], params={}),
        FeatureProposal(op="standardize", inputs=["raw:age"], params={}),
    ]


class TestComposesInsideSklearnPipeline:
    def test_fit_transform_inside_pipeline(self, frame, proposals, _sklearn):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline

        pipe = Pipeline([
            ("fiae", SklearnFeatureTransformer(proposals)),
            ("rf", RandomForestClassifier(n_estimators=25, random_state=0)),
        ])
        X = frame[["age", "tenure", "spend"]]
        y = frame["is_returned"]

        pipe.fit(X, y)
        step = pipe.named_steps["fiae"]

        assert step.transform(X).shape == (len(X), 3)
        assert step.get_feature_names_out() == [
            "log1p(spend)", "sqrt(spend)", "standardize(age)",
        ]

    def test_cross_val_score_runs_through_the_step(self, frame, proposals, _sklearn):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline

        pipe = Pipeline([
            ("fiae", SklearnFeatureTransformer(proposals)),
            ("rf", RandomForestClassifier(n_estimators=25, random_state=0)),
        ])
        X = frame[["age", "tenure", "spend"]]
        y = frame["is_returned"]

        scores = cross_val_score(pipe, X, y, cv=3, scoring="roc_auc")
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_clone_and_gridsearch_params(self, proposals, _sklearn):
        from sklearn.base import clone

        step = SklearnFeatureTransformer(proposals, drop_missing=False)
        copied = clone(step)
        assert isinstance(copied, SklearnFeatureTransformer)
        assert sorted(step.get_params(deep=False)) == [
            "columns", "drop_missing", "proposals",
        ]

    def test_set_output_feature_names_convention(self, frame, proposals):
        step = SklearnFeatureTransformer(proposals)
        step.fit(frame[["age", "tenure", "spend"]])
        assert list(step.get_feature_names_out()) == list(step.feature_names_out_)


class TestStdlibOnlyFallback:
    """NFR-002: core imports must not require scikit-learn."""

    def test_import_and_use_without_sklearn(self, monkeypatch):
        """The regression: duplicate-``object`` base broke the stdlib path."""
        real_import = builtins.__import__

        def blocked(name, *a, **k):
            if name == "sklearn" or name.startswith("sklearn."):
                raise ImportError("sklearn blocked for this test")
            return real_import(name, *a, **k)

        saved = {m: sys.modules[m] for m in list(sys.modules)
                 if m.startswith("sklearn") or m.startswith("fiae")}
        for m in saved:
            del sys.modules[m]
        monkeypatch.setattr(builtins, "__import__", blocked)
        try:
            import fiae.sklearn_api as mod

            fresh = mod.SklearnFeatureTransformer(
                [FeatureProposal(op="sqrt", inputs=["raw:spend"], params={})],
                columns=["spend"],
            )
            out = fresh.fit([[4.0], [9.0], [16.0]]).transform([[25.0], [36.0]])
            assert [[round(v, 3) for v in row] for row in out] == [[5.0], [6.0]]
            assert sorted(fresh.get_params(deep=False)) == [
                "columns", "drop_missing", "proposals",
            ]
            fresh.set_params(drop_missing=False)
            assert fresh.drop_missing is False
        finally:
            monkeypatch.undo()
            for m in [k for k in sys.modules if k.startswith("fiae")]:
                del sys.modules[m]
            sys.modules.update(saved)


class TestColumnHandling:
    def test_missing_column_dropped_by_default(self):
        tr = SklearnFeatureTransformer(
            [FeatureProposal(op="sqrt", inputs=["raw:absent"], params={})])
        tr.fit([[1.0], [4.0]])
        assert tr.get_feature_names_out() == []

    def test_missing_column_raises_when_asked(self):
        tr = SklearnFeatureTransformer(
            [FeatureProposal(op="sqrt", inputs=["raw:absent"], params={})],
            drop_missing=False,
        )
        with pytest.raises(FIAEError) as exc:
            tr.fit([[1.0], [4.0]])
        assert exc.value.code == ErrorCode.SCHEMA_AMBIGUITY

    def test_transform_before_fit_raises(self):
        tr = SklearnFeatureTransformer([])
        with pytest.raises(FIAEError) as exc:
            tr.transform([[1.0]])
        assert exc.value.code == ErrorCode.FEATURE_PRECONDITION_FAILED

    def test_set_params_rejects_unknown_key(self):
        """Unknown keys are rejected.

        The exception type follows the active backend: sklearn's
        BaseEstimator raises ``ValueError``, while the stdlib fallback
        raises ``FIAEError``. The contract is the rejection, not the type.
        """
        with pytest.raises((FIAEError, ValueError)):
            SklearnFeatureTransformer([]).set_params(nonsense=1)

    def test_to_columnar_rejects_name_count_mismatch(self):
        with pytest.raises(FIAEError) as exc:
            _to_columnar([[1.0, 2.0], [3.0, 4.0]], columns=["only_one"])
        assert exc.value.code == ErrorCode.SCHEMA_AMBIGUITY

    def test_positional_names_are_synthesised(self):
        out = _to_columnar([[1.0, 2.0]])
        assert sorted(out) == ["f0", "f1"]
