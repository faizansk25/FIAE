"""scikit-learn interoperability for FIAE feature engineering.

Why this module exists
----------------------
``FittedPipeline.fit(proposals, columnar_data)`` takes the proposals as a
*fit argument*. scikit-learn's ``Pipeline`` calls ``transformer.fit(X, y)`` and
nothing else, so ``FittedPipeline`` cannot be dropped into an existing
scikit-learn workflow. That made FIAE a closed system: an ML engineer could
run it standalone but could not compose it with the estimators, CV and
grid-search infrastructure they already had.

``SklearnFeatureTransformer`` adapts the fitted-pipeline lifecycle to the
scikit-learn estimator contract:

    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from fiae import SklearnFeatureTransformer

    pipe = Pipeline([
        ("fiae", SklearnFeatureTransformer(proposals)),
        ("rf", RandomForestClassifier(random_state=0)),
    ])
    pipe.fit(X_train, y_train)

Dependency policy (NFR-002)
---------------------------
Core imports require only the standard library. scikit-learn and numpy are
optional: when present this class *also* inherits ``BaseEstimator`` and
``TransformerMixin`` so ``clone()``, ``GridSearchCV`` and ``set_output`` work
as expected; when absent the class still imports, fits and transforms, using a
stdlib ``get_params``/``set_params`` pair that follows the same contract.
"""

from __future__ import annotations

import inspect
from typing import Any

from .errors import ErrorCode, FIAEError
from .fitted_pipeline import FittedPipeline, canonical_feature_id
from .search.triggers import FeatureProposal

try:  # pragma: no cover - exercised by whichever branch the env provides
    from sklearn.base import BaseEstimator as _SkBaseEstimator, TransformerMixin as _SkTransformerMixin

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - stdlib-only install
    _SkBaseEstimator = object  # type: ignore[assignment,misc]
    _SkTransformerMixin = object  # type: ignore[assignment,misc]
    _SKLEARN_AVAILABLE = False


def _to_columnar(X: Any, columns: list[str] | None = None) -> dict[str, list]:
    """Normalise an array-like or DataFrame ``X`` into FIAE's columnar form.

    Column names come from the DataFrame when available so that proposals
    referencing ``raw:<name>`` resolve; otherwise positional names are
    synthesised and must be supplied via ``columns``.
    """
    if hasattr(X, "columns") and hasattr(X, "to_dict"):
        frame = X.to_dict("list")
        return {str(k): list(v) for k, v in frame.items()}

    rows = [list(row) for row in X]
    if columns is None:
        width = len(rows[0]) if rows else 0
        columns = [f"f{i}" for i in range(width)]
    if rows and len(columns) != len(rows[0]):
        raise FIAEError(
            code=ErrorCode.SCHEMA_AMBIGUITY,
            safe_message=(
                f"column count mismatch: {len(columns)} names "
                f"for {len(rows[0])} values"
            ),
        )
    return {name: [row[i] for row in rows] for i, name in enumerate(columns)}


# When scikit-learn is absent both mixins alias ``object``, and naming them
# twice would raise "duplicate base class object". Collapse to a single base.
_BASES: tuple[type, ...] = (
    (_SkTransformerMixin, _SkBaseEstimator) if _SKLEARN_AVAILABLE else (object,)
)


class SklearnFeatureTransformer(*_BASES):  # type: ignore[misc]
    """A scikit-learn transformer that applies FIAE feature proposals.

    Parameters
    ----------
    proposals:
        The feature proposals to materialise. Passed at construction time
        (not to ``fit``) because that is the scikit-learn convention.
    columns:
        Column names to use when ``X`` has no header. Required when passing a
        bare list-of-lists whose features are referenced by name.
    drop_missing:
        When True (default) a proposal whose input column is absent is
        skipped; when False a missing column raises instead.

    Attributes
    ----------
    feature_names_out_:
        Names of the produced columns, in output order. Matches
        scikit-learn's ``get_feature_names_out`` convention.
    """

    def __init__(
        self,
        proposals: list[FeatureProposal] | None = None,
        columns: list[str] | None = None,
        drop_missing: bool = True,
    ) -> None:
        # scikit-learn requires __init__ to store parameters unmodified and
        # do no validation here; all checks happen in fit().
        self.proposals = proposals
        self.columns = columns
        self.drop_missing = drop_missing
        self.feature_names_out_: list[str] = []

    # -- scikit-learn parameter protocol (stdlib fallback) --------------------
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor parameters, mirroring ``BaseEstimator.get_params``."""
        if _SKLEARN_AVAILABLE:  # pragma: no cover - sklearn base already provides it
            return super().get_params(deep=deep)
        return {
            name: getattr(self, name)
            for name in inspect.signature(self.__init__).parameters
            if name != "self"
        }

    def set_params(self, **params: Any) -> "SklearnFeatureTransformer":
        """Set constructor parameters, mirroring ``BaseEstimator.set_params``."""
        if _SKLEARN_AVAILABLE:  # pragma: no cover - sklearn base already provides it
            super().set_params(**params)
            return self
        valid = set(inspect.signature(self.__init__).parameters) - {"self"}
        for key, value in params.items():
            if key not in valid:
                raise FIAEError(
                    code=ErrorCode.FEATURE_PRECONDITION_FAILED,
                    safe_message=(
                        f"invalid parameter {key!r} for SklearnFeatureTransformer"
                    ),
                )
            setattr(self, key, value)
        return self

    # -- fit / transform ------------------------------------------------------
    def fit(self, X: Any, y: Any = None) -> "SklearnFeatureTransformer":
        """Fit L1 operators on ``X``. ``y`` is accepted and ignored.

        FIAE's proposals are unsupervised transforms of the input columns, so
        the target plays no part in fitting them.
        """
        proposals = self.proposals or []
        columnar = _to_columnar(X, self.columns)

        usable: list[FeatureProposal] = []
        for proposal in proposals:
            missing = [
                (fid[4:] if fid.startswith("raw:") else fid)
                for fid in proposal.inputs
                if (fid[4:] if fid.startswith("raw:") else fid) not in columnar
            ]
            if missing:
                if self.drop_missing:
                    continue
                raise FIAEError(
                    code=ErrorCode.SCHEMA_AMBIGUITY,
                    safe_message=(
                        f"proposal {canonical_feature_id(proposal)} needs missing "
                        f"column(s): {sorted(set(missing))}"
                    ),
                )
            usable.append(proposal)

        self._pipeline = FittedPipeline()
        # transform() needs to know which proposals to materialise, so the
        # usable set is retained rather than recomputed from X at transform
        # time (that would silently drift if a column disappeared).
        self._usable = usable
        outputs = self._pipeline.fit(usable, columnar)
        self._ordered_names = sorted(outputs)
        self.feature_names_out_ = list(self._ordered_names)
        self.n_features_out_ = len(self._ordered_names)
        return self

    def transform(self, X: Any) -> Any:
        """Apply the fitted operators, returning a 2-D array/list of columns."""
        if not hasattr(self, "_pipeline"):
            raise FIAEError(
                code=ErrorCode.FEATURE_PRECONDITION_FAILED,
                safe_message="SklearnFeatureTransformer.transform called before fit",
            )
        columnar = _to_columnar(X, self.columns)
        outputs = self._pipeline.transform(self._usable, columnar)

        names = [n for n in self._ordered_names if n in outputs]
        rows = [[outputs[name][i] for name in names] for i in range(self._n_rows(columnar))]
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - numpy is optional
            return rows
        return np.asarray(rows, dtype=float)

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> Any:
        """Fit then transform. Present because TransformerMixin supplies it."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        """Output feature names, following the scikit-learn 1.0+ convention."""
        return list(self.feature_names_out_)

    @staticmethod
    def _n_rows(columnar: dict[str, list]) -> int:
        return len(next(iter(columnar.values()))) if columnar else 0
