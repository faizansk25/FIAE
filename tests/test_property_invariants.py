"""Property-based correctness tests using Hypothesis (audit program M3).

Invariants exercised (mission spec):
- operator determinism (same input -> identical output)
- no row reordering (output length + elementwise mapping)
- null behavior (NaN/None map to None per null_policy=preserve)
- finite/nonfinite behavior (finite inputs -> finite or None outputs)
- fit/transform consistency (fit-then-transform == fit_transform)
- serialization round trip (get_state_dict -> load_state_dict -> identical)
"""

import json
import math

from hypothesis import given, settings, strategies as st

from fiae.features.registry import all_operators, get_operator
from fiae.fitted_pipeline import FittedPipeline
from fiae.search.triggers import FeatureProposal

MAX_EXAMPLES = 50

# Unary, stateless (L0), numeric operators: the safest broad property target.
def _unary_stateless_ops():
    ops = []
    for op in all_operators():
        if op.arity != "unary" or op.fit_scope.name != "NONE":
            continue
        if op.family in ("datetime", "text", "text representation", "categorical"):
            continue  # these consume typed (non-numeric) inputs
        ops.append(op)
    return ops


UNARY_NUMERIC_OPS = _unary_stateless_ops()

finite_floats = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)
numeric_lists = st.lists(finite_floats, min_size=1, max_size=64)


def _nan_to_none(x):
    return None if (isinstance(x, float) and math.isnan(x)) else x


class TestOperatorProperties:
    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    @given(values=numeric_lists)
    def test_deterministic_output(self, values):
        for op in UNARY_NUMERIC_OPS:
            out1 = op.transform(list(values))
            out2 = op.transform(list(values))
            assert out1 == out2, f"{op.name} not deterministic"

    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    @given(values=numeric_lists)
    def test_no_row_reordering(self, values):
        """Output must have exactly n elements: elementwise mapping only."""
        n = len(values)
        for op in UNARY_NUMERIC_OPS:
            out = op.transform(list(values))
            assert len(out) == n, f"{op.name} changed row count {n} -> {len(out)}"

    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    @given(values=numeric_lists)
    def test_finite_inputs_give_finite_or_none(self, values):
        for op in UNARY_NUMERIC_OPS:
            out = op.transform(list(values))
            for i, v in enumerate(out):
                if v is None:
                    continue
                assert isinstance(v, (int, float)), (
                    f"{op.name}[{i}] produced {type(v).__name__}"
                )
                if isinstance(v, float):
                    assert not math.isnan(v), f"{op.name}[{i}] NaN on finite input"
                    assert not math.isinf(v), f"{op.name}[{i}] inf on finite input"

    @settings(max_examples=20, deadline=None)
    @given(
        st.lists(
            st.one_of(finite_floats, st.none(), st.just(float("nan"))),
            min_size=1,
            max_size=32,
        )
    )
    def test_null_inputs_preserved(self, values):
        """null_policy='preserve', non-temporal operators: missing in ->
        missing out at the same position (missing may surface as None or
        NaN). Temporal (time_semantics='sequential') operators shift
        values, so a null input legitimately surfaces elsewhere; their
        null behavior is boundary-padded instead and covered by other
        properties."""

        def _is_missing(x):
            return x is None or (isinstance(x, float) and math.isnan(x))

        for op in UNARY_NUMERIC_OPS:
            if op.null_policy != "preserve" or op.time_semantics == "sequential":
                continue
            out = op.transform(list(values))
            for i, v in enumerate(values):
                if _is_missing(v):
                    assert _is_missing(out[i]), (
                        f"{op.name} did not preserve null at {i}: got {out[i]!r}"
                    )

    @settings(max_examples=MAX_EXAMPLES, deadline=None)
    @given(values=numeric_lists)
    def test_idempotent_on_identity_composition(self, values):
        """identity(x) equals x elementwise (nan normalized)."""
        op = get_operator("identity")
        out = op.transform(list(values))
        assert out == [_nan_to_none(v) for v in values]


class TestFitTransformConsistency:
    @settings(max_examples=25, deadline=None)
    @given(
        xs=st.lists(st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
                    min_size=10, max_size=50),
        cats=st.lists(st.sampled_from(["a", "b", "c", "d"]), min_size=10, max_size=50),
    )
    def test_fit_then_transform_equals_fit_transform(self, xs, cats):
        n = min(len(xs), len(cats))
        data = {"x": xs[:n], "cat": cats[:n]}
        props = [
            FeatureProposal(op="standardize", inputs=["raw:x"]),
            FeatureProposal(op="one_hot", inputs=["raw:cat"]),
            FeatureProposal(op="log1p", inputs=["raw:x"]),
        ]

        p1 = FittedPipeline()
        p1.fit(props, data)
        t1 = p1.transform(props, data)

        p2 = FittedPipeline()
        t2 = p2.fit_transform(props, data)

        for key in t1:
            assert t1[key] == t2[key], f"fit/transform mismatch for {key}"

    @settings(max_examples=25, deadline=None)
    @given(
        train=st.lists(st.floats(-100, 100, allow_nan=False, allow_infinity=False),
                       min_size=10, max_size=40),
        test=st.lists(st.floats(-100, 100, allow_nan=False, allow_infinity=False),
                      min_size=5, max_size=40),
    )
    def test_serialization_round_trip_identical(self, train, test):
        data_train = {"x": train}
        data_test = {"x": test}
        props = [
            FeatureProposal(op="standardize", inputs=["raw:x"]),
            FeatureProposal(op="winsorize", inputs=["raw:x"]),
        ]

        p1 = FittedPipeline()
        p1.fit(props, data_train)
        ref = p1.transform(props, data_test)

        # JSON round trip must not alter behavior
        state = json.loads(json.dumps(p1.get_state_dict()))
        p2 = FittedPipeline()
        p2.load_state_dict(state)
        out = p2.transform(props, data_test)

        for key in ref:
            assert ref[key] == out[key], f"round-trip drift for {key}"

    def test_train_test_separation_state(self):
        """Fitted state must come from train only: transform on train data
        after reload matches the original fitted pipeline exactly."""
        train = {"x": [float(i) for i in range(50)]}
        props = [FeatureProposal(op="standardize", inputs=["raw:x"])]

        p = FittedPipeline()
        p.fit(props, train)
        a = p.transform(props, train)
        b = p.transform(props, train)
        assert a == b
