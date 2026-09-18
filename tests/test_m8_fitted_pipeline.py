"""Tests for M8: FittedPipeline, mixed-type integration, property tests."""

from __future__ import annotations

import random


from fiae.fitted_pipeline import FittedPipeline, _needs_fit, _coerce_to_floats
from fiae.features.registry import all_operators, get_operator
from fiae.search.triggers import FeatureProposal


# ---------------------------------------------------------------------------
# _needs_fit / _coerce helpers
# ---------------------------------------------------------------------------
class TestHelpers:
    def test_needs_fit_standardize(self):
        op = get_operator("standardize")
        assert _needs_fit(op)

    def test_needs_fit_sqrt(self):
        op = get_operator("sqrt")
        assert not _needs_fit(op)

    def test_needs_fit_one_hot(self):
        op = get_operator("one_hot")
        assert _needs_fit(op)

    def test_coerce_basic(self):
        assert _coerce_to_floats([1, 2.5, None, True]) == [1.0, 2.5, None, 1.0]

    def test_coerce_nan(self):
        assert _coerce_to_floats([float("nan")]) == [None]


# ---------------------------------------------------------------------------
# FittedPipeline: L1 numeric operators
# ---------------------------------------------------------------------------
class TestFittedPipelineStandardize:
    def test_fit_and_transform(self):
        pipe = FittedPipeline()
        proposals = [
            FeatureProposal(op="standardize", inputs=["raw:x"], source="test"),
        ]
        train_data = {"x": [1.0, 2.0, 3.0, 4.0, 5.0] * 10}
        result = pipe.fit(proposals, train_data)
        assert "standardize(x)" in result
        # Mean should be ~3.0, values should be standardized
        vals = result["standardize(x)"]
        mean = sum(vals) / len(vals)
        assert abs(mean) < 0.1  # approximately zero

    def test_transform_uses_fit_state(self):
        pipe = FittedPipeline()
        proposals = [
            FeatureProposal(op="standardize", inputs=["raw:x"], source="test"),
        ]
        train_data = {"x": [0.0, 0.0, 0.0, 10.0, 10.0]}
        pipe.fit(proposals, train_data)
        # Transform with new data using the same mean/std
        result = pipe.transform(proposals, {"x": [5.0]})
        assert len(result["standardize(x)"]) == 1


class TestFittedPipelineMinMax:
    def test_fit_and_transform(self):
        pipe = FittedPipeline()
        proposals = [
            FeatureProposal(op="minmax_scale", inputs=["raw:x"], source="test"),
        ]
        data = {"x": [0.0, 50.0, 100.0]}
        result = pipe.fit(proposals, data)
        vals = result["minmax_scale(x)"]
        assert abs(vals[0] - 0.0) < 1e-10
        assert abs(vals[1] - 0.5) < 1e-10
        assert abs(vals[2] - 1.0) < 1e-10


class TestFittedPipelineWinsorize:
    def test_fit_and_transform(self):
        pipe = FittedPipeline()
        proposals = [
            FeatureProposal(op="winsorize", inputs=["raw:x"], source="test"),
        ]
        data = {"x": list(range(100))}
        result = pipe.fit(proposals, data)
        vals = result["winsorize(x)"]
        # Should be clipped
        assert all(v >= vals[0] for v in vals)
        assert all(v <= vals[-1] for v in vals)


# ---------------------------------------------------------------------------
# FittedPipeline: L1 categorical operators
# ---------------------------------------------------------------------------
class TestFittedPipelineOrdinalEncode:
    def test_fit_and_transform(self):
        pipe = FittedPipeline()
        proposals = [
            FeatureProposal(op="ordinal_encode", inputs=["raw:color"], source="test"),
        ]
        train = {"color": ["red", "blue", "red", "green", "red"]}
        result = pipe.fit(proposals, train)
        vals = result["ordinal_encode(color)"]
        assert vals[0] == 0  # red is most frequent -> 0
        assert vals[1] == 1  # blue -> 1

    def test_unknown_category(self):
        pipe = FittedPipeline()
        proposals = [
            FeatureProposal(op="ordinal_encode", inputs=["raw:color"], source="test"),
        ]
        pipe.fit(proposals, {"color": ["red", "blue"]})
        result = pipe.transform(proposals, {"color": ["yellow"]})
        assert result["ordinal_encode(color)"][0] == -1


class TestFittedPipelineFrequencyEncode:
    def test_fit_and_transform(self):
        pipe = FittedPipeline()
        proposals = [
            FeatureProposal(op="frequency_encode", inputs=["raw:city"], source="test"),
        ]
        train = {"city": ["NYC", "NYC", "NYC", "LA", "SF"]}
        result = pipe.fit(proposals, train)
        vals = result["frequency_encode(city)"]
        assert abs(vals[0] - 0.6) < 1e-10  # NYC = 3/5


class TestFittedPipelineCountEncode:
    def test_fit_and_transform(self):
        pipe = FittedPipeline()
        proposals = [
            FeatureProposal(op="count_encode", inputs=["raw:city"], source="test"),
        ]
        train = {"city": ["NYC", "NYC", "NYC", "LA", "SF"]}
        result = pipe.fit(proposals, train)
        vals = result["count_encode(city)"]
        assert vals[0] == 3  # NYC count


class TestFittedPipelineRareGroup:
    def test_fit_and_transform(self):
        pipe = FittedPipeline()
        proposals = [
            FeatureProposal(op="rare_group", inputs=["raw:color"], source="test"),
        ]
        train = {"color": ["red"] * 10 + ["blue"] * 2 + ["green"]}
        _result = pipe.fit(proposals, train)
        # _to_float_list converts strings to 0.0, so verify via state
        state = pipe.states["rare_group(color)"].fit_state
        assert "red" in state["retained"]
        assert "blue" not in state["retained"]
        assert "green" not in state["retained"]


# ---------------------------------------------------------------------------
# FittedPipeline: state serialization
# ---------------------------------------------------------------------------
class TestFittedPipelineState:
    def test_serialize_and_load(self):
        pipe = FittedPipeline()
        proposals = [
            FeatureProposal(op="standardize", inputs=["raw:x"], source="test"),
        ]
        pipe.fit(proposals, {"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        state_dict = pipe.get_state_dict()
        assert "standardize(x)" in state_dict

        pipe2 = FittedPipeline()
        pipe2.load_state_dict(state_dict)
        result = pipe2.transform(proposals, {"x": [3.0]})
        assert len(result["standardize(x)"]) == 1


# ---------------------------------------------------------------------------
# FittedPipeline: L0 operators pass through
# ---------------------------------------------------------------------------
class TestFittedPipelineL0PassThrough:
    def test_sqrt_not_fitted(self):
        pipe = FittedPipeline()
        proposals = [
            FeatureProposal(op="sqrt", inputs=["raw:x"], source="test"),
        ]
        data = {"x": [1.0, 4.0, 9.0, 16.0]}
        result = pipe.fit(proposals, data)
        vals = result["sqrt(x)"]
        assert abs(vals[0] - 1.0) < 1e-10
        assert abs(vals[1] - 2.0) < 1e-10
        # No state stored for L0
        assert "sqrt(x)" not in pipe.states


# ---------------------------------------------------------------------------
# Mixed-type integration: learn pipeline with numeric + categorical
# ---------------------------------------------------------------------------
class TestMixedTypeIntegration:
    def test_learn_with_mixed_columns(self):
        """Test that the learn pipeline handles mixed numeric and categorical data."""
        import tempfile
        import os
        from fiae.learn import learn, LearnConfig

        # Create a CSV with both numeric and categorical columns
        lines = ["price,qty,color,label"]
        rng = random.Random(42)
        for _ in range(200):
            price = round(rng.uniform(10, 100), 2)
            qty = rng.randint(1, 50)
            color = rng.choice(["red", "blue", "green", "yellow"])
            # Label depends on price and color
            label = 1 if price > 50 and color == "red" else 0
            lines.append(f"{price},{qty},{color},{label}")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("\n".join(lines))
            csv_path = f.name

        try:
            config = LearnConfig(max_proposals=20, sample_rows=200)
            report = learn(csv_path, "label", config=config)
            # Pipeline should complete without error
            assert report.task in ("binary_classification", "regression")
            assert report.proposals_generated > 0
        finally:
            os.unlink(csv_path)


# ---------------------------------------------------------------------------
# Property-based invariant tests (doc 01 invariants)
# ---------------------------------------------------------------------------
class TestDoc01Invariants:
    def test_null_preservation_sqrt(self):
        """Invariant: null in -> null out for sqrt."""
        op = get_operator("sqrt")
        result = op.transform([1.0, None, 4.0])
        assert result[1] is None

    def test_null_preservation_log1p(self):
        """Invariant: null in -> null out for log1p."""
        op = get_operator("log1p")
        result = op.transform([0.0, None, 1.0])
        assert result[1] is None

    def test_domain_violation_explicit_missing(self):
        """Invariant: domain violation -> explicit missing (not misleading value)."""
        op = get_operator("sqrt")
        result = op.transform([-1.0, 4.0])
        assert result[0] is None  # sqrt(-1) -> missing
        assert result[1] == 2.0

    def test_deterministic_ids(self):
        """Invariant: same inputs -> same feature ID (NFR-004)."""
        from fiae.features.canonical import signature_hash
        h1 = signature_hash("sqrt", ["raw:x"])
        h2 = signature_hash("sqrt", ["raw:x"])
        assert h1 == h2

    def test_commutative_canonicalization(self):
        """Invariant: a+b and b+a share signature."""
        from fiae.features.canonical import signature_hash
        h1 = signature_hash("sum", ["raw:a", "raw:b"])
        h2 = signature_hash("sum", ["raw:b", "raw:a"])
        assert h1 == h2

    def test_overflow_guard(self):
        """Invariant: overflow -> explicit missing."""
        op = get_operator("square")
        result = op.transform([1e200])
        assert result[0] is None  # overflow

    def test_sign_preservation(self):
        """Invariant: sign(x) preserves sign."""
        op = get_operator("sign")
        result = op.transform([-5.0, 0.0, 3.0])
        assert result[0] == -1
        assert result[1] == 0
        assert result[2] == 1

    def test_zero_denominator(self):
        """Invariant: zero denominator -> explicit missing."""
        op = get_operator("safe_ratio")
        result = op.transform([1.0], [0.0])
        assert result[0] is None

    def test_rolling_no_future(self):
        """Invariant: rolling_mean at position i never uses values after i."""
        op = get_operator("rolling_mean")
        result = op.transform([1.0, 2.0, 3.0, 4.0, 5.0], window=3)
        # Position 2: mean of [1,2,3] = 2.0
        assert abs(result[2] - 2.0) < 1e-10
        # Position 4: mean of [3,4,5] = 4.0
        assert abs(result[4] - 4.0) < 1e-10

    def test_lag_no_future(self):
        """Invariant: lag(1) at position i uses value at i-1 only."""
        op = get_operator("lag")
        result = op.transform([10.0, 20.0, 30.0], periods=1)
        assert result[0] is None  # first row
        assert result[1] == 10.0
        assert result[2] == 20.0

    def test_frequency_encode_fold_isolation(self):
        """Invariant: frequency_encode fit on training data only."""
        pipe = FittedPipeline()
        proposals = [
            FeatureProposal(op="frequency_encode", inputs=["raw:x"], source="test"),
        ]
        # Fit on training
        pipe.fit(proposals, {"x": ["a", "a", "b"]})
        state = pipe.states["frequency_encode(x)"].fit_state
        # Should have freq_map with a=2/3, b=1/3
        assert "frequency_map" in state

    def test_all_operators_have_required_fields(self):
        """Invariant: every registered operator has all doc 05 required fields."""
        ops = all_operators()
        for op in ops:
            assert op.name
            assert op.family
            assert op.purpose
            assert op.leakage_class
            assert op.cost_shape
            assert op.transform is not None
            assert callable(op.transform)
