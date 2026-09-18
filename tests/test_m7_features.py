"""Tests for M7 features: F5 gate, model registry, fitted-state pipeline."""

from __future__ import annotations


from fiae.funnel import (
    FunnelPolicy,
    _pearson,
    _to_floats_safe,
    f5_complementarity,
)
from fiae.model_registry import (
    ModelAssignment,
    ModelFamily,
    ROUTING_TABLE,
    assign_models,
    available_families,
    get_model_spec,
    route_model,
)
from fiae.search.triggers import FeatureProposal


# ---------------------------------------------------------------------------
# F5 Complementarity Gate
# ---------------------------------------------------------------------------
class TestPearson:
    def test_perfect_correlation(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0] * 3
        b = [2.0, 4.0, 6.0, 8.0, 10.0] * 3
        assert abs(_pearson(a, b) - 1.0) < 1e-6

    def test_negative_correlation(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0] * 3
        b = [5.0, 4.0, 3.0, 2.0, 1.0] * 3
        assert _pearson(a, b) < -0.9

    def test_zero_correlation(self):
        a = [1.0, -1.0, 1.0, -1.0, 1.0]
        b = [1.0, 1.0, -1.0, -1.0, 0.0]
        assert abs(_pearson(a, b)) < 0.1

    def test_too_few_pairs(self):
        a = [1.0, 2.0]
        b = [3.0, 4.0]
        assert _pearson(a, b) == 0.0


class TestToFloatsSafe:
    def test_basic(self):
        assert _to_floats_safe([1, 2.5, None, True, False]) == [1.0, 2.5, None, 1.0, 0.0]

    def test_strings(self):
        assert _to_floats_safe(["hello", "world"]) == [None, None]


class TestF5Complementarity:
    def test_empty_portfolio_passes(self):
        proposal = FeatureProposal(op="sqrt", inputs=["raw:a"], source="test")
        verdict = f5_complementarity(proposal, [1.0, 2.0, 3.0], [], FunnelPolicy())
        assert verdict.passed
        assert verdict.stage == "F5"

    def test_redundant_rejected(self):
        proposal = FeatureProposal(op="sqrt", inputs=["raw:a"], source="test")
        # Candidate identical to portfolio member -> correlation = 1.0
        candidate = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        portfolio = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]]
        verdict = f5_complementarity(proposal, candidate, portfolio, FunnelPolicy())
        assert not verdict.passed
        assert "redundant" in verdict.reason.lower() or "correlation" in verdict.reason.lower()

    def test_complementary_passed(self):
        proposal = FeatureProposal(op="diff", inputs=["raw:a", "raw:b"], source="test")
        # Candidate is noise, orthogonal to portfolio
        import random
        rng = random.Random(42)
        candidate = [rng.gauss(0, 1) for _ in range(50)]
        portfolio = [[float(i) for i in range(50)]]  # linear trend
        verdict = f5_complementarity(proposal, candidate, portfolio, FunnelPolicy())
        assert verdict.passed

    def test_moderate_correlation_passed(self):
        proposal = FeatureProposal(op="sum", inputs=["raw:a", "raw:b"], source="test")
        import random as _rng
        r = _rng.Random(42)
        x = [float(i) for i in range(50)]
        # Portfolio has heavy noise mixed with x -> correlation ~0.5
        portfolio = [[xi + r.gauss(0, 20.0) for xi in x]]
        verdict = f5_complementarity(proposal, x, portfolio, FunnelPolicy())
        assert verdict.passed  # corr < 0.95 threshold


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------
class TestModelRegistry:
    def test_all_families_registered(self):
        assert len(ROUTING_TABLE) == 12  # 12 families in doc 07

    def test_get_spec(self):
        spec = get_model_spec(ModelFamily.RANDOM_FOREST)
        assert spec is not None
        assert spec.family == ModelFamily.RANDOM_FOREST
        assert spec.is_ensemble

    def test_available_families_no_optional(self):
        families = available_families(include_optional=False)
        # At least sklearn families should be available
        assert len(families) >= 7
        assert all(f.backend.value == "sklearn" for f in families)

    def test_available_families_with_optional(self):
        families = available_families(include_optional=True)
        assert len(families) == 12


class TestModelRouting:
    def test_basic_routing(self):
        families = route_model("binary_classification", 1000, 20)
        assert len(families) > 0
        # Should not include logistic regression in regression routing
        families_reg = route_model("regression", 1000, 20)
        family_names = [f.family for f in families_reg]
        assert ModelFamily.LOGISTIC not in family_names

    def test_high_dimensional_prefers_linear(self):
        families = route_model("regression", 100, 500)
        linear_families = [ModelFamily.LINEAR, ModelFamily.SGD_LINEAR]
        # Linear family should be in top 3
        top3 = [f.family for f in families[:3]]
        assert any(f in top3 for f in linear_families)

    def test_small_data_prefers_simple(self):
        families = route_model("binary_classification", 100, 10)
        # Simple models should be near the top
        top_families = [f.family for f in families[:3]]
        assert any(
            f in top_families
            for f in (ModelFamily.LINEAR, ModelFamily.LOGISTIC,
                      ModelFamily.RANDOM_FOREST)
        )

    def test_categorical_boost(self):
        families_cat = route_model("binary_classification", 1000, 20, has_categoricals=True)
        families_no = route_model("binary_classification", 1000, 20, has_categoricals=False)
        # CatBoost should rank higher when categoricals present
        cat_rank_next = next(
            (i for i, f in enumerate(families_cat) if f.family == ModelFamily.CATBOOST), 99
        )
        cat_rank_no = next(
            (i for i, f in enumerate(families_no) if f.family == ModelFamily.CATBOOST), 99
        )
        assert cat_rank_next <= cat_rank_no


class TestModelAssignment:
    def test_assign_basic(self):
        assignment = assign_models("regression", 1000, 20)
        assert isinstance(assignment, ModelAssignment)
        assert len(assignment.families) > 0
        assert assignment.data_profile["task"] == "regression"

    def test_assign_high_dim(self):
        assignment = assign_models("regression", 100, 500)
        assert "high-dimensional" in assignment.reason
