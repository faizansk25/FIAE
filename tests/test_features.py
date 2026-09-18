"""Tests for the Feature IR subsystem (docs 04, 05, 12).

Covers: registry, canonicalization, DAG (acyclic + duplicate collapse),
numeric/datetime/temporal transforms, and static domain checks.
"""

import math

import pytest

from fiae.features import (
    FeatureDAG,
    all_operators,
    check_domain,
    drop_identity_inputs,
    feature_signature,
    get_operator,
    make_feature_node,
    signature_hash,
)
from fiae.features import ops_numeric as _ops_num
from fiae.features import ops_datetime as _ops_dt
from fiae.features import ops_temporal as _ops_tmp


# ======================================================================
# Registry
# ======================================================================
class TestRegistry:
    def test_all_operators_nonempty(self):
        ops = all_operators()
        assert len(ops) > 0

    def test_get_operator_known(self):
        op = get_operator("log1p")
        assert op.name == "log1p"
        assert op.arity == "unary"
        assert op.leakage_class.name == "L0"

    def test_get_operator_unknown_raises(self):
        from fiae.errors import ErrorCode, FIAEError
        with pytest.raises(FIAEError) as exc:
            get_operator("nonexistent_op")
        assert exc.value.code is ErrorCode.FEATURE_PRECONDITION_FAILED

    def test_register_duplicate_raises(self):
        from fiae.errors import FIAEError
        op = get_operator("identity")
        with pytest.raises(FIAEError):
            get_operator("registry")  # not registered
            # Re-registering should fail:
            _ops_num.register(op) if hasattr(_ops_num, "register") else None

    def test_13_field_contract(self):
        """Every operator must declare the full doc 05 contract fields."""
        for op in all_operators():
            assert op.name
            assert op.family
            assert op.arity in ("unary", "binary", "datetime", "temporal")
            assert op.input_types
            assert op.output_type
            assert op.purpose
            assert isinstance(op.preconditions, tuple)
            assert op.fit_scope is not None
            assert op.null_policy
            assert op.leakage_class is not None
            assert op.target_permission is not None
            assert op.cost_shape
            assert op.generation_trigger
            assert isinstance(op.rejection_conditions, tuple)
            assert op.validation
            assert op.inference_requirement
            assert callable(op.transform)

    def test_numeric_catalog_completeness(self):
        expected_unary = {
            "identity", "log1p", "signed_log1p", "sqrt", "cbrt",
            "square", "cube", "abs", "sign", "reciprocal", "exp_clip",
            "zero_indicator", "positive_indicator", "missing_indicator",
            "finite_indicator",
        }
        expected_binary = {
            "sum", "difference", "product", "safe_ratio",
            "relative_difference", "min_pair", "max_pair", "mean_pair",
            "harmonic_mean_pair", "geometric_mean_pair", "euclidean_norm_pair",
            "absolute_difference",
        }
        registered = {op.name for op in all_operators() if op.family == "numeric"}
        assert expected_unary <= registered
        binary_registered = {
            op.name for op in all_operators() if op.family == "numeric interaction"
        }
        assert expected_binary <= binary_registered

    def test_domain_check_attached(self):
        dc_ops = {op.name for op in all_operators() if op.domain_check is not None}
        assert {"log1p", "sqrt", "safe_ratio", "harmonic_mean_pair"} <= dc_ops

    def test_temporal_operators_have_time_semantics(self):
        for op in all_operators():
            if op.family == "temporal":
                assert op.time_semantics == "sequential"


# ======================================================================
# Canonicalization
# ======================================================================
class TestCanonical:
    def test_signature_deterministic(self):
        s1 = feature_signature("sum", ["a", "b"])
        s2 = feature_signature("sum", ["a", "b"])
        assert s1 == s2

    def test_commutative_fold(self):
        from fiae.features.canonical import canonical_inputs
        op = get_operator("sum")
        folded = canonical_inputs(op, ["b", "a"])
        assert folded == ["a", "b"]

    def test_non_commutative_preserved(self):
        from fiae.features.canonical import canonical_inputs
        op = get_operator("difference")
        assert canonical_inputs(op, ["a", "b"]) == ["a", "b"]

    def test_make_feature_node_idempotent(self):
        n1 = make_feature_node("log1p", ["x"])
        n2 = make_feature_node("log1p", ["x"])
        assert n1.feature_id == n2.feature_id
        assert n1.lineage_hash == n2.lineage_hash

    def test_commutative_node_collapse(self):
        n1 = make_feature_node("sum", ["a", "b"])
        n2 = make_feature_node("sum", ["b", "a"])
        assert n1.feature_id == n2.feature_id

    def test_drop_identity_sum(self):
        assert drop_identity_inputs("sum", ["x", "0"]) == ["x"]
        assert drop_identity_inputs("sum", ["0", "0"]) == ["0", "0"]

    def test_drop_identity_product(self):
        assert drop_identity_inputs("product", ["x", "1"]) == ["x"]
        assert drop_identity_inputs("product", ["1", "1"]) == ["1", "1"]

    def test_signature_hash_changes_with_inputs(self):
        h1 = signature_hash("log1p", ["a"])
        h2 = signature_hash("log1p", ["b"])
        assert h1 != h2

    def test_feature_node_has_cost_hint(self):
        node = make_feature_node("log1p", ["x"])
        assert "cost_shape" in node.cost_hint
        assert "estimated_cpu_seconds_per_1k_rows" in node.cost_hint

    def test_time_semantics_propagated(self):
        node = make_feature_node("lag", ["x"], {"periods": 1})
        assert node.time_semantics == "sequential"

    def test_non_temporal_has_no_time_semantics(self):
        node = make_feature_node("log1p", ["x"])
        assert node.time_semantics is None


# ======================================================================
# DAG
# ======================================================================
class TestFeatureDAG:
    def test_empty_dag(self):
        g = FeatureDAG()
        assert len(g) == 0
        assert g.nodes == []
        assert g.roots() == []
        assert g.leaves() == []

    def test_add_or_create_creates_node(self):
        g = FeatureDAG()
        node = g.add_or_create("log1p", ["raw_x"])
        assert len(g) == 1
        assert node.feature_id in g
        assert g.get(node.feature_id) is node

    def test_duplicate_collapse(self):
        g = FeatureDAG()
        n1 = g.add_or_create("log1p", ["raw_x"])
        n2 = g.add_or_create("log1p", ["raw_x"])
        assert n1.feature_id == n2.feature_id
        assert len(g) == 1

    def test_commutative_duplicate_collapse(self):
        g = FeatureDAG()
        n1 = g.add_or_create("sum", ["a", "b"])
        n2 = g.add_or_create("sum", ["b", "a"])
        assert n1.feature_id == n2.feature_id
        assert len(g) == 1

    def test_identity_collapse_in_dag(self):
        g = FeatureDAG()
        n1 = g.add_or_create("sum", ["x", "0"])
        n2 = g.add_or_create("sum", ["x"])
        assert n1.feature_id == n2.feature_id
        assert len(g) == 1

    def test_topological_order(self):
        g = FeatureDAG()
        log = g.add_or_create("log1p", ["raw_x"])
        sq = g.add_or_create("square", ["raw_y"])
        combined = g.add_or_create("sum", [log.feature_id, sq.feature_id])
        order = g.topological_order()
        assert len(order) == 3
        ids = [n.feature_id for n in order]
        assert ids.index(log.feature_id) < ids.index(combined.feature_id)
        assert ids.index(sq.feature_id) < ids.index(combined.feature_id)

    def test_roots_and_leaves(self):
        g = FeatureDAG()
        a = g.add_or_create("log1p", ["raw_x"])
        b = g.add_or_create("square", ["raw_y"])
        c = g.add_or_create("sum", [a.feature_id, b.feature_id])
        leaves = g.leaves()
        leaf_ids = [n.feature_id for n in leaves]
        assert c.feature_id in leaf_ids
        roots = g.roots()
        root_ids = [n.feature_id for n in roots]
        assert a.feature_id in root_ids
        assert b.feature_id in root_ids

    def test_add_existing_returns_same(self):
        g = FeatureDAG()
        node = g.add_or_create("log1p", ["raw_x"])
        same = g.add_node(node)
        assert len(g) == 1
        assert same.feature_id == node.feature_id

    def test_cycle_detection(self):
        """Adding a node whose input transitively references itself must raise."""
        from fiae.features.canonical import make_feature_node as _mfn
        from fiae.features.dag import CycleError

        g = FeatureDAG()
        a = g.add_or_create("identity", ["raw_x"])
        b = _mfn("log1p", [a.feature_id])
        g.add_node(b)
        # Mutate b to point at c (which points back at b) — a cycle.
        c = _mfn("abs", [b.feature_id])
        b.inputs = [c.feature_id]
        with pytest.raises(CycleError):
            g._add_node(c)


# ======================================================================
# Numeric transforms
# ======================================================================
class TestNumericTransforms:
    def test_tf_identity_null_preserved(self):
        assert _ops_num.tf_identity([1.0, None, 3.0]) == [1.0, None, 3.0]

    def test_tf_log1p_domain_violation(self):
        out = _ops_num.tf_log1p([0.0, -2.0, 3.0])
        # -2.0 is below log1p domain (-1); -> explicit missing
        assert out == [math.log1p(0.0), None, math.log1p(3.0)]

    def test_tf_log1p_monotonicity(self):
        vals = [0.0, 1.0, 2.0, 3.0, 4.0]
        out = _ops_num.tf_log1p(vals)
        assert out == sorted(out)  # monotonic increasing

    def test_tf_log1p_null_preserved(self):
        assert _ops_num.tf_log1p([1.0, None, 2.0]) == [math.log1p(1.0), None, math.log1p(2.0)]

    def test_tf_sqrt_domain_violation(self):
        out = _ops_num.tf_sqrt([4.0, -1.0, 9.0])
        assert out == [2.0, None, 3.0]

    def test_tf_signed_log1p(self):
        out = _ops_num.tf_signed_log1p([1.0, -1.0, 0.0])
        assert out[0] == math.log1p(1.0)
        assert out[1] == -math.log1p(1.0)  # copysign
        assert out[2] == 0.0

    def test_tf_cbrt_signed(self):
        out = _ops_num.tf_cbrt([-8.0, 0.0, 8.0])
        assert out[0] == -2.0
        assert out[1] == 0.0
        assert out[2] == 2.0

    def test_tf_square_overflow_guard(self):
        out = _ops_num.tf_square([10.0, 2e155, 3.0])
        assert out[0] == 100.0
        assert out[1] is None  # overflow -> explicit missing
        assert out[2] == 9.0

    def test_tf_cube(self):
        out = _ops_num.tf_cube([2.0, -3.0, None])
        assert out[0] == 8.0
        assert out[1] == -27.0
        assert out[2] is None

    def test_tf_abs(self):
        out = _ops_num.tf_abs([-5.0, 3.0, None])
        assert out == [5.0, 3.0, None]

    def test_tf_sign(self):
        out = _ops_num.tf_sign([-1.0, 0.0, 2.0, None])
        assert out == [-1, 0, 1, None]

    def test_tf_reciprocal_zero_policy(self):
        out = _ops_num.tf_reciprocal([1.0, 0.0, 2.0, None])
        assert out[0] == 1.0
        assert out[1] is None  # near-zero -> explicit missing
        assert out[2] == 0.5
        assert out[3] is None

    def test_tf_exp_clip_bounds(self):
        out = _ops_num.tf_exp_clip([-30.0, 10.0, 5.0])
        assert out[0] == math.exp(-20.0)  # clamped to lo
        assert out[1] == math.exp(10.0)   # clamped to hi
        assert out[2] == math.exp(5.0)

    def test_tf_zero_indicator(self):
        out = _ops_num.tf_zero_indicator([0.0, 1.0, None, 0])
        assert out == [True, False, None, True]

    def test_tf_missing_indicator(self):
        out = _ops_num.tf_missing_indicator([1.0, None, float("nan"), 3.0])
        assert out == [False, True, True, False]

    def test_tf_finite_indicator(self):
        out = _ops_num.tf_finite_indicator([1.0, float("inf"), float("nan"), 3.0])
        assert out == [True, False, False, True]

    def test_tf_sum(self):
        out = _ops_num.tf_sum([1.0, None, 3.0], [10.0, 5.0, None])
        assert out == [11.0, None, None]

    def test_tf_difference(self):
        out = _ops_num.tf_difference([5.0, None, 3.0], [2.0, 1.0, None])
        assert out == [3.0, None, None]

    def test_tf_safe_ratio(self):
        out = _ops_num.tf_safe_ratio([10.0, 5.0, 1.0], [2.0, 0.0, 0.5])
        assert out == [5.0, None, 2.0]  # zero denom -> explicit missing

    def test_tf_relative_difference(self):
        out = _ops_num.tf_relative_difference([11.0, 10.0], [10.0, 0.0])
        assert abs(out[0] - 1.0 / (10.0 + 1e-12)) < 1e-9
        assert out[1] == 10.0 / (0.0 + 1e-12)

    def test_tf_product(self):
        out = _ops_num.tf_product([2.0, None, 3.0], [4.0, 5.0, None])
        assert out == [8.0, None, None]

    def test_tf_harmonic_mean_pair(self):
        out = _ops_num.tf_harmonic_mean_pair([2.0, -1.0, 4.0], [4.0, 2.0, 0.0])
        # 2*2*4/(2+4) = 16/6
        assert abs(out[0] - 16.0 / 6.0) < 1e-10
        assert out[1] is None  # negative -> domain violation
        assert out[2] is None  # zero -> domain violation

    def test_tf_geometric_mean_pair(self):
        out = _ops_num.tf_geometric_mean_pair([4.0, -1.0, 9.0], [9.0, 2.0, 4.0])
        assert out[0] == 6.0  # sqrt(4*9)
        assert out[1] is None  # negative -> domain violation
        assert out[2] == 6.0  # sqrt(9*4)

    def test_tf_euclidean_norm_pair(self):
        out = _ops_num.tf_euclidean_norm_pair([3.0, None, 0.0], [4.0, 5.0, None])
        assert out == [5.0, None, None]  # 5-12-13 triangle-ish

    def test_tf_absolute_difference(self):
        out = _ops_num.tf_absolute_difference([5.0, None, 3.0], [2.0, 1.0, None])
        assert out == [3.0, None, None]

    def test_tf_mean_pair(self):
        out = _ops_num.tf_mean_pair([1.0, None, 3.0], [3.0, 5.0, None])
        assert out == [2.0, None, None]

    def test_tf_min_pair(self):
        out = _ops_num.tf_min_pair([1.0, None, 3.0], [2.0, 5.0, None])
        assert out == [1.0, None, None]

    def test_tf_max_pair(self):
        out = _ops_num.tf_max_pair([1.0, None, 3.0], [2.0, 5.0, None])
        assert out == [2.0, None, None]


# ======================================================================
# Domain checks
# ======================================================================
class TestDomainChecks:
    def test_dc_log1p_passes_clean(self):
        op = get_operator("log1p")
        assert check_domain(op, ([1.0, 2.0, 3.0],), {}) is None

    def test_dc_log1p_fails_on_negatives(self):
        op = get_operator("log1p")
        reason = check_domain(op, ([-2.0, -3.0, -1.0],), {})
        assert reason is not None
        assert "log1p" in reason

    def test_dc_sqrt_passes_nonnegative(self):
        op = get_operator("sqrt")
        assert check_domain(op, ([0.0, 1.0, 4.0],), {}) is None

    def test_dc_sqrt_fails_on_negatives(self):
        op = get_operator("sqrt")
        reason = check_domain(op, ([-1.0, -2.0, -3.0],), {})
        assert reason is not None
        assert "sqrt" in reason

    def test_dc_sqrt_tolerates_few_negatives(self):
        op = get_operator("sqrt")
        vals = [1.0] * 19 + [-1.0]
        assert check_domain(op, (vals,), {}) is None

    def test_dc_safe_ratio_fails_on_zeros(self):
        op = get_operator("safe_ratio")
        reason = check_domain(op, ([1.0, 2.0], [0.0, 0.0]), {})
        assert reason is not None
        assert "zero" in reason.lower()

    def test_dc_safe_ratio_passes(self):
        op = get_operator("safe_ratio")
        assert check_domain(op, ([1.0, 2.0], [10.0, 5.0]), {}) is None

    def test_dc_harmonic_fails_on_nonpositive(self):
        op = get_operator("harmonic_mean_pair")
        reason = check_domain(op, ([-1.0, -2.0], [3.0, 4.0]), {})
        assert reason is not None
        assert "harmonic" in reason.lower()

    def test_dc_none_for_no_domain_check(self):
        op = get_operator("identity")
        assert check_domain(op, ([1.0],), {}) is None


# ======================================================================
# Datetime transforms
# ======================================================================
class TestDatetimeTransforms:
    def test_tf_year(self):
        out = _ops_dt.tf_year(["2023-05-15", "2024-01-01", None, "bad"])
        assert out == [2023, 2024, None, None]

    def test_tf_month(self):
        out = _ops_dt.tf_month(["2023-05-15", "2023-12-01"])
        assert out == [5, 12]

    def test_tf_quarter(self):
        out = _ops_dt.tf_quarter(["2023-01-15", "2023-04-15", "2023-07-15", "2023-10-15"])
        assert out == [1, 2, 3, 4]

    def test_tf_day_of_week(self):
        out = _ops_dt.tf_day_of_week(["2023-05-15"])  # Monday
        assert out == [0]

    def test_tf_is_weekend(self):
        # 2023-05-15 is Monday; 2023-05-20 is Saturday
        out = _ops_dt.tf_is_weekend(["2023-05-15", "2023-05-20"])
        assert out == [False, True]

    def test_tf_is_month_start(self):
        out = _ops_dt.tf_is_month_start(["2023-05-01", "2023-05-15"])
        assert out == [True, False]

    def test_tf_is_month_end(self):
        out = _ops_dt.tf_is_month_end(["2023-05-31", "2023-05-15"])
        assert out == [True, False]

    def test_tf_hour(self):
        out = _ops_dt.tf_hour(["2023-05-15 14:30:00"])
        assert out == [14]

    def test_tf_datetime_null_preserved(self):
        out = _ops_dt.tf_year(["2023-01-01", None, "2024-06-15"])
        assert out == [2023, None, 2024]

    def test_tf_datetime_invalid_parse(self):
        out = _ops_dt.tf_year(["not-a-date", "2023-01-01"])
        assert out == [None, 2023]


# ======================================================================
# Temporal transforms
# ======================================================================
class TestTemporalTransforms:
    def test_tf_lag(self):
        out = _ops_tmp.tf_lag([1.0, 2.0, 3.0, 4.0], periods=1)
        assert out == [None, 1.0, 2.0, 3.0]

    def test_tf_lag_periods_2(self):
        out = _ops_tmp.tf_lag([10.0, 20.0, 30.0, 40.0, 50.0], periods=2)
        assert out == [None, None, 10.0, 20.0, 30.0]

    def test_tf_lead(self):
        out = _ops_tmp.tf_lead([1.0, 2.0, 3.0, 4.0], periods=1)
        assert out == [2.0, 3.0, 4.0, None]

    def test_tf_diff(self):
        out = _ops_tmp.tf_diff([10.0, 15.0, 7.0, 20.0], periods=1)
        assert out == [None, 5.0, -8.0, 13.0]

    def test_tf_pct_change(self):
        out = _ops_tmp.tf_pct_change([100.0, 110.0, 99.0], periods=1)
        assert abs(out[1] - 0.1) < 1e-9
        assert out[2] is None or abs(out[2] - (-0.1)) < 1e-9

    def test_tf_rolling_mean(self):
        out = _ops_tmp.tf_rolling_mean([1.0, 2.0, 3.0, 4.0, 5.0], window=3)
        assert out == [None, None, 2.0, 3.0, 4.0]

    def test_tf_rolling_sum(self):
        out = _ops_tmp.tf_rolling_sum([1.0, 2.0, 3.0, 4.0], window=2)
        assert out == [None, 3.0, 5.0, 7.0]

    def test_tf_rolling_max(self):
        out = _ops_tmp.tf_rolling_max([3.0, 1.0, 4.0, 1.0, 5.0], window=3)
        assert out == [None, None, 4.0, 4.0, 5.0]

    def test_tf_rolling_min(self):
        out = _ops_tmp.tf_rolling_min([3.0, 1.0, 4.0, 1.0, 5.0], window=3)
        assert out == [None, None, 1.0, 1.0, 1.0]

    def test_tf_rolling_std(self):
        out = _ops_tmp.tf_rolling_std([2.0, 2.0, 2.0, 2.0], window=2)
        # All same values -> std = 0
        assert out == [None, 0.0, 0.0, 0.0]

    def test_tf_rolling_count(self):
        out = _ops_tmp.tf_rolling_count([1.0, None, 3.0, None], window=2)
        assert out == [1, 1, 1, 1]

    def test_tf_rolling_no_future(self):
        """Rolling windows must never include future values (insert-after-emit)."""
        vals = [10.0, 0.0, 0.0, 0.0, 0.0]
        out = _ops_tmp.tf_rolling_mean(vals, window=3)
        # At position 2, the window is [10, 0, 0] -> mean = 3.33
        # At position 3, the window is [0, 0, 0] -> mean = 0.0
        # The future 0s don't affect position 1 (only [10, 0] available).
        assert out[2] == 10.0 / 3.0
        assert out[3] == 0.0

    def test_tf_lag_null_preserved(self):
        out = _ops_tmp.tf_lag([1.0, None, 3.0, 4.0], periods=1)
        assert out == [None, 1.0, None, 3.0]

    def test_tf_diff_null_preserved(self):
        out = _ops_tmp.tf_diff([10.0, None, 7.0], periods=1)
        assert out == [None, None, None]

