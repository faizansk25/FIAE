"""Tests for sklearn-based dimensionality, cluster, and model-informed operators."""

from __future__ import annotations



from fiae.features.registry import all_operators


# ---------------------------------------------------------------------------
# Registry verification (all ops should be present when sklearn installed)
# ---------------------------------------------------------------------------
class TestSklearnRegistryPresence:
    def test_all_sklearn_ops_registered(self):
        ops = {o.name: o for o in all_operators()}
        expected = [
            "pca", "truncated_svd", "text_svd",
            "kmeans_label", "kmeans_distances",
            "residual_interaction_proposal", "tree_leaf_oof",
        ]
        for name in expected:
            assert name in ops, f"{name} not registered"

    def test_dimensionality_ops(self):
        ops = {o.name: o for o in all_operators()}
        for name in ["pca", "truncated_svd", "text_svd"]:
            assert ops[name].family == "dimensionality"
            assert ops[name].fit_scope.value == "training_fold"

    def test_cluster_ops(self):
        ops = {o.name: o for o in all_operators()}
        for name in ["kmeans_label", "kmeans_distances"]:
            assert ops[name].family == "cluster"
            assert ops[name].leakage_class.value == "L1"

    def test_model_informed_ops(self):
        ops = {o.name: o for o in all_operators()}
        for name in ["residual_interaction_proposal", "tree_leaf_oof"]:
            assert ops[name].family == "model-informed"
            assert ops[name].leakage_class.value == "L2"


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------
class TestPCA:
    def test_fit_and_transform(self):
        from fiae.features.ops_sklearn import tf_pca_fit, tf_pca_transform
        cols = [[1.0, 2.0, 3.0, 4.0, 5.0] * 2,
                [2.0, 4.0, 6.0, 8.0, 10.0] * 2,
                [1.0, 0.0, 1.0, 0.0, 1.0] * 2]
        state = tf_pca_fit(cols, n_components=2)
        assert state["n_components"] <= 2
        result = tf_pca_transform(cols, state)
        assert len(result) == 10
        assert all(len(row) == state["n_components"] for row in result)


class TestKMeans:
    def test_fit_and_label(self):
        from fiae.features.ops_sklearn import tf_kmeans_fit, tf_kmeans_label_transform
        cols = [[1.0] * 10 + [10.0] * 10,
                [1.0] * 10 + [10.0] * 10]
        state = tf_kmeans_fit(cols, n_clusters=2)
        labels = tf_kmeans_label_transform(cols, state)
        assert len(labels) == 20
        assert set(labels) <= {0, 1}

    def test_fit_and_distance(self):
        from fiae.features.ops_sklearn import tf_kmeans_fit, tf_kmeans_distance_transform
        cols = [[1.0] * 10 + [10.0] * 10,
                [1.0] * 10 + [10.0] * 10]
        state = tf_kmeans_fit(cols, n_clusters=2)
        dists = tf_kmeans_distance_transform(cols, state)
        assert len(dists) == 20
        assert all(len(d) == 2 for d in dists)
        assert all(all(v >= 0 for v in d) for d in dists)


class TestTreeLeaf:
    def test_fit_and_transform(self):
        from fiae.features.ops_model_informed import (
            tf_tree_leaf_oof_fit, tf_tree_leaf_oof_transform
        )
        cols = [[1.0, 2.0, 3.0, 4.0, 5.0] * 4]
        target = [0, 0, 0, 1, 1] * 4
        state = tf_tree_leaf_oof_fit(cols, target, max_depth=3)
        leaves = tf_tree_leaf_oof_transform(cols, state)
        assert len(leaves) == 20
        assert len(set(leaves)) > 1  # multiple leaf partitions


class TestTextSVD:
    def test_fit_and_transform(self):
        from fiae.features.ops_sklearn import tf_text_svd_fit, tf_text_svd_transform
        docs = ["hello world test", "goodbye world foo", "hello test bar"] * 10
        state = tf_text_svd_fit(docs, n_components=2)
        result = tf_text_svd_transform(["hello world"], state)
        assert len(result) == 1
        assert len(result[0]) == state["n_components"]

    def test_null_text(self):
        from fiae.features.ops_sklearn import tf_text_svd_fit, tf_text_svd_transform
        docs = ["hello world"] * 5
        state = tf_text_svd_fit(docs)
        result = tf_text_svd_transform([None], state)
        assert result[0] == [0.0] * state["n_components"]


class TestResidualProposal:
    def test_basic(self):
        from fiae.features.ops_model_informed import tf_residual_proposal
        cols = [[float(i) for i in range(50)]]
        # Residuals correlated with feature magnitude
        residuals = [float(i % 10) for i in range(50)]
        proposals = tf_residual_proposal(cols, residuals, top_k=3)
        assert isinstance(proposals, list)
        assert all("column_idx" in p for p in proposals)
        assert all("strength" in p for p in proposals)


# ---------------------------------------------------------------------------
# Total operator count
# ---------------------------------------------------------------------------
class TestTotalCoverage:
    def test_95_operators(self):
        ops = all_operators()
        assert len(ops) == 95, f"Expected 95 operators, got {len(ops)}"

    def test_all_families_present(self):
        ops = all_operators()
        families = {o.family for o in ops}
        expected_families = {
            "numeric", "numeric interaction", "categorical",
            "categorical interaction", "target-aware categorical",
            "datetime", "temporal", "group aggregate",
            "text", "text representation",
            "dimensionality", "cluster", "model-informed",
        }
        assert expected_families.issubset(families)
