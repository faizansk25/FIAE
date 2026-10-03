"""Regression tests for imbalanced-data CV (M29).

Before the fix, ``_evaluate_cv`` used plain KFold for classification. On
imbalanced targets a fold could contain a single class, making ROC-AUC
undefined, all HPO trials fail honestly, and the flagship example report
"No model trained". The fix switches classification to StratifiedKFold and
rebalances the example churn generator (>=10% positives).
"""

import os
import sys


import pytest

from fiae.orchestration.model_training import _evaluate_cv

sklearn = pytest.importorskip("sklearn")


class TestEvaluateCvImbalanced:
    def test_imbalanced_binary_gets_real_roc_auc(self):
        """90/10 imbalanced binary target must yield defined ROC-AUC folds,
        not the degenerate 'failed:roc_auc' marker."""
        X = [[float(i % 13), float(i % 7)] for i in range(100)]
        y = [1.0 if i < 10 else 0.0 for i in range(100)]  # 10% positives

        results = _evaluate_cv(
            _logistic(), X, y, task="classification", n_folds=5, seed=42
        )

        assert not str(results.get("scoring", "")).startswith("failed:")
        assert results["scoring"] == "roc_auc"
        assert 0.0 <= results["mean"] <= 1.0

    def test_few_positives_still_stratifies(self):
        """Even with only 6 positives, every fold must contain both classes."""
        X = [[float(i), float(i % 5)] for i in range(60)]
        y = [1.0] * 6 + [0.0] * 54

        results = _evaluate_cv(
            _logistic(), X, y, task="classification", n_folds=5, seed=42
        )

        assert not str(results.get("scoring", "")).startswith("failed:")
        assert results["scoring"] == "roc_auc"

    def test_regression_unaffected(self):
        """Regression targets still use plain KFold + MSE.

        M40: FIAE publishes the loss itself ("mse", minimize) rather than
        sklearn's negated scorer name, which was an optimization
        convention leaking into the domain contract.
        """
        X = [[float(i), float(i % 3)] for i in range(40)]
        y = [float(i) for i in range(40)]

        results = _evaluate_cv(
            _logistic(), X, y, task="regression", n_folds=4, seed=42
        )

        assert results["scoring"] == "mse"
        assert results["direction"] == "minimize"
        assert results["mean"] >= 0.0, "MSE must be a positive loss"


class TestExampleDatasetBalance:
    def test_churn_generator_target_is_stratifiable(self):
        """The flagship example's synthetic churn target must have both
        classes well represented (>=10% positives) so every example run
        trains real models instead of failing CV."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "churn"))
        try:
            from run_api import generate_dataset  # type: ignore[import-not-found]
        finally:
            sys.path.pop(0)

        import tempfile

        from collections import Counter

        tmp = tempfile.mkdtemp()
        csv_path = os.path.join(tmp, "churn.csv")
        generate_dataset(csv_path)

        with open(csv_path, encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            ti = header.index("is_returned")
            values = [line.rstrip("\n").split(",")[ti] for line in f]

        dist = Counter(values)
        positives = int(dist.get("1", 0))
        assert positives >= 0.10 * len(values), (
            f"example churn target has only {positives}/{len(values)} "
            "positives -- CV folds will degenerate; rebalance the generator"
        )
        assert dist.get("0", 0) > 0


def _logistic():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=200)
