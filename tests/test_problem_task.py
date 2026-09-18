import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fiae.contracts import ColumnProfile, SemanticType, Task
from fiae.problem.task import (
    infer_task,
    make_problem,
    route_metrics,
)


def cp(distinct, st=SemanticType.CONTINUOUS_NUMERIC, phys=None, ratio=1.0):
    phys = phys or ("integer" if st in (SemanticType.COUNT, SemanticType.CONTINUOUS_NUMERIC) else "string")
    return ColumnProfile(
        name="y",
        physical_dtype=phys,
        semantic_type=st,
        null_fraction=0.0,
        distinct_estimate=distinct,
        distinct_ratio=ratio,
    )


def test_boolean_target_is_binary():
    inf = infer_task(cp(2, SemanticType.BOOLEAN, "boolean"))
    assert inf.task is Task.BINARY
    assert inf.confidence == 1.0
    assert inf.reasons


def test_numeric_low_cardinality_is_binary():
    inf = infer_task(cp(2, SemanticType.COUNT, "integer"))
    assert inf.task is Task.BINARY


def test_numeric_few_classes_is_multiclass():
    inf = infer_task(cp(5, SemanticType.COUNT, "integer"))
    assert inf.task is Task.MULTICLASS


def test_numeric_broad_support_is_regression():
    inf = infer_task(cp(200, SemanticType.CONTINUOUS_NUMERIC, "float"))
    assert inf.task is Task.REGRESSION


def test_forecasting_takes_precedence_with_time_and_horizon():
    inf = infer_task(cp(200, SemanticType.CONTINUOUS_NUMERIC), time_key="ts", horizon="7d")
    assert inf.task is Task.FORECASTING


def test_explicit_override_wins():
    inf = infer_task(cp(3, SemanticType.LOW_CARDINALITY_CATEGORICAL), explicit_task=Task.REGRESSION)
    assert inf.task is Task.REGRESSION
    assert inf.confidence == 1.0


def test_unsupervised_requires_explicit():
    inf = infer_task(None)
    assert inf.task is Task.AUTO
    assert any("unsupervised" in a for a in inf.ambiguities)
    inf2 = infer_task(None, explicit_task=Task.UNSUPERVISED)
    assert inf2.task is Task.UNSUPERVISED


def test_ranking_requires_group_semantics():
    inf = infer_task(cp(2), explicit_task=Task.RANKING)
    assert any("ranking" in a for a in inf.ambiguities)


def test_positive_class_and_rate():
    values = [0, 0, 0, 1, 1, 1, 1]
    inf = infer_task(cp(2, SemanticType.COUNT, "integer"), target_values=values)
    assert inf.positive_class == 1
    assert abs(inf.positive_rate - 4 / 7) < 1e-9
    assert not inf.is_imbalanced


def test_imbalance_flag():
    values = [0] * 19 + [1]
    inf = infer_task(cp(2, SemanticType.COUNT, "integer"), target_values=values)
    assert inf.is_imbalanced


def test_positive_class_default_from_truthy_strings():
    inf = infer_task(
        cp(2, SemanticType.LOW_CARDINALITY_CATEGORICAL),
        target_values=["no", "no", "yes", "yes"],
    )
    assert inf.positive_class == "yes"


def test_route_metrics_binary():
    m = route_metrics(Task.BINARY)
    names = [t.name for t in m]
    assert "roc_auc" in names and "log_loss" in names
    assert "average_precision" not in names
    m_imb = route_metrics(Task.BINARY, is_imbalanced=True)
    assert "average_precision" in [t.name for t in m_imb]


def test_route_metrics_regression_and_multiclass():
    r = {t.name for t in route_metrics(Task.REGRESSION)}
    assert {"rmse", "mae", "r2"} <= r
    r_nonneg = route_metrics(Task.REGRESSION, non_negative_target=True)
    assert "rmsle" in {t.name for t in r_nonneg}
    m = {t.name for t in route_metrics(Task.MULTICLASS)}
    assert {"macro_f1", "balanced_accuracy", "log_loss"} <= m


def test_make_problem_builds_problem_definition():
    inf = infer_task(cp(2, SemanticType.COUNT), target_values=[0, 0, 1, 1])
    prob = make_problem(
        inf,
        target="y",
        templates=route_metrics(Task.BINARY, is_imbalanced=inf.is_imbalanced),
    )
    assert prob.task is Task.BINARY
    assert prob.target == "y"
    assert prob.positive_class == 1
    assert prob.metrics and all(m.split == "plan" for m in prob.metrics)
    assert prob.ambiguities == []
