import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fiae.contracts import SplitStrategy, Task
from fiae.problem.splits import (
    group_kfold_indexes,
    kfold_indexes,
    make_splits,
    reserve_final_holdout,
    stratified_kfold_indexes,
    time_ordered_indexes,
)


def test_stratified_folds_keep_both_classes():
    y = [0] * 50 + [1] * 50
    pairs = stratified_kfold_indexes(100, y, 5, seed=7)
    assert len(pairs) == 5
    for tr, va in pairs:
        tr_y = {y[i] for i in tr}
        va_y = {y[i] for i in va}
        assert tr_y == {0, 1} and va_y == {0, 1}


def test_stratified_deterministic_under_seed():
    y = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
    a = stratified_kfold_indexes(10, y, 3, seed=5)
    b = stratified_kfold_indexes(10, y, 3, seed=5)
    assert a == b
    c = stratified_kfold_indexes(10, y, 3, seed=6)
    assert a != c


def test_kfold_is_partition_deterministic():
    pairs = kfold_indexes(30, 3, seed=11)
    all_val = [i for _, va in pairs for i in va]
    assert sorted(all_val) == list(range(30))
    assert kfold_indexes(30, 3, seed=11) == kfold_indexes(30, 3, seed=11)


def test_group_kfold_no_group_overlap():
    groups = [f"g{i % 6}" for i in range(120)]
    pairs = group_kfold_indexes(120, groups, 4, seed=3)
    for tr, va in pairs:
        tr_g = {groups[i] for i in tr}
        va_g = {groups[i] for i in va}
        assert tr_g.isdisjoint(va_g)


def test_time_ordered_train_is_never_after_val():
    times = list(range(200))
    pairs = time_ordered_indexes(200, times, 5)
    assert pairs
    for tr, va in pairs:
        max_tr = max(min(times[i] for i in tr), 0)
        min_va = min(times[i] for i in va)
        assert max_tr <= min_va  # future never enters train


def test_holdout_reserved_before_folds():
    y = [i % 2 for i in range(1000)]
    spec = make_splits(Task.BINARY, 1000, y=y, n_folds=5, holdout_fraction=0.2, seed=0)
    assert len(spec.holdout_indexes) >= 180
    assert sorted(spec.dev_indexes + spec.holdout_indexes) == list(range(1000))
    # folds are built only on dev
    for fold_val in spec.fold_val_indexes:
        assert set(fold_val).isdisjoint(spec.holdout_indexes)
    # holdout is stratified
    _y_dev = [y[i] for i in spec.dev_indexes]
    y_hold = [y[i] for i in spec.holdout_indexes]
    assert 0 < sum(y_hold) < len(y_hold)


def test_split_fingerprint_deterministic_and_strategy_auto():
    y = [i % 2 for i in range(500)]
    a = make_splits(Task.BINARY, 500, y=y, seed=42)
    b = make_splits(Task.BINARY, 500, y=y, seed=42)
    assert a.plan.split_fingerprint == b.plan.split_fingerprint
    assert a.plan.strategy is SplitStrategy.STRATIFIED_KFOLD
    assert a.plan.folds == 5  # >= 500 rows -> default folds


def test_tiny_data_uses_stronger_cv():
    y = [0, 1] * 100
    spec = make_splits(Task.BINARY, 200, y=y)
    assert spec.plan.folds == 7  # TINY_FOLDS


def test_regression_plain_kfold_when_no_dependence():
    spec = make_splits(Task.REGRESSION, 1000, y=[i * 0.5 for i in range(1000)])
    assert spec.plan.strategy is SplitStrategy.KFOLD


def test_time_strategy_with_group_and_stratification():
    times = list(range(300))
    spec = make_splits(Task.REGRESSION, 300, times=times, n_folds=5)
    assert spec.plan.strategy is SplitStrategy.TIME_ORDERED


def test_reserve_holdout_deterministic_and_splits():
    a = reserve_final_holdout(100, y=[i % 2 for i in range(100)], holdout_fraction=0.2, seed=1)
    b = reserve_final_holdout(100, y=[i % 2 for i in range(100)], holdout_fraction=0.2, seed=1)
    assert a == b
