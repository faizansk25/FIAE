"""Validation/split planner and deterministic fold construction (doc 03).

Rules (doc 03, Split planner / Final holdout / Progressive validation):
- independent classification -> stratified K-fold
- independent regression -> K-fold with a deterministic seed
- group dependence -> GroupKFold so one entity does not leak across train/validation
- time dependence -> ordered backtests/time splits; random row shuffling is invalid
- tiny data -> stronger CV and a smaller candidate search
- final holdout is reserved BEFORE feature/model search and never used for selection
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Optional

from ..contracts import SplitStrategy, Task, ValidationPlan
from ..ids import split_fingerprint

TINY_N = 500
TINY_FOLDS = 7
DEFAULT_FOLDS = 5
DEFAULT_NESTED_POLICY = "probe_3fold-screen_5fold-final_nested-only-finalists"


@dataclass
class SplitSpec:
    """Materialized fold indexes on global row positions (doc 03/13)."""

    plan: ValidationPlan
    dev_indexes: list[int]
    holdout_indexes: list[int]
    fold_train_indexes: list[list[int]]
    fold_val_indexes: list[list[int]]


# --------------------------------------------------------------------------
# Deterministic fold index builders (seeded random for reproducibility)
# --------------------------------------------------------------------------
def stratified_kfold_indexes(
    n: int, y: list[Any], k: int, seed: int
) -> list[tuple[list[int], list[int]]]:
    """Seeded stratified K-fold: every fold keeps class proportions."""
    rng = random.Random(seed)
    k = max(2, min(int(k), n))
    by_class: dict[str, list[int]] = {}
    for i, yi in enumerate(y):
        by_class.setdefault(str(yi), []).append(i)
    fold_id = [0] * n
    for idxs in by_class.values():
        rng.shuffle(idxs)
        for slot, i in enumerate(idxs):
            fold_id[i] = slot % k
    out = []
    for f in range(k):
        val = [i for i in range(n) if fold_id[i] == f]
        tr = [i for i in range(n) if fold_id[i] != f]
        out.append((tr, val))
    return out


def kfold_indexes(n: int, k: int, seed: int) -> list[tuple[list[int], list[int]]]:
    """Seeded plain K-fold for independent regression.

    Rows are shuffled under ``seed`` and then dealt into k folds as evenly
    as possible; every fold's validation set is non-empty (n >= k), so the
    effective fold count always matches the requested one.
    """
    rng = random.Random(seed)
    if n < 2:
        # Degenerate: two non-empty folds are impossible; return the single
        # partition (empty train) and let consumers skip it.
        return [([], list(range(n)))] if n else []
    k = max(2, min(int(k), n))
    idx = list(range(n))
    rng.shuffle(idx)
    base = n // k
    rem = n % k
    out = []
    start = 0
    for f in range(k):
        size = base + (1 if f < rem else 0)
        val = idx[start : start + size]
        start += size
        vset = set(val)
        tr = [i for i in range(n) if i not in vset]
        out.append((tr, val))
    return out


def group_kfold_indexes(
    n: int, group_ids: list[Any], k: int, seed: int
) -> list[tuple[list[int], list[int]]]:
    """GroupKFold: no entity appears in both train and validation of a fold."""
    rng = random.Random(seed)
    k = max(2, min(int(k), n))
    by_group: dict[Any, list[int]] = {}
    for i, g in enumerate(group_ids):
        by_group.setdefault(g, []).append(i)
    uniq = sorted(by_group)
    rng.shuffle(uniq)
    out = []
    for f in range(k):
        val = []
        tr = []
        for s, g in enumerate(uniq):
            if s % k == f:
                val.extend(by_group[g])
            else:
                tr.extend(by_group[g])
        out.append((tr, val))
    return out


def time_ordered_indexes(
    n: int, times: list[Any], k: int
) -> list[tuple[list[int], list[int]]]:
    """Expanding-window chronological backtests (future never enters train).

    The sorted order is split into k contiguous chrono slices; fold f covers
    train = all slices before slice f, val = slice f. Because the order is
    strictly chronological, max(train time) <= min(val time) holds per fold.
    """
    k = max(2, min(int(k), n))
    order = sorted(range(n), key=lambda i: times[i])
    borders = [round(n * f / k) for f in range(k + 1)]
    out = []
    for f in range(1, k):  # fold 0 would have an empty train; skip it
        tr = order[: borders[f]]
        val = order[borders[f] : borders[f + 1]]
        if tr and val:
            out.append((tr, val))
    if not out:
        mid = n // 2
        return [(order[:mid], order[mid:])]
    return out


def reserve_final_holdout(
    n: int,
    *,
    y: Optional[list[Any]] = None,
    group_ids: Optional[list[Any]] = None,
    holdout_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    """Reserve a final untouched holdout BEFORE any fitted preprocessing (doc 03)."""
    rng = random.Random(seed)
    n_hold = max(1, round(n * holdout_fraction))
    if n_hold >= n:
        n_hold = max(1, n - 1)
    if group_ids is not None:
        by_group: dict[Any, list[int]] = {}
        for i, g in enumerate(group_ids):
            by_group.setdefault(g, []).append(i)
        uniq = list(by_group)
        rng.shuffle(uniq)
        hold = []
        taken = 0
        for g in uniq:
            if taken >= n_hold:
                break
            hold.extend(by_group[g])
            taken += len(by_group[g])
        holdset = set(hold)
        dev = [i for i in range(n) if i not in holdset]
        return sorted(dev), sorted(hold)
    if y is not None:
        by_class: dict[str, list[int]] = {}
        for i, yi in enumerate(y):
            by_class.setdefault(str(yi), []).append(i)
        hold = []
        for idxs in by_class.values():
            h = max(1, round(len(idxs) * holdout_fraction))
            hold.extend(rng.sample(idxs, min(h, len(idxs))))
        holdset = set(hold)
        dev = [i for i in range(n) if i not in holdset]
        return sorted(dev), sorted(hold)
    hold = rng.sample(range(n), n_hold)
    holdset = set(hold)
    dev = [i for i in range(n) if i not in holdset]
    return sorted(dev), sorted(hold)
def make_splits(
    problem: Task,
    n: int,
    *,
    y: Optional[list[Any]] = None,
    group_ids: Optional[list[Any]] = None,
    times: Optional[list[Any]] = None,
    n_folds: Optional[int] = None,
    holdout_fraction: float = 0.2,
    seed: int = 0,
    strategy: Optional[SplitStrategy] = None,
    nested_policy: str = DEFAULT_NESTED_POLICY,
) -> SplitSpec:
    """Select strategy by dependence structure and materialize folds (doc 03)."""
    if strategy is None:
        if times is not None:
            strategy = SplitStrategy.TIME_ORDERED
        elif group_ids is not None:
            strategy = SplitStrategy.GROUP_KFOLD
        elif y is not None and problem in (Task.BINARY, Task.MULTICLASS):
            strategy = SplitStrategy.STRATIFIED_KFOLD
        else:
            strategy = SplitStrategy.KFOLD

    effective_folds = n_folds or (TINY_FOLDS if n < TINY_N else DEFAULT_FOLDS)

    if strategy is SplitStrategy.TIME_ORDERED and times is None:
        raise ValueError("TIME_ORDERED strategy requires a time array")

    fp = split_fingerprint(
        {
            "strategy": strategy.value,
            "n_folds": effective_folds,
            "seed": seed,
            "holdout_fraction": holdout_fraction,
            "nested_policy": nested_policy,
        }
    )
    plan = ValidationPlan(
        strategy=strategy,
        folds=effective_folds,
        seed=seed,
        split_fingerprint=fp,
        final_holdout=True,
        nested_policy=nested_policy,
    )

    dev, hold = reserve_final_holdout(
        n, y=y, group_ids=group_ids, holdout_fraction=holdout_fraction, seed=seed
    )
    y_dev = None if y is None else [y[i] for i in dev]
    g_dev = None if group_ids is None else [group_ids[i] for i in dev]
    t_dev = None if times is None else [times[i] for i in dev]

    if strategy is SplitStrategy.STRATIFIED_KFOLD and y_dev is not None:
        pairs = stratified_kfold_indexes(len(dev), y_dev, effective_folds, seed)
    elif strategy is SplitStrategy.GROUP_KFOLD and g_dev is not None:
        pairs = group_kfold_indexes(len(dev), g_dev, effective_folds, seed)
    elif strategy is SplitStrategy.TIME_ORDERED:
        pairs = time_ordered_indexes(len(dev), t_dev, effective_folds)
    else:
        pairs = kfold_indexes(len(dev), effective_folds, seed)

    fold_train = [[dev[p] for p in tr] for tr, _ in pairs]
    fold_val = [[dev[p] for p in va] for _, va in pairs]
    return SplitSpec(
        plan=plan,
        dev_indexes=dev,
        holdout_indexes=hold,
        fold_train_indexes=fold_train,
        fold_val_indexes=fold_val,
    )
