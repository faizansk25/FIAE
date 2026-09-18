"""HPO algorithms (doc 07).

Implements core hyperparameter optimization strategies:
- Random search (baseline, cold-start)
- TPE-like selection (after enough observations)
- Successive halving (resource-efficient pruning)
- Hyperband (multi-bracket exploration)

All algorithms work with the HPO spaces defined in model_registry.py.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable

from ..model_registry import HPDimension


@dataclass
class HPOTrial:
    """Record of one HPO trial."""

    trial_id: str
    params: dict[str, Any]
    score: float = 0.0
    cost_s: float = 0.0
    completed: bool = False


@dataclass
class HPOResult:
    """Result of an HPO run."""

    best_params: dict[str, Any]
    best_score: float
    all_trials: list[HPOTrial]
    algorithm: str = ""


def sample_random(space: list[HPDimension], rng: random.Random) -> dict[str, Any]:
    """Sample one random configuration from an HPO space."""
    params = {}
    for dim in space:
        if dim.dtype == "float":
            if dim.log_scale:
                log_low = math.log(max(dim.low, 1e-10))
                log_high = math.log(dim.high)
                params[dim.name] = math.exp(rng.uniform(log_low, log_high))
            else:
                params[dim.name] = rng.uniform(dim.low, dim.high)
        elif dim.dtype == "int":
            params[dim.name] = rng.randint(int(dim.low), int(dim.high))
        elif dim.dtype == "categorical":
            params[dim.name] = rng.choice(dim.choices)
    return params


def random_search(
    space: list[HPDimension],
    objective: Callable[[dict], float],
    n_trials: int = 20,
    seed: int = 42,
    maximize: bool = False,
) -> HPOResult:
    """Random search over HPO space (doc 07).

    Baseline and cold-start method.  Samples n_trials random configurations
    and returns the best.
    """
    rng = random.Random(seed)
    trials = []

    for i in range(n_trials):
        params = sample_random(space, rng)
        try:
            score = objective(params)
        except Exception:
            score = float("-inf") if maximize else float("inf")
        trials.append(HPOTrial(
            trial_id=f"rs_{i}", params=params,
            score=score, completed=True,
        ))

    if maximize:
        best = max(trials, key=lambda t: t.score)
    else:
        best = min(trials, key=lambda t: t.score)

    return HPOResult(
        best_params=best.params, best_score=best.score,
        all_trials=trials, algorithm="random_search",
    )


def tpe_select(
    trials: list[HPOTrial],
    space: list[HPDimension],
    rng: random.Random,
    n_candidates: int = 24,
    gamma: float = 0.25,
) -> dict[str, Any]:
    """TPE-like selection: split trials into good/bad, sample from good (doc 07).

    Simplified TPE: sorts by score, takes top gamma fraction as "good",
    samples candidates from around good points.
    """
    completed = [t for t in trials if t.completed]
    if len(completed) < 5:
        return sample_random(space, rng)

    sorted_trials = sorted(completed, key=lambda t: t.score)
    n_good = max(1, int(len(sorted_trials) * gamma))
    good = sorted_trials[:n_good]

    # Sample candidates by perturbing good trial params
    best = rng.choice(good)
    params = dict(best.params)

    for dim in space:
        if rng.random() < 0.3:  # 30% chance to resample each dimension
            if dim.dtype == "float":
                if dim.log_scale:
                    log_low = math.log(max(dim.low, 1e-10))
                    log_high = math.log(dim.high)
                    val = math.exp(rng.uniform(log_low, log_high))
                else:
                    val = rng.uniform(dim.low, dim.high)
                params[dim.name] = val
            elif dim.dtype == "int":
                params[dim.name] = rng.randint(int(dim.low), int(dim.high))
            elif dim.dtype == "categorical":
                params[dim.name] = rng.choice(dim.choices)

    return params


def successive_halving(
    space: list[HPDimension],
    objective: Callable[[dict], float],
    n_initial: int = 27,
    reduction_factor: int = 3,
    seed: int = 42,
    maximize: bool = False,
) -> HPOResult:
    """Successive halving: allocate resources progressively, prune worst (doc 07).

    Starts with n_initial cheap trials, keeps top 1/reduction_factor,
    increases budget, repeat until one remains.
    """
    rng = random.Random(seed)
    all_trials = []

    # Generate initial configs
    configs = [sample_random(space, rng) for _ in range(n_initial)]
    stage = 0

    while len(configs) > 1:
        # Evaluate all configs at current budget
        stage_trials = []
        for i, params in enumerate(configs):
            try:
                score = objective(params)
            except Exception:
                score = float("-inf") if maximize else float("inf")
            trial = HPOTrial(
                trial_id=f"sh_s{stage}_{i}", params=params,
                score=score, completed=True,
            )
            stage_trials.append(trial)
            all_trials.append(trial)

        # Keep top fraction
        if maximize:
            stage_trials.sort(key=lambda t: t.score, reverse=True)
        else:
            stage_trials.sort(key=lambda t: t.score)

        n_keep = max(1, len(stage_trials) // reduction_factor)
        configs = [t.params for t in stage_trials[:n_keep]]
        stage += 1

    # Final evaluation of winner
    if configs:
        try:
            final_score = objective(configs[0])
        except Exception:
            final_score = float("-inf") if maximize else float("inf")
        all_trials.append(HPOTrial(
            trial_id="sh_final", params=configs[0],
            score=final_score, completed=True,
        ))

    best = max(all_trials, key=lambda t: t.score) if maximize else min(all_trials, key=lambda t: t.score)
    return HPOResult(
        best_params=best.params, best_score=best.score,
        all_trials=all_trials, algorithm="successive_halving",
    )


def hyperband(
    space: list[HPDimension],
    objective: Callable[[dict], float],
    max_resource: int = 81,
    eta: int = 3,
    seed: int = 42,
    maximize: bool = False,
) -> HPOResult:
    """Hyperband: multiple brackets of successive halving (doc 07).

    Explores the trade-off between many cheap trials and few expensive ones
    by running multiple brackets with different starting budgets.
    """
    rng = random.Random(seed)
    all_trials = []

    # Standard Hyperband schedule (Li et al. 2016): bracket s starts with
    # n = ceil(B / r_max / (s+1) * eta**s) configs at resource r = r_max*eta**-s
    # and promotes the top 1/eta through s+1 stages. The stdlib objective
    # callable is resource-agnostic, so stages differ in config count only;
    # per-stage resources are r_stage = max_resource * eta**-bracket_offset.
    s_max = int(math.log(max_resource) / math.log(eta))
    B = (s_max + 1) * max_resource
    for s in range(s_max, -1, -1):
        n = math.ceil(B / max_resource / (s + 1) * eta ** s)

        configs = [sample_random(space, rng) for _ in range(n)]

        for bracket_stage in range(s + 1):
            n_i = n * eta ** (-bracket_stage)
            stage_trials = []
            for i, params in enumerate(configs):
                try:
                    score = objective(params)
                except Exception:
                    score = float("-inf") if maximize else float("inf")
                trial = HPOTrial(
                    trial_id=f"hb_b{s}_s{bracket_stage}_{i}",
                    params=params, score=score, completed=True,
                )
                stage_trials.append(trial)
                all_trials.append(trial)

            # Keep top 1/eta per the schedule (never below 1)
            if maximize:
                stage_trials.sort(key=lambda t: t.score, reverse=True)
            else:
                stage_trials.sort(key=lambda t: t.score)
            n_keep = max(1, int(n_i / eta))
            configs = [t.params for t in stage_trials[:n_keep]]

    best = max(all_trials, key=lambda t: t.score) if maximize else min(all_trials, key=lambda t: t.score)
    return HPOResult(
        best_params=best.params, best_score=best.score,
        all_trials=all_trials, algorithm="hyperband",
    )
