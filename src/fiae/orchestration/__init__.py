"""Orchestration subsystem — doc 07 (HPO/Ensembles) + doc 08 (Scheduler)."""
from .trial_runner import TrialRunner, FidelityLadder, PruningPolicy, ResourceBudget
from .executor import SchedulerExecutor, TokenPool, ExecutionToken
from .ensemble import build_ensemble, EnsemblePolicy, EnsembleCandidate, pareto_front
from .hpo import random_search, successive_halving, hyperband, HPOResult, sample_random
from .model_training import TrialSpec, compute_utility, get_fidelity_ladder
from .parallel_executor import ParallelExecutor, InlineExecutor, ResourcePool, EventStream
from .stacking import StackingConfig, StackingResult, check_stacking_guard
