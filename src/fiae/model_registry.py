"""Model registry and routing table (doc 07).

The model registry defines available model families, their capabilities,
HPO search spaces, and routing priorities.  Core depends on nothing beyond
stdlib; model families that require external packages are marked as optional.

Normative source: doc 07 section "Model routing table".
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional


class ModelFamily(str, enum.Enum):
    """Model family identifiers matching doc 07 routing table."""

    LINEAR = "linear"
    LOGISTIC = "logistic"
    SGD_LINEAR = "sgd_linear"
    NAIVE_BAYES = "naive_bayes"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    EXTRA_TREES = "extra_trees"
    HIST_GRADIENT_BOOSTING = "hist_gradient_boosting"
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    CATBOOST = "catboost"
    MLP = "mlp"


class ModelBackend(str, enum.Enum):
    """Backend implementations."""

    SKLEARN = "sklearn"
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    CATBOOST = "catboost"
    PYTORCH = "pytorch"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# HPO search spaces (doc 07 section "HPO spaces")
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HPDimension:
    """One dimension in an HPO search space."""

    name: str
    dtype: str  # "int" | "float" | "categorical"
    low: Optional[float] = None
    high: Optional[float] = None
    log_scale: bool = False
    choices: tuple = ()
    default: Any = None


# Standard HPO spaces per model family (doc 07)
LINEAR_SPACE = [
    HPDimension("C", "float", low=1e-4, high=1e4, log_scale=True, default=1.0),
    HPDimension("penalty", "categorical", choices=("l2", "l1", "elasticnet"), default="l2"),
    HPDimension("max_iter", "int", low=100, high=2000, default=1000),
]

FOREST_SPACE = [
    HPDimension("n_estimators", "int", low=50, high=500, default=100),
    HPDimension("max_depth", "int", low=3, high=30, default=None),
    HPDimension("min_samples_leaf", "int", low=1, high=20, default=1),
    HPDimension("max_features", "categorical", choices=("sqrt", "log2", 0.3, 0.5), default="sqrt"),
]

GRADIENT_BOOSTING_SPACE = [
    HPDimension("n_estimators", "int", low=50, high=1000, default=100),
    HPDimension("learning_rate", "float", low=0.01, high=0.3, log_scale=True, default=0.1),
    HPDimension("max_depth", "int", low=3, high=15, default=6),
    HPDimension("min_samples_leaf", "int", low=1, high=30, default=1),
    HPDimension("l2_regularization", "float", low=0.0, high=10.0, default=0.0),
]

MLP_SPACE = [
    HPDimension("hidden_layer_sizes", "categorical",
                choices=((64,), (128,), (64, 32), (128, 64)), default=(64,)),
    HPDimension("learning_rate_init", "float", low=1e-4, high=1e-1, log_scale=True, default=1e-3),
    HPDimension("alpha", "float", low=1e-5, high=1e-1, log_scale=True, default=1e-4),
    HPDimension("max_iter", "int", low=100, high=1000, default=200),
]


# ---------------------------------------------------------------------------
# Model family catalog entry
# ---------------------------------------------------------------------------
@dataclass
class ModelFamilySpec:
    """Full specification of one model family (doc 07 routing table)."""

    family: ModelFamily
    backend: ModelBackend
    main_role: str
    use_when: str
    hpo_space: list[HPDimension] = field(default_factory=list)
    supports_missing: bool = False
    supports_categorical: bool = False
    is_ensemble: bool = False
    is_optional: bool = False  # requires optional dependency
    priority: int = 50  # routing priority (higher = prefer)


# ---------------------------------------------------------------------------
# Default routing table (doc 07)
# ---------------------------------------------------------------------------
ROUTING_TABLE: list[ModelFamilySpec] = [
    ModelFamilySpec(
        family=ModelFamily.LINEAR, backend=ModelBackend.SKLEARN,
        main_role="cheap baseline, sparse/high-dimensional",
        use_when="classification/regression, fast first pass",
        hpo_space=LINEAR_SPACE, priority=40,
    ),
    ModelFamilySpec(
        family=ModelFamily.LOGISTIC, backend=ModelBackend.SKLEARN,
        main_role="cheap classification baseline",
        use_when="binary/multiclass classification",
        hpo_space=LINEAR_SPACE, priority=41,
    ),
    ModelFamilySpec(
        family=ModelFamily.SGD_LINEAR, backend=ModelBackend.SKLEARN,
        main_role="very large/sparse data",
        use_when="streamable cheap linear",
        priority=35,
    ),
    ModelFamilySpec(
        family=ModelFamily.NAIVE_BAYES, backend=ModelBackend.SKLEARN,
        main_role="probabilistic text/count baseline",
        use_when="selected text/count classification",
        priority=30,
    ),
    ModelFamilySpec(
        family=ModelFamily.DECISION_TREE, backend=ModelBackend.SKLEARN,
        main_role="nonlinear diagnostic",
        use_when="tabular, interpretability needed",
        priority=25,
    ),
    ModelFamilySpec(
        family=ModelFamily.RANDOM_FOREST, backend=ModelBackend.SKLEARN,
        main_role="robust/diverse baseline",
        use_when="tabular, robust default",
        hpo_space=FOREST_SPACE, is_ensemble=True, priority=55,
    ),
    ModelFamilySpec(
        family=ModelFamily.EXTRA_TREES, backend=ModelBackend.SKLEARN,
        main_role="diversity source for ensembles",
        use_when="tabular, diversity needed",
        hpo_space=FOREST_SPACE, is_ensemble=True, priority=45,
    ),
    ModelFamilySpec(
        family=ModelFamily.HIST_GRADIENT_BOOSTING, backend=ModelBackend.SKLEARN,
        main_role="strong low-dependency baseline",
        use_when="tabular, best sklearn GBM",
        hpo_space=GRADIENT_BOOSTING_SPACE, priority=60,
    ),
    ModelFamilySpec(
        family=ModelFamily.LIGHTGBM, backend=ModelBackend.LIGHTGBM,
        main_role="strong fast GBM",
        use_when="tabular, speed + quality",
        hpo_space=GRADIENT_BOOSTING_SPACE,
        is_optional=True, supports_categorical=True, priority=65,
    ),
    ModelFamilySpec(
        family=ModelFamily.XGBOOST, backend=ModelBackend.XGBOOST,
        main_role="strong GBM challenger",
        use_when="tabular, quality challenger",
        hpo_space=GRADIENT_BOOSTING_SPACE,
        is_optional=True, priority=63,
    ),
    ModelFamilySpec(
        family=ModelFamily.CATBOOST, backend=ModelBackend.CATBOOST,
        main_role="categorical-aware challenger",
        use_when="tabular with many categoricals",
        hpo_space=GRADIENT_BOOSTING_SPACE,
        is_optional=True, supports_categorical=True, supports_missing=True, priority=62,
    ),
    ModelFamilySpec(
        family=ModelFamily.MLP, backend=ModelBackend.PYTORCH,
        main_role="conditional deep model",
        use_when="regime/budget justifies, non-tabular",
        hpo_space=MLP_SPACE,
        is_optional=True, priority=20,
    ),
]


# ---------------------------------------------------------------------------
# Model registry API
# ---------------------------------------------------------------------------
def get_model_spec(family: ModelFamily) -> Optional[ModelFamilySpec]:
    """Look up a model family by enum value."""
    for spec in ROUTING_TABLE:
        if spec.family == family:
            return spec
    return None


def available_families(include_optional: bool = False) -> list[ModelFamilySpec]:
    """Return model families available in the current environment.

    If ``include_optional`` is False (default), only families whose
    backend package is importable are returned.
    """
    result = []
    for spec in ROUTING_TABLE:
        # Skip optional families whose backend is not importable.
        if spec.is_optional and not include_optional \
                and not _backend_available(spec.backend):
            continue
        result.append(spec)
    return result


def _backend_available(backend: ModelBackend) -> bool:
    """Check if a model backend is importable."""
    import importlib
    module_map = {
        ModelBackend.SKLEARN: "sklearn",
        ModelBackend.LIGHTGBM: "lightgbm",
        ModelBackend.XGBOOST: "xgboost",
        ModelBackend.CATBOOST: "catboost",
        ModelBackend.PYTORCH: "torch",
    }
    mod = module_map.get(backend)
    if mod is None:
        return False
    try:
        importlib.import_module(mod)
        return True
    except ImportError:
        return False


def route_model(
    task: str,
    n_rows: int,
    n_features: int,
    has_categoricals: bool = False,
    budget_seconds: float = 300.0,
) -> list[ModelFamilySpec]:
    """Route to appropriate model families based on data characteristics.

    Returns a ranked list of candidates (best first) from the routing table.
    This is the core of doc 07's model routing.
    """
    candidates = available_families()

    # Filter by task compatibility
    task_lower = task.lower()
    if "binary" in task_lower or "multiclass" in task_lower:
        # Classification families
        candidates = [
            s for s in candidates
            if s.family not in (ModelFamily.LINEAR,)  # prefer logistic for classification
        ]
    elif "regression" in task_lower:
        candidates = [
            s for s in candidates
            if s.family not in (ModelFamily.LOGISTIC, ModelFamily.NAIVE_BAYES)
        ]

    # Prefer categorical-aware models if categoricals present
    if has_categoricals:
        for s in candidates:
            if s.supports_categorical:
                s.priority += 10  # boost priority

    # Small data: prefer simpler models
    if n_rows < 500:
        for s in candidates:
            if s.family in (ModelFamily.LINEAR, ModelFamily.LOGISTIC,
                            ModelFamily.NAIVE_BAYES, ModelFamily.RANDOM_FOREST):
                s.priority += 5

    # High-dimensional: prefer linear/sparse
    if n_features > n_rows:
        for s in candidates:
            if s.family in (ModelFamily.LINEAR, ModelFamily.SGD_LINEAR):
                s.priority += 10

    # Sort by priority descending
    candidates.sort(key=lambda s: s.priority, reverse=True)
    return candidates


@dataclass
class ModelAssignment:
    """Result of model routing: which families to try and in what order."""

    families: list[ModelFamilySpec]
    reason: str
    data_profile: dict[str, Any] = field(default_factory=dict)


def assign_models(
    task: str, n_rows: int, n_features: int,
    has_categoricals: bool = False,
) -> ModelAssignment:
    """High-level model assignment: pick the best model families for a dataset."""
    families = route_model(task, n_rows, n_features, has_categoricals)
    reasons = []
    if has_categoricals:
        reasons.append("categorical features detected")
    if n_features > n_rows:
        reasons.append(f"high-dimensional ({n_features} features > {n_rows} rows)")
    if n_rows < 500:
        reasons.append(f"small dataset ({n_rows} rows)")

    return ModelAssignment(
        families=families,
        reason="; ".join(reasons) if reasons else "default routing",
        data_profile={
            "n_rows": n_rows,
            "n_features": n_features,
            "has_categoricals": has_categoricals,
            "task": task,
        },
    )
