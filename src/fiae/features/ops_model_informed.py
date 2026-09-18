"""Model-informed feature operators (doc 05 entries 94-95).

These operators use model predictions/residuals or tree partitions as
feature representations.  They are L2 (target-aware) and require a
fitted model on OOF predictions only.

- residual_interaction_proposal (94): propose interactions where baseline
  errors remain structured
- tree_leaf_oof (95): use tree partitions as representation
"""

from __future__ import annotations

import contextlib
import math
from typing import Any

from ..contracts import FitScope, LeakageClass, TargetPermission
from .registry import FeatureOperator, register


# ---------------------------------------------------------------------------
# Optional sklearn import
# ---------------------------------------------------------------------------
_SKLEARN_AVAILABLE = False
try:
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    _SKLEARN_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    DecisionTreeClassifier = None  # type: ignore


def _require_sklearn():
    if not _SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for model-informed operators. "
            "Install with: pip install fiae[tier1]"
        )


def _to_matrix(columns: list[list], n_rows: int) -> Any:
    """Convert columnar lists to a 2D numpy array, treating None as NaN."""
    _require_sklearn()
    matrix = np.full((n_rows, len(columns)), np.nan)
    for j, col in enumerate(columns):
        for i in range(min(n_rows, len(col))):
            v = col[i]
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                with contextlib.suppress(TypeError, ValueError):
                    matrix[i, j] = float(v)
    return matrix


# ---------------------------------------------------------------------------
# residual_interaction_proposal (entry 94)
# ---------------------------------------------------------------------------
def tf_residual_proposal(
    columns: list[list], residuals: list[float],
    top_k: int = 5, **_: Any
) -> list:
    """Analyze OOF residuals and propose feature interactions.

    This is a *proposal* operator: it doesn't produce a final feature but
    generates ranked hypotheses.  The caller evaluates each via the funnel.

    Returns a list of dicts with keys: column_idx, direction, strength.
    """
    n_rows = max(len(c) for c in columns) if columns else 0
    if not residuals or n_rows == 0:
        return []

    X = _to_matrix(columns, n_rows)
    y_res = np.array(residuals[:n_rows], dtype=float)

    proposals = []
    for j in range(X.shape[1]):
        col = X[:, j]
        valid = ~np.isnan(col)
        if valid.sum() < 10:
            continue
        x_valid = col[valid]
        r_valid = y_res[valid]

        # Correlation of feature with absolute residual (structured error)
        abs_r = np.abs(r_valid)
        mean_x = x_valid.mean()
        mean_r = abs_r.mean()
        cov = ((x_valid - mean_x) * (abs_r - mean_r)).mean()
        std_x = x_valid.std()
        std_r = abs_r.std()
        if std_x > 0 and std_r > 0:
            corr = cov / (std_x * std_r)
        else:
            corr = 0.0

        # Check for non-linear residual pattern via signed correlation
        signed_corr = 0.0
        if std_x > 0 and r_valid.std() > 0:
            signed_corr = (
                ((x_valid - mean_x) * r_valid).mean() / (std_x * r_valid.std())
            )

        strength = abs(corr) + abs(signed_corr) * 0.5
        if strength > 0.01:
            direction = "nonlinear" if abs(signed_corr) > 0.1 else "magnitude"
            proposals.append({
                "column_idx": j,
                "direction": direction,
                "strength": float(strength),
                "signed_corr": float(signed_corr),
            })

    # Sort by strength descending
    proposals.sort(key=lambda p: p["strength"], reverse=True)
    return proposals[:top_k]


# ---------------------------------------------------------------------------
# tree_leaf_oof (entry 95)
# ---------------------------------------------------------------------------
def tf_tree_leaf_oof_fit(
    columns: list[list], target: list,
    max_depth: int = 4, **_: Any
) -> dict:
    """Fit a decision tree on training data for OOF leaf encoding.

    Uses a small decision tree to partition the feature space into leaves,
    which serve as a categorical representation.
    """
    _require_sklearn()
    n_rows = min(len(target), max(len(c) for c in columns) if columns else 0)
    X = _to_matrix(columns, n_rows)
    y = np.array([float(t) if t is not None else 0.0 for t in target[:n_rows]])

    tree = DecisionTreeClassifier(
        max_depth=max_depth, random_state=42
    )
    tree.fit(X, y)

    # Extract leaf IDs for training data
    leaves = tree.apply(X)
    n_leaves = len(set(leaves))

    return {"tree": tree, "n_leaves": n_leaves, "max_depth": max_depth}


def tf_tree_leaf_oof_transform(
    columns: list[list], state: dict, **_: Any
) -> list:
    """Transform: map rows to tree leaf IDs."""
    _require_sklearn()
    tree = state.get("tree")
    if tree is None:
        return [0] * max(len(c) for c in columns) if columns else []
    n_rows = max(len(c) for c in columns)
    X = _to_matrix(columns, n_rows)
    leaves = tree.apply(X)
    return leaves.tolist()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
if _SKLEARN_AVAILABLE:

    register(
        FeatureOperator(
            name="residual_interaction_proposal",
            family="model-informed",
            arity="unary",
            input_types=("continuous_numeric",),
            output_type="numeric",
            purpose="propose interactions where baseline errors remain structured",
            preconditions=("valid OOF residuals only",),
            fit_scope=FitScope.DEVELOPMENT,
            null_policy="child semantics",
            leakage_class=LeakageClass.L2,
            target_permission=TargetPermission.P1_CROSSFIT,
            cost_shape="variable",
            generation_trigger="stable residual slices",
            rejection_conditions=("uses final holdout", "in-sample residuals"),
            validation="nested development evaluation",
            inference_requirement="store proposal evidence",
            transform=tf_residual_proposal,
            mandatory_tests=("OOF-only",),
        )
    )

    register(
        FeatureOperator(
            name="tree_leaf_oof",
            family="model-informed",
            arity="unary",
            input_types=("continuous_numeric",),
            output_type="categorical",
            purpose="use tree partitions as categorical representation",
            preconditions=("tree candidate exists",),
            fit_scope=FitScope.TRAINING_FOLD,
            null_policy="n/a",
            leakage_class=LeakageClass.L2,
            target_permission=TargetPermission.P1_CROSSFIT,
            cost_shape="high",
            generation_trigger="stacking/representation evidence",
            rejection_conditions=("overfit", "latency", "memory"),
            validation="OOF only for downstream training",
            inference_requirement="store final tree",
            transform=tf_tree_leaf_oof_transform,
            mandatory_tests=("in-sample prohibition",),
        )
    )
