"""Dimensionality reduction and clustering operators (doc 05 entries 89-93).

These operators require scikit-learn (tier1 optional dependency).
They follow the same null-preserving, fail-closed semantics.
Import fails gracefully if sklearn is not installed.
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
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.cluster import KMeans
    _SKLEARN_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    PCA = None  # type: ignore
    TruncatedSVD = None  # type: ignore
    KMeans = None  # type: ignore


def _require_sklearn():
    if not _SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for this operator. "
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
# text_svd (entry 89): project TF-IDF sparse matrix to k components
# ---------------------------------------------------------------------------
def tf_text_svd_fit(
    values: list, n_components: int = 5, max_features: int = 100, **_: Any
) -> dict:
    """Fit truncated SVD on TF-IDF vectors from training text.

    values: list of strings (raw text)
    Returns state with fitted SVD model and vocabulary.
    """
    _require_sklearn()

    # Build simple word vocabulary
    from collections import Counter
    import re

    doc_freq: Counter = Counter()
    all_tokens: list[list[str]] = []
    for v in values:
        if v is None:
            all_tokens.append([])
            continue
        toks = list(set(re.findall(r"\w+", str(v).lower())))
        all_tokens.append(toks)
        doc_freq.update(toks)

    # Select top features
    top_tokens = [t for t, _ in doc_freq.most_common(max_features)]
    vocab = {t: i for i, t in enumerate(top_tokens)}
    n_terms = len(vocab)
    n_docs = len(values)

    # Build dense matrix (TF counts)
    X = np.zeros((n_docs, n_terms))
    for i, toks in enumerate(all_tokens):
        for tok in toks:
            if tok in vocab:
                X[i, vocab[tok]] += 1.0

    k = min(n_components, min(n_docs, n_terms) - 1, n_terms)
    k = max(k, 1)
    svd = TruncatedSVD(n_components=k, random_state=42)
    # Zero-variance input (e.g. identical docs) makes sklearn divide by zero
    # when computing explained_variance_ratio_; the projections are still valid.
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", RuntimeWarning)
        svd.fit(X)

    return {"svd": svd, "vocab": vocab, "n_components": k}


def tf_text_svd_transform(values: list, state: dict, **_: Any) -> list:
    """Transform text to SVD-reduced vectors."""
    _require_sklearn()
    import re
    svd = state.get("svd")
    vocab = state.get("vocab", {})
    k = state.get("n_components", 1)
    if svd is None:
        return [[0.0] * k for _ in values]

    out: list = []
    for v in values:
        if v is None:
            out.append([0.0] * k)
            continue
        toks = re.findall(r"\w+", str(v).lower())
        vec = np.zeros(len(vocab))
        for tok in toks:
            if tok in vocab:
                vec[vocab[tok]] += 1.0
        proj = svd.transform(vec.reshape(1, -1))
        out.append(proj[0].tolist())
    return out


# ---------------------------------------------------------------------------
# pca (entry 90): compress correlated numeric dimensions
# ---------------------------------------------------------------------------
def tf_pca_fit(
    columns: list[list], n_components: int = 5, **_: Any
) -> dict:
    """Fit PCA on training numeric columns."""
    _require_sklearn()
    n_rows = max(len(c) for c in columns) if columns else 0
    X = _to_matrix(columns, n_rows)
    n_comp = min(n_components, X.shape[1], X.shape[0] - 1, X.shape[1])
    n_comp = max(n_comp, 1)
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(X)
    return {"pca": pca, "n_components": n_comp}


def tf_pca_transform(columns: list[list], state: dict, **_: Any) -> list:
    """Transform numeric columns using fitted PCA."""
    _require_sklearn()
    pca = state.get("pca")
    k = state.get("n_components", 1)
    if pca is None:
        return [[0.0] * k for _ in range(max(len(c) for c in columns) if columns else 0)]
    n_rows = max(len(c) for c in columns)
    X = _to_matrix(columns, n_rows)
    proj = pca.transform(X)
    return [row.tolist() for row in proj]


# ---------------------------------------------------------------------------
# truncated_svd (entry 91): project sparse numeric to dense
# ---------------------------------------------------------------------------
def tf_truncated_svd_fit(
    columns: list[list], n_components: int = 5, **_: Any
) -> dict:
    """Fit truncated SVD on sparse-ish numeric matrix."""
    _require_sklearn()
    n_rows = max(len(c) for c in columns) if columns else 0
    X = _to_matrix(columns, n_rows)
    n_comp = min(n_components, X.shape[1], X.shape[0] - 1, X.shape[1])
    n_comp = max(n_comp, 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    svd.fit(X)
    return {"svd": svd, "n_components": n_comp}


def tf_truncated_svd_transform(columns: list[list], state: dict, **_: Any) -> list:
    """Transform using fitted truncated SVD."""
    _require_sklearn()
    svd = state.get("svd")
    k = state.get("n_components", 1)
    if svd is None:
        return [[0.0] * k for _ in range(max(len(c) for c in columns) if columns else 0)]
    n_rows = max(len(c) for c in columns)
    X = _to_matrix(columns, n_rows)
    proj = svd.transform(X)
    return [row.tolist() for row in proj]


# ---------------------------------------------------------------------------
# kmeans_label (entry 92): nearest center cluster ID
# ---------------------------------------------------------------------------
def tf_kmeans_fit(
    columns: list[list], n_clusters: int = 5, **_: Any
) -> dict:
    """Fit KMeans on training numeric columns."""
    _require_sklearn()
    n_rows = max(len(c) for c in columns) if columns else 0
    X = _to_matrix(columns, n_rows)
    k = min(n_clusters, X.shape[0], X.shape[1])
    k = max(k, 2)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    return {"kmeans": km, "n_clusters": k}


def tf_kmeans_label_transform(columns: list[list], state: dict, **_: Any) -> list:
    """Transform: assign each row to nearest cluster center."""
    _require_sklearn()
    km = state.get("kmeans")
    if km is None:
        return [0] * max(len(c) for c in columns) if columns else []
    n_rows = max(len(c) for c in columns)
    X = _to_matrix(columns, n_rows)
    labels = km.predict(X)
    return labels.tolist()


def tf_kmeans_distance_transform(columns: list[list], state: dict, **_: Any) -> list:
    """Transform: distance to each cluster center."""
    _require_sklearn()
    km = state.get("kmeans")
    if km is None:
        n = max(len(c) for c in columns) if columns else 0
        return [[0.0] * state.get("n_clusters", 1)] * n
    n_rows = max(len(c) for c in columns)
    X = _to_matrix(columns, n_rows)
    dists = km.transform(X)
    return [row.tolist() for row in dists]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
if _SKLEARN_AVAILABLE:

    register(
        FeatureOperator(
            name="pca",
            family="dimensionality",
            arity="unary",
            input_types=("continuous_numeric",),
            output_type="numeric",
            purpose="compress correlated numeric dimensions via PCA",
            preconditions=("scaled/imputed numeric; n,p sufficient",),
            fit_scope=FitScope.TRAINING_FOLD,
            null_policy="preprocess missing",
            leakage_class=LeakageClass.L1,
            target_permission=TargetPermission.P0_NONE,
            cost_shape="O(np*k)",
            generation_trigger="high-dimensional correlated numeric",
            rejection_conditions=("interpretability", "no need", "cost"),
            validation="inside-fold",
            inference_requirement="stored preprocessing/components",
            transform=tf_pca_transform,
            mandatory_tests=("component shape",),
        )
    )

    register(
        FeatureOperator(
            name="truncated_svd",
            family="dimensionality",
            arity="unary",
            input_types=("continuous_numeric",),
            output_type="numeric",
            purpose="compress sparse representations via truncated SVD",
            preconditions=("sparse input; k feasible",),
            fit_scope=FitScope.TRAINING_FOLD,
            null_policy="sparse zeros natural",
            leakage_class=LeakageClass.L1,
            target_permission=TargetPermission.P0_NONE,
            cost_shape="O(nnz*k)",
            generation_trigger="TFIDF/high-dimensional sparse",
            rejection_conditions=("k too high", "cost"),
            validation="inside-fold",
            inference_requirement="stored components",
            transform=tf_truncated_svd_transform,
            mandatory_tests=("shape",),
        )
    )

    register(
        FeatureOperator(
            name="text_svd",
            family="dimensionality",
            arity="unary",
            input_types=("free_text",),
            output_type="numeric",
            purpose="compress sparse text TF-IDF via truncated SVD",
            preconditions=("TFIDF exists, k budgeted",),
            fit_scope=FitScope.TRAINING_FOLD,
            null_policy="empty -> zeros",
            leakage_class=LeakageClass.L1,
            target_permission=TargetPermission.P0_NONE,
            cost_shape="O(nnz*k)",
            generation_trigger="large sparse text + dense learner",
            rejection_conditions=("small n", "cost", "no gain"),
            validation="inside-fold",
            inference_requirement="stored components",
            transform=tf_text_svd_transform,
            mandatory_tests=("projection equivalence",),
        )
    )

    register(
        FeatureOperator(
            name="kmeans_label",
            family="cluster",
            arity="unary",
            input_types=("continuous_numeric",),
            output_type="categorical",
            purpose="coarse regime membership via KMeans cluster ID",
            preconditions=("scaled/imputed",),
            fit_scope=FitScope.TRAINING_FOLD,
            null_policy="preprocess missing",
            leakage_class=LeakageClass.L1,
            target_permission=TargetPermission.P0_NONE,
            cost_shape="O(nkp)",
            generation_trigger="cluster structure prior",
            rejection_conditions=("unstable", "no gain", "cost"),
            validation="inside-fold",
            inference_requirement="stored centers/preprocess",
            transform=tf_kmeans_label_transform,
            mandatory_tests=("cluster permutation stability",),
        )
    )

    register(
        FeatureOperator(
            name="kmeans_distances",
            family="cluster",
            arity="unary",
            input_types=("continuous_numeric",),
            output_type="numeric",
            purpose="distance to each KMeans cluster center",
            preconditions=("scaled/imputed",),
            fit_scope=FitScope.TRAINING_FOLD,
            null_policy="preprocess missing",
            leakage_class=LeakageClass.L1,
            target_permission=TargetPermission.P0_NONE,
            cost_shape="O(nkp)",
            generation_trigger="cluster geometry",
            rejection_conditions=("too many centers", "latency"),
            validation="inside-fold",
            inference_requirement="stored centers",
            transform=tf_kmeans_distance_transform,
            mandatory_tests=("distance finite",),
        )
    )
