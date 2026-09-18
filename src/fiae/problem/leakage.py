"""Leakage detector pipeline and cross-fit target encoding (doc 03).

Pipeline:
- Stage A deterministic: exact target copy, inverted binary, deterministic
  bijection, target embedded in strings, identifiers/row numbers.
- Stage B statistical suspicion: near-perfect single-feature AUC, extreme
  mutual information, missingness nearly determines target.
- Stage C prediction-time availability: max(availability_time) <= P_i.
- Stage D graph fold-safety: learned transforms fit on training partitions only.

Statistical suspicion is not proof (doc 03). It creates REVIEW_REQUIRED unless
semantic/temporal evidence confirms.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from ..contracts import (
    FeatureNode,
    FitScope,
    LeakageClass,
    LeakageFinding,
    LeakageSeverity,
    TargetPermission,
)

MISSING_TOKENS = frozenset({"", "nan", "none", "null", "na", "n/a"})
AUC_SUSPICION = 0.98
MI_SUSPICION = 0.9
MISSING_BIAS_SUSPICION = 0.9


def _norm(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if s.lower() in MISSING_TOKENS:
        return None
    return s


def _finding(
    kind: str,
    subject: str,
    severity: LeakageSeverity,
    leak_class: LeakageClass,
    evidence: dict[str, Any],
    action: str = "warn",
) -> LeakageFinding:
    return LeakageFinding(
        finding_id=f"L|{kind}|{subject}",
        subject=subject,
        severity=severity,
        type=leak_class,
        evidence=evidence,
        action="reject" if severity is LeakageSeverity.HARD_REJECT else action,
        override_allowed=severity is not LeakageSeverity.HARD_REJECT,
    )


# --------------------------------------------------------------------------
# Ranking / information primitives (pure stdlib)
# --------------------------------------------------------------------------
def average_ranks(scores: list[float]) -> list[float]:
    """Average ranks with tie handling (1-based)."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    n = len(order)
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def rank_auc(scores: list[float], y: list[int]) -> Optional[float]:
    """Rank-based AUC (Mann-Whitney statistic) for binary 0/1 y."""
    ranks = average_ranks(scores)
    n_pos = sum(1 for yi in y if yi == 1)
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    sum_pos = sum(ranks[i] for i in range(len(y)) if y[i] == 1)
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _entropy(counts: Iterable[int], total: int) -> float:
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


# --------------------------------------------------------------------------
# Stage A: deterministic checks
# --------------------------------------------------------------------------
def detect_deterministic(
    feature_values: list[Any],
    target_values: list[Any],
    *,
    feature_name: str = "feature",
    target_name: str = "target",
) -> Optional[LeakageFinding]:
    """Deterministic leakage rules (doc 03 Stage A). Returns first hard finding."""
    pairs = []
    for f, t in zip(feature_values, target_values):
        nf, nt = _norm(f), _norm(t)
        if nf is None or nt is None:
            continue
        # Boolean-token normalization: a renamed boolean target copy
        # ("yes"/"no") must be caught as an exact copy (doc 12 adversarial
        # suite). Raw tokens are kept otherwise.
        lower = nf.lower()
        if lower in ("yes", "true", "y"):
            nf = "1"
        elif lower in ("no", "false", "n"):
            nf = "0"
        pairs.append((nf, nt))
    if len(pairs) < 4:
        return None
    f_set = {nf for nf, _ in pairs}
    t_set = {nt for _, nt in pairs}
    if len(t_set) < 2 or len(f_set) < 2:
        return None

    # 1) Exact target copy.
    n_eq = sum(1 for nf, nt in pairs if nf == nt)
    if n_eq / len(pairs) >= 0.99:
        return _finding(
            "target_copy",
            feature_name,
            LeakageSeverity.HARD_REJECT,
            LeakageClass.L5,
            {"match_fraction": round(n_eq / len(pairs), 4), "target_classes": len(t_set)},
        )

    # 2) Inverted binary target copy.
    if len(t_set) == 2 and len(f_set) == 2:
        t_sorted = sorted(t_set)

        def complement(x):
            return t_sorted[0] if x == t_sorted[1] else t_sorted[1]

        n_inv = sum(1 for nf, nt in pairs if nf == complement(nt))
        if n_inv / len(pairs) >= 0.99:
            return _finding(
                "inverted_target_copy",
                feature_name,
                LeakageSeverity.HARD_REJECT,
                LeakageClass.L5,
                {"invert_fraction": round(n_inv / len(pairs), 4)},
            )

    # 3) Deterministic bijection (identifier-like memorization). Only
    # near-unique feature values (distinct ratio ~1, e.g. row numbers/IDs)
    # constitute Stage A proof. A low-cardinality categorical that maps
    # deterministically to the target is statistical suspicion handled by
    # Stage B (REVIEW_REQUIRED), never an automatic Stage A rejection.
    distinct_ratio = len(f_set) / len(pairs)
    target_by_f: dict[str, set[str]] = {}
    for nf, nt in pairs:
        target_by_f.setdefault(nf, set()).add(nt)
    if distinct_ratio >= 0.9 and all(len(v) == 1 for v in target_by_f.values()):
        return _finding(
            "deterministic_bijection",
            feature_name,
            LeakageSeverity.HARD_REJECT,
            LeakageClass.L5,
            {"feature_classes": len(f_set), "target_classes": len(t_set)},
        )

    # 4) Target embedded in strings. A genuinely embedded target leaves a
    # constant residue when stripped (e.g. "item_A"/"item_B" with target A/B),
    # which avoids coincidental substring false positives like a numeric feature
    # value "10.5" containing target digit "0".
    if len(t_set) >= 2:
        n_emb = 0
        residues: set[Optional[str]] = set()
        for nf, nt in pairs:
            if nt and nt in nf:
                n_emb += 1
                residues.add(nf.replace(nt, "", 1))
            else:
                residues.add(None)  # non-matching rows break constant residue
        embed_frac = n_emb / len(pairs)
        long_tokens = all(len(nt) >= 3 for nt in t_set)
        constant_residue = len({r for r in residues if r is not None}) == 1
        if embed_frac >= 0.95 and (long_tokens or constant_residue):
            return _finding(
                "target_embedded_in_string",
                feature_name,
                LeakageSeverity.HARD_REJECT,
                LeakageClass.L5,
                {"embed_fraction": round(embed_frac, 4), "target_tokens": sorted(t_set)},
            )
    return None
# --------------------------------------------------------------------------
# Stage B: statistical suspicion (never proof)
# --------------------------------------------------------------------------
def statistical_triage(
    feature_values: list[Any],
    target_values: list[Any],
    *,
    feature_name: str = "feature",
    target_name: str = "target",
) -> Optional[LeakageFinding]:
    """Statistical suspicion signals (doc 03 Stage B). Returns first suspicion."""
    t_set = sorted({_norm(t) for t in target_values if _norm(t) is not None})
    if len(t_set) != 2:
        return None  # statistical suspicion defined for binary targets

    # a) Near-perfect single-feature AUC on numeric-like features.
    scores, ys = [], []
    for f, t in zip(feature_values, target_values):
        nf, nt = _norm(f), _norm(t)
        if nf is None or nt is None:
            continue
        try:
            fl = float(nf)
        except ValueError:
            continue
        scores.append(fl)
        ys.append(1 if nt == t_set[1] else 0)
    if len(ys) >= 20 and len(set(ys)) == 2:
        auc = rank_auc(scores, ys)
        if auc is not None and auc >= AUC_SUSPICION:
            return _finding(
                "near_perfect_auc",
                feature_name,
                LeakageSeverity.REVIEW_REQUIRED,
                LeakageClass.L4,
                {"auc": round(auc, 4), "note": "statistical suspicion; needs semantic review"},
                action="review",
            )

    # b) Extreme mutual information on categorical features.
    pairs = []
    for f, t in zip(feature_values, target_values):
        nf, nt = _norm(f), _norm(t)
        if nf is None or nt is None:
            continue
        pairs.append((nf, nt))
    if pairs and len({nf for nf, _ in pairs}) <= 200:
        counts: dict[tuple[str, str], int] = {}
        fx: dict[str, int] = {}
        ty: dict[str, int] = {}
        for nf, nt in pairs:
            counts[(nf, nt)] = counts.get((nf, nt), 0) + 1
            fx[nf] = fx.get(nf, 0) + 1
            ty[nt] = ty.get(nt, 0) + 1
        total = len(pairs)
        mi = 0.0
        for (nf, nt), c in counts.items():
            pxy = c / total
            px = fx[nf] / total
            py = ty[nt] / total
            if px > 0 and py > 0:
                mi += pxy * math.log2(pxy / (px * py))
        hx = _entropy(fx.values(), total)
        hy = _entropy(ty.values(), total)
        denom = min(hx, hy)
        if denom > 0 and len(fx) >= 2:
            norm = mi / denom
            if norm >= MI_SUSPICION:
                return _finding(
                    "extreme_mutual_information",
                    feature_name,
                    LeakageSeverity.REVIEW_REQUIRED,
                    LeakageClass.L4,
                    {"normalized_mi": round(norm, 4), "note": "statistical suspicion"},
                    action="review",
                )

    # c) Missingness nearly determines target.
    miss_ys, pres_ys = [], []
    for f, t in zip(feature_values, target_values):
        nf, nt = _norm(f), _norm(t)
        if nt is None:
            continue
        label = 1 if nt == t_set[1] else 0
        if nf is None:
            miss_ys.append(label)
        else:
            pres_ys.append(label)
    if len(miss_ys) >= 5 and len(pres_ys) >= 5:
        pm = sum(miss_ys) / len(miss_ys)
        pp = sum(pres_ys) / len(pres_ys)
        if abs(pm - pp) >= MISSING_BIAS_SUSPICION:
            return _finding(
                "missingness_determines_target",
                feature_name,
                LeakageSeverity.REVIEW_REQUIRED,
                LeakageClass.L4,
                {
                    "pos_rate_missing": round(pm, 4),
                    "pos_rate_present": round(pp, 4),
                    "note": "statistical suspicion; check how missingness arises",
                },
                action="review",
            )
    return None

# --------------------------------------------------------------------------
# Stage C: prediction-time availability
# --------------------------------------------------------------------------
def review_availability(
    availability: dict[str, str],
    prediction_time: str,
    *,
    subjects: Optional[list[str]] = None,
) -> list[LeakageFinding]:
    """max(availability_time) <= P_i for every source fact (doc 03)."""
    from datetime import datetime

    try:
        p = datetime.fromisoformat(prediction_time)
    except ValueError:
        p = None
    names = subjects or list(availability)
    findings: list[LeakageFinding] = []
    for name in names:
        raw = availability.get(name)
        if raw is None:
            findings.append(
                _finding(
                    "availability_unknown",
                    name,
                    LeakageSeverity.REVIEW_REQUIRED,
                    LeakageClass.L4,
                    {"note": "feature availability time unknown; cannot prove prediction-time safety"},
                    action="review",
                )
            )
            continue
        try:
            av = datetime.fromisoformat(raw)
        except ValueError:
            findings.append(
                _finding(
                    "availability_unparseable",
                    name,
                    LeakageSeverity.INFORMATIONAL,
                    LeakageClass.L4,
                    {"raw": raw},
                )
            )
            continue
        if p is not None and av > p:
            findings.append(
                _finding(
                    "availability_after_prediction",
                    name,
                    LeakageSeverity.REVIEW_REQUIRED,
                    LeakageClass.L4,
                    {"availability": raw, "prediction_time": prediction_time},
                    action="review",
                )
            )
    return findings


def confirm_unavailable(subjects: list[str], reason: str) -> list[LeakageFinding]:
    """Proven post-outcome/operationally unavailable features (L5 hard reject)."""
    return [
        _finding(
            "prediction_time_unavailable",
            s,
            LeakageSeverity.HARD_REJECT,
            LeakageClass.L5,
            {"reason": reason, "note": "proven unavailable at prediction time"},
        )
        for s in subjects
    ]


def check_target_history_safety(
    target_history_used: bool,
    *,
    label_availability_known: bool = False,
) -> list[LeakageFinding]:
    """Target lags require explicit label-availability semantics (doc 03/05)."""
    if not target_history_used:
        return []
    if not label_availability_known:
        return [
            _finding(
                "label_delay_unknown",
                "target_history",
                LeakageSeverity.REVIEW_REQUIRED,
                LeakageClass.L3,
                {"note": "label delay unknown; target-history features disabled by default"},
                action="review",
            )
        ]
    return []


# --------------------------------------------------------------------------
# Stage D: fold-safety of learned transforms
# --------------------------------------------------------------------------
@dataclass
class FoldSafetyReport:
    ok: bool
    findings: list[str] = field(default_factory=list)


def fold_safety_report(nodes: list[FeatureNode]) -> FoldSafetyReport:
    """Learned transforms must fit on training partitions only (doc 03 Stage D)."""
    findings: list[str] = []
    for node in nodes:
        if node.fit_scope is FitScope.DEVELOPMENT:
            findings.append(
                f"{node.feature_id}: development-scope fit is final-deployment only; "
                "cannot be used for fold evaluation"
            )
        if (
            node.target_permission is TargetPermission.P1_CROSSFIT
            and node.fit_scope is not FitScope.TRAINING_FOLD
        ):
            findings.append(
                f"{node.feature_id}: target-aware feature requires cross-fit "
                "(fit scope must be TRAINING_FOLD)"
            )
    return FoldSafetyReport(ok=not findings, findings=findings)


# --------------------------------------------------------------------------
# Cross-fit target encoding (doc 03 algorithm)
# --------------------------------------------------------------------------


class CrossFitTargetEncoder:
    """Cross-fit target encoding: a row never contributes its target to its
    own encoded value (doc 03, FR-011).

    Algorithm for K folds:
        for fold k:
            train_idx = all folds except k
            val_idx = fold k
            learn smoothed category->target statistics on train_idx
            transform val_idx
        learn final mapping on all development data for deployment
    """

    def __init__(self, smoothing: float = 1.0, prior: Optional[float] = None):
        self.smoothing = float(smoothing)
        self._user_prior = prior
        self.prior: float = 0.5 if prior is None else float(prior)
        self._map: dict[str, float] = {}
        self._fitted = False

    # -- internal helpers -------------------------------------------------
    @staticmethod
    def _fold_assignments(n: int, k: int, seed: int) -> list[int]:
        """Deterministic seeded fold id per row (contiguous chunks)."""
        rng = random.Random(seed)
        idx = list(range(n))
        rng.shuffle(idx)
        fold_id = [0] * n
        size = (n + k - 1) // k
        for slot, i in enumerate(idx):
            fold_id[i] = min(slot // size, k - 1)
        return fold_id

    @staticmethod
    def _group_fold_assignments(
        n: int, group_ids: list[Any], k: int, seed: int
    ) -> list[int]:
        """Group-aware fold assignment: all rows of a group share a fold."""
        rng = random.Random(seed)
        by_group: dict[Any, list[int]] = {}
        for i, g in enumerate(group_ids):
            by_group.setdefault(g, []).append(i)
        uniq = sorted(by_group)
        rng.shuffle(uniq)
        fold_id = [0] * n
        for s, g in enumerate(uniq):
            for i in by_group[g]:
                fold_id[i] = s % k
        return fold_id

    def _smoothed_mean(
        self, cat: str, cat_targets: dict[str, list[float]]
    ) -> float:
        targets = cat_targets.get(cat)
        if targets:
            n = len(targets)
            cat_mean = sum(targets) / n
            n_adj = n + self.smoothing
            return (n * cat_mean + self.smoothing * self.prior) / n_adj
        return self.prior

    # -- public API -------------------------------------------------------
    def fit_transform(
        self,
        cats: list[str],
        targets: list[float],
        n_folds: int = 3,
        seed: int = 0,
        *,
        group_ids: Optional[list[Any]] = None,
    ) -> list[float]:
        """Return out-of-fold encoded values for each row."""
        n = len(cats)
        k = max(2, min(int(n_folds), n))

        # global prior from all targets
        if self._user_prior is None and targets:
            self.prior = sum(targets) / len(targets)

        if group_ids is not None:
            fold_id = self._group_fold_assignments(n, group_ids, k, seed)
        else:
            fold_id = self._fold_assignments(n, k, seed)

        encoded = [0.0] * n
        for f in range(k):
            val_idx = [i for i in range(n) if fold_id[i] == f]
            train_idx = [i for i in range(n) if fold_id[i] != f]
            # learn smoothed stats on training partition only
            cat_targets: dict[str, list[float]] = {}
            for i in train_idx:
                cat_targets.setdefault(cats[i], []).append(targets[i])
            for i in val_idx:
                encoded[i] = self._smoothed_mean(cats[i], cat_targets)

        # learn final deployment mapping on all development data
        self.fit_final(cats, targets)
        return encoded

    def fit_final(self, cats: list[str], targets: list[float]) -> None:
        """Learn the final category->target map on all development data."""
        if targets and self._user_prior is None:
            self.prior = sum(targets) / len(targets)
        cat_targets: dict[str, list[float]] = {}
        for c, t in zip(cats, targets):
            cat_targets.setdefault(c, []).append(t)
        self._map = {
            c: self._smoothed_mean(c, {c: ts}) for c, ts in cat_targets.items()
        }
        self._fitted = True

    def transform(self, cats: list[str]) -> list[float]:
        """Encode new data using the final deployment mapping."""
        if not self._fitted:
            raise RuntimeError("CrossFitTargetEncoder: call fit_final first")
        return [self._map.get(c, self.prior) for c in cats]
