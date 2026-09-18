"""Calibration and decision-policy contracts (doc 03).

- produce unbiased development probabilities (OOF)
- measure reliability/Brier/log loss
- fit a calibrator only on development-safe data
- probability estimation and decision action are separate; the threshold is
  optimized on development predictions and FROZEN before final holdout
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Any, Optional

from ..ids import content_hash
from .leakage import rank_auc


def brier(y_true: list[float], y_pred: list[float]) -> float:
    """Brier score (doc 03 metric routing)."""
    n = len(y_true)
    if n == 0:
        return 0.0
    return sum((p - y) ** 2 for y, p in zip(y_true, y_pred)) / n


def log_loss_binary(
    y_true: list[float], y_pred: list[float], eps: float = 1e-15
) -> float:
    """Binary log loss with clipping (doc 03 metric routing)."""
    n = len(y_true)
    if n == 0:
        return 0.0
    total = 0.0
    for y, p in zip(y_true, y_pred):
        p = max(eps, min(1.0 - eps, p))
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / n


class BinnedCalibrator:
    """Equal-sample bin calibrator with monotone (isotonic-style) projection.

    Fits only on development-safe (OOF) predictions. Monotonicity is enforced
    by a pool-adjacent-violators merge so calibrated values never invert the
    score order; documented as a simple stdlib calibrator (M6 may add more).
    """

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = max(2, int(n_bins))
        self._edges: list[float] = []
        self._values: list[float] = []
        self.fitted = False

    def fit(self, scores: list[float], y: list[float]) -> "BinnedCalibrator":
        n = len(scores)
        if n == 0:
            raise ValueError("cannot fit calibrator on empty data")
        order = sorted(range(n), key=lambda i: scores[i])
        bin_size = max(1, n // self.n_bins)
        bins: list[tuple[float, list[int]]] = []
        for start in range(0, n, bin_size):
            chunk = order[start : start + bin_size]
            if not chunk:
                break
            bins.append((scores[chunk[0]], chunk))

        # pool-adjacent-violators -> monotone non-decreasing calibrated values
        stack_v: list[float] = []
        stack_w: list[float] = []
        stack_idx: list[list[int]] = []
        for slot, (_edge, chunk) in enumerate(bins):
            mean = sum(y[i] for i in chunk) / len(chunk)
            stack_v.append(mean)
            stack_w.append(float(len(chunk)))
            stack_idx.append([slot])
            while len(stack_v) >= 2 and stack_v[-1] < stack_v[-2] - 1e-12:
                idx2 = stack_idx.pop()
                w2 = stack_w.pop()
                v2 = stack_v.pop()
                idx1 = stack_idx.pop()
                w1 = stack_w.pop()
                v1 = stack_v.pop()
                w = w1 + w2
                stack_v.append((w1 * v1 + w2 * v2) / w)
                stack_w.append(w)
                stack_idx.append(idx1 + idx2)
        # After PAV, flatten the stack into calibrated values per bin
        flat: dict[int, float] = {}
        for v, idxs in zip(stack_v, stack_idx):
            for ci in idxs:
                flat[ci] = v
        self._values = [flat[i] for i in range(len(bins))]
        self._edges = [b[0] for b in bins]
        self.fitted = True
        return self

    def transform(self, scores: list[float]) -> list[float]:
        """Map raw scores to calibrated probabilities using frozen bin state."""
        if not self.fitted:
            raise RuntimeError("BinnedCalibrator must be fit before transform")
        out = []
        n = len(self._values)
        for s in scores:
            i = bisect.bisect_right(self._edges, s) - 1
            if i < 0:
                out.append(self._values[0])
            elif i >= n:
                out.append(self._values[-1])
            else:
                out.append(self._values[i])
        return out


@dataclass
class ThresholdResult:
    threshold: float
    objective: str
    value: float
    n_pos: int
    n_neg: int
    at_precision: float
    at_recall: float
    warnings: list[str] = field(default_factory=list)


def optimize_threshold(
    scores: list[float],
    y: list[float],
    *,
    objective: str = "f1",
    precision_floor: Optional[float] = None,
    recall_floor: Optional[float] = None,
    utility: Optional[dict[str, float]] = None,
    max_candidates: int = 500,
) -> ThresholdResult:
    """Optimize a decision threshold on DEVELOPMENT predictions only (doc 03)."""
    y01 = [1.0 if float(yi) == 1 else 0.0 for yi in y]
    n_pos = int(sum(y01))
    n_neg = len(y01) - n_pos
    cands = sorted(set(scores))
    if len(cands) > max_candidates:
        step = max(1, len(cands) // max_candidates)
        cands = cands[::step]
    if not cands:
        return ThresholdResult(
            0.5, objective, 0.0, n_pos, n_neg, 0.0, 0.0,
            warnings=["no distinct thresholds; default 0.5"],
        )
    best = None  # (score, threshold, prec, rec)
    for t in cands:
        tp = fp = tn = fn = 0
        for p, yi in zip(scores, y01):
            pred = 1 if p >= t else 0
            if pred == 1 and yi == 1:
                tp += 1
            elif pred == 1 and yi == 0:
                fp += 1
            elif pred == 0 and yi == 0:
                tn += 1
            elif pred == 0 and yi == 1:
                fn += 1
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if objective == "f1":
            score = f1
        elif objective == "precision_floor":
            if precision_floor is None or prec < precision_floor - 1e-9:
                continue
            score = rec
        elif objective == "recall_floor":
            if recall_floor is None or rec < recall_floor - 1e-9:
                continue
            score = prec
        elif objective == "utility":
            if not utility:
                raise ValueError("utility objective requires a cost/utility matrix")
            score = (
                tp * utility.get("tp", 0.0)
                + fp * utility.get("fp", 0.0)
                + tn * utility.get("tn", 0.0)
                + fn * utility.get("fn", 0.0)
            )
        else:
            raise ValueError(f"unknown threshold objective: {objective}")
        if best is None or score > best[0] + 1e-12:
            best = (score, t, prec, rec)
    if best is None:
        return ThresholdResult(
            0.5, objective, 0.0, n_pos, n_neg, 0.0, 0.0,
            warnings=[f"no threshold satisfies objective '{objective}'; default 0.5"],
        )
    return ThresholdResult(
        threshold=best[1],
        objective=objective,
        value=best[0],
        n_pos=n_pos,
        n_neg=n_neg,
        at_precision=best[2],
        at_recall=best[3],
    )


@dataclass
class FrozenDecisionPolicy:
    """Threshold/calibration frozen before final holdout (doc 03)."""

    objective: str
    threshold: float
    calibrator: Optional[str]
    fingerprint: str


def freeze_decision_policy(
    result: ThresholdResult, *, calibrator_ref: Optional[str] = None
) -> FrozenDecisionPolicy:
    """Freeze the decision policy; reopening/re-tuning is forbidden after this."""
    fp = content_hash(
        {
            "objective": result.objective,
            "threshold": result.threshold,
            "calibrator": calibrator_ref,
        }
    )
    return FrozenDecisionPolicy(
        objective=result.objective,
        threshold=result.threshold,
        calibrator=calibrator_ref,
        fingerprint=fp,
    )


def distribution_shift_probe(
    train_values: list[float],
    test_values: list[float],
    *,
    high_separation: float = 0.95,
) -> dict[str, Any]:
    """Domain-classifier style separation probe (doc 03, distribution shift).

    Time-series tasks need special handling: chronological shift may be natural
    and informative; callers decide how to weight this signal.
    """
    if not train_values or not test_values:
        return {"separation_auc": None, "high_shift_risk": False, "note": "empty input"}
    scores = [float(x) for x in train_values] + [float(x) for x in test_values]
    labels = [0] * len(train_values) + [1] * len(test_values)
    auc = rank_auc(scores, labels)
    return {
        "separation_auc": round(auc, 4) if auc is not None else None,
        "high_shift_risk": auc is not None and auc >= high_separation,
        "n_train": len(train_values),
        "n_test": len(test_values),
    }
