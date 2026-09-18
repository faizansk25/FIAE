"""Auto-tuning of funnel policies from the Experience Store (doc 06).

The :class:`~fiae.funnel.FunnelPolicy` thresholds are explicitly *policy,
not universal constants*. This module closes the loop: it aggregates the
failure taxonomy recorded in the Experience Store and proposes bounded,
deterministic adjustments to the funnel policy.

Design guarantees:
- **Deterministic** — same store content yields the same tuned policy.
- **Bounded** — every adjustment is clamped to a documented safe range.
- **Conservative** — adjustments only trigger with sufficient evidence
  (``min_cases`` and a minimum tag rate), never on a single bad run.
- **Transparent** — every adjustment carries its reason and is reported.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from .experience.store import ExperienceStore
from .funnel import FunnelPolicy


# Bounds: adjustment clamps (policy safety envelope)
_F5_MAX_CORR_RANGE = (0.80, 0.99)
_F5_MIN_MI_RANGE = (0.05, 0.40)
_MAX_DEPTH_RANGE = (1, 3)
_F2_INVALID_RANGE = (0.10, 0.50)
_MIN_PEAK_RAM_MB = 64.0
_MIN_CPU_SECONDS = 5.0

# Evidence thresholds
_MIN_TAG_RATE = 0.30       # tag must appear in >= 30% of relevant cases
_MIN_SUCCESS_CASES = 10    # relaxation needs a real track record


@dataclass
class PolicyAdjustment:
    """One bounded change to the funnel policy."""

    field: str
    old: Any
    new: Any
    reason: str


@dataclass
class TuningReport:
    """Result of tuning a FunnelPolicy from experience."""

    cases_analyzed: int = 0
    success_rate: float = 0.0
    tag_counts: dict[str, int] = field(default_factory=dict)
    adjustments: list[PolicyAdjustment] = field(default_factory=list)
    policy: Optional[FunnelPolicy] = None

    def to_dict(self) -> dict:
        return {
            "cases_analyzed": self.cases_analyzed,
            "success_rate": round(self.success_rate, 4),
            "tag_counts": dict(sorted(self.tag_counts.items())),
            "adjustments": [
                {
                    "field": a.field,
                    "old": a.old,
                    "new": a.new,
                    "reason": a.reason,
                }
                for a in self.adjustments
            ],
            "policy": {
                f: getattr(self.policy, f)
                for f in self.policy.__dataclass_fields__
            }
            if self.policy is not None
            else None,
        }


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def tune_policy_from_experience(
    store: ExperienceStore,
    base_policy: Optional[FunnelPolicy] = None,
    min_cases: int = 5,
) -> TuningReport:
    """Derive a tuned :class:`FunnelPolicy` from stored case diagnoses.

    Parameters
    ----------
    store : ExperienceStore
        Experience store with historical case records.
    base_policy : FunnelPolicy, optional
        Starting policy (defaults to the library defaults).
    min_cases : int
        Minimum number of cases required before any adjustment is made.

    Returns
    -------
    TuningReport
        Aggregated evidence, adjustments, and the tuned policy.
    """
    policy = base_policy or FunnelPolicy()
    cases = store.all_cases()
    report = TuningReport(cases_analyzed=len(cases))
    report.tag_counts = dict(
        Counter(
            tag.value if hasattr(tag, "value") else str(tag)
            for c in cases
            for tag in c.diagnosis.failure_tags
        )
    )

    n = len(cases)
    if n < min_cases:
        # Insufficient evidence — keep base policy untouched.
        report.policy = policy
        return report

    successes = sum(1 for c in cases if c.is_success())
    report.success_rate = successes / n

    def rate(tag: str) -> float:
        return report.tag_counts.get(tag, 0) / n

    adjustments: list[PolicyAdjustment] = []
    current: dict[str, Any] = {
        f: getattr(policy, f) for f in policy.__dataclass_fields__
    }

    def setv(fname: str, new, reason: str) -> None:
        old = current[fname]
        if new != old:
            current[fname] = new
            adjustments.append(PolicyAdjustment(fname, old, new, reason))

    # --- F5 redundancy gate -------------------------------------------
    if rate("REDUNDANT") >= _MIN_TAG_RATE:
        setv(
            "f5_max_correlation",
            round(_clamp(current["f5_max_correlation"] - 0.05, *_F5_MAX_CORR_RANGE), 4),
            f"REDUNDANT tag rate {rate('REDUNDANT'):.0%} — demand lower correlation",
        )
    if rate("NO_INCREMENTAL_GAIN") >= _MIN_TAG_RATE:
        setv(
            "f5_min_mutual_info_ratio",
            round(_clamp(current["f5_min_mutual_info_ratio"] + 0.05, *_F5_MIN_MI_RANGE), 4),
            f"NO_INCREMENTAL_GAIN rate {rate('NO_INCREMENTAL_GAIN'):.0%} — demand more new information",
        )

    # --- Resource gates -------------------------------------------------
    if rate("RESOURCE_RAM") >= _MIN_TAG_RATE:
        setv(
            "max_estimated_peak_ram_mb",
            max(current["max_estimated_peak_ram_mb"] * 0.8, _MIN_PEAK_RAM_MB),
            f"RESOURCE_RAM rate {rate('RESOURCE_RAM'):.0%} — tighten RAM budget",
        )
    if rate("RESOURCE_TIME") >= _MIN_TAG_RATE:
        setv(
            "max_estimated_cpu_seconds",
            max(current["max_estimated_cpu_seconds"] * 0.8, _MIN_CPU_SECONDS),
            f"RESOURCE_TIME rate {rate('RESOURCE_TIME'):.0%} — tighten CPU budget",
        )

    # --- Fit diagnosis ---------------------------------------------------
    if rate("OVERFIT_HIGH_VARIANCE") >= _MIN_TAG_RATE:
        setv(
            "max_depth",
            int(_clamp(current["max_depth"] - 1, *_MAX_DEPTH_RANGE)),
            f"OVERFIT_HIGH_VARIANCE rate {rate('OVERFIT_HIGH_VARIANCE'):.0%} — reduce composition depth",
        )
    if rate("UNDERFIT_HIGH_BIAS") >= _MIN_TAG_RATE:
        setv(
            "max_depth",
            int(_clamp(current["max_depth"] + 1, *_MAX_DEPTH_RANGE)),
            f"UNDERFIT_HIGH_BIAS rate {rate('UNDERFIT_HIGH_BIAS'):.0%} — allow deeper compositions",
        )

    # --- Domain/validity problems: stricter F2 invalid tolerance ---------
    domain_rate = (
        rate("INVALID_DOMAIN") + rate("NUMERIC_OVERFLOW") + rate("TOO_MANY_LEVELS")
    )
    if domain_rate >= _MIN_TAG_RATE:
        setv(
            "f2_max_invalid_rate",
            round(_clamp(current["f2_max_invalid_rate"] - 0.1, *_F2_INVALID_RANGE), 4),
            f"domain/overflow/level failures {domain_rate:.0%} — stricter F2 invalid tolerance",
        )

    # --- Everything sails through: mild relaxation ------------------------
    if (
        successes >= _MIN_SUCCESS_CASES
        and report.success_rate >= 0.90
        and not report.tag_counts
    ):
        setv(
            "f5_min_mutual_info_ratio",
            round(_clamp(current["f5_min_mutual_info_ratio"] - 0.02, *_F5_MIN_MI_RANGE), 4),
            f"success rate {report.success_rate:.0%} over {n} cases with no failures — mild relaxation",
        )

    report.adjustments = adjustments
    report.policy = FunnelPolicy(**current)
    return report

