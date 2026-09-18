"""Feature acceptance record (doc 04).

Every accepted/rejected feature stores: proposal sources, lineage, operation/parameters,
precondition evidence, leakage class, availability status, scores by stage,
incremental gain, resource cost, fold stability, redundancy/interaction notes,
and finally a reason code. Both the funnel (F0-F2) and the portfolio layer
(F3-F6) append their own evidence to this single record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class StageVerdict:
    """Outcome of a single funnel gate (F0/F1/F2)."""

    stage: str
    passed: bool
    reason: Optional[str] = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "passed": self.passed,
            "reason": self.reason,
            "metrics": _copy_mapping(self.metrics),
        }


def _copy_mapping(d: Optional[dict[str, Any]]) -> dict[str, Any]:
    return dict(d) if d else {}


@dataclass
class FeatureAcceptanceRecord:
    """All evidence gathered about one candidate feature through the funnel."""

    feature_id: str
    operator: str
    inputs: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    source: list[str] = field(default_factory=list)
    trigger_reason: str = ""
    lineage: list[str] = field(default_factory=list)
    leakage_class: str = "L0"
    availability: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    incremental_gain: Optional[float] = None
    resource: dict[str, Any] = field(default_factory=dict)
    fold_stability: Optional[float] = None
    redundancy_notes: list[str] = field(default_factory=list)
    final_reason: str = ""
    stages: list[StageVerdict] = field(default_factory=list, repr=False)

    @property
    def passed(self) -> bool:
        return bool(self.stages) and all(v.passed for v in self.stages)

    def add_stage(self, verdict: StageVerdict) -> None:
        self.stages.append(verdict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "operator": self.operator,
            "inputs": list(self.inputs),
            "params": _copy_mapping(self.params),
            "source": list(self.source),
            "trigger_reason": self.trigger_reason,
            "lineage": list(self.lineage),
            "leakage_class": self.leakage_class,
            "availability": self.availability,
            "scores": _copy_mapping(self.scores),
            "incremental_gain": self.incremental_gain,
            "resource": _copy_mapping(self.resource),
            "fold_stability": self.fold_stability,
            "redundancy_notes": list(self.redundancy_notes),
            "final_reason": self.final_reason,
            "stages": [v.to_dict() for v in self.stages],
        }
