"""Task inference and metric routing (doc 03).

Rules (doc 03, Task inference and Metric routing):
- explicit task wins; otherwise infer from target profile with confidence+reasons
- string/bool/categorical target -> classification candidate
- numeric target with low semantic cardinality -> classification candidate
- numeric target with broad continuous support -> regression
- time-indexed target plus future horizon -> forecasting
- ranking requires explicit query/group semantics
- no-target anomaly/unsupervised mode must be explicit
- material ambiguity is surfaced, never silently guessed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..contracts import (
    ColumnProfile,
    Direction,
    MetricValue,
    ProblemDefinition,
    SemanticType,
    Task,
)

MULTICLASS_CAP = 10
IMBALANCE_POSITIVE_RATE = 0.2
# NOTE: 1 and True hash identically, so the single literal 1 covers both.
TRUTHY_POSITIVE = frozenset({1, "1", "true", "yes", "y"})
_NUMERIC_SEMANTICS = frozenset(
    {
        SemanticType.CONTINUOUS_NUMERIC,
        SemanticType.COUNT,
        SemanticType.ORDINAL,
        SemanticType.CURRENCY,
        SemanticType.PERCENTAGE,
    }
)


@dataclass
class MetricTemplate:
    """A metric to compute, with its optimization direction (doc 03)."""

    name: str
    direction: Direction
    unit: Optional[str] = None
    note: str = ""


@dataclass
class TaskInference:
    task: Task
    confidence: float
    reasons: list[str] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)
    positive_class: Optional[Any] = None
    class_count: Optional[int] = None
    positive_rate: Optional[float] = None
    is_imbalanced: bool = False


def default_positive(classes: list[Any]) -> Optional[Any]:
    """Choose a default positive class for binary targets (doc 03 step 0025)."""
    for c in classes:
        if c is True or c == 1:
            return c
        s = str(c).strip().lower()
        if s in TRUTHY_POSITIVE:
            return c
    return classes[-1] if classes else None
def infer_task(
    target_profile: Optional[ColumnProfile],
    *,
    explicit_task: Optional[Task] = None,
    target_values: Optional[list[Any]] = None,
    time_key: Optional[str] = None,
    horizon: Optional[str] = None,
    group_keys: Optional[list[str]] = None,
    positive_class: Any = None,
    multiclass_cap: int = MULTICLASS_CAP,
) -> TaskInference:
    """Infer or confirm the task with recorded confidence, reasons, ambiguities."""
    reasons: list[str] = []
    ambiguities: list[str] = []

    # 1) Explicit task honored (rules 1, 6, 7).
    if explicit_task is not None and explicit_task is not Task.AUTO:
        reasons.append("explicit user override")
        if explicit_task is Task.RANKING and not group_keys:
            ambiguities.append("ranking requires explicit query/group semantics")
        if explicit_task in (Task.BINARY, Task.MULTICLASS) and target_profile is not None:
            _classification_validate(target_profile, multiclass_cap, ambiguities, reasons)
        if explicit_task is Task.UNSUPERVISED:
            reasons.append("explicit unsupervised mode; target ignored")
        return TaskInference(
            task=explicit_task,
            confidence=1.0,
            reasons=reasons,
            ambiguities=ambiguities,
            class_count=target_profile.distinct_estimate if target_profile else None,
        )

    # 2) Unsupervised must be explicit (rule 7).
    if target_profile is None:
        ambiguities.append(
            "no target supplied and task not explicit; unsupervised/anomaly mode must be explicit"
        )
        return TaskInference(task=Task.AUTO, confidence=0.3, ambiguities=ambiguities)

    # 3) Auto inference from target semantics (rules 2-4, 8-9).
    candidates: list[Task] = []
    st = target_profile.semantic_type
    phys = target_profile.physical_dtype
    distinct = target_profile.distinct_estimate

    if st is SemanticType.BOOLEAN or phys == "boolean":
        candidates.append(Task.BINARY)
        reasons.append("boolean target")
    elif st is SemanticType.DATETIME:
        ambiguities.append("datetime target is not a supported supervised target")
    elif st in (
        SemanticType.LOW_CARDINALITY_CATEGORICAL,
        SemanticType.HIGH_CARDINALITY_CATEGORICAL,
    ):
        if distinct == 2:
            candidates.append(Task.BINARY)
            reasons.append(f"categorical target with {distinct} classes")
        elif distinct <= multiclass_cap:
            candidates.append(Task.MULTICLASS)
            reasons.append(f"categorical target with {distinct} classes")
        else:
            ambiguities.append(
                "high-cardinality categorical target; requires explicit task/class policy"
            )
    elif st in _NUMERIC_SEMANTICS or phys in ("integer", "float"):
        if distinct <= 2:
            candidates.append(Task.BINARY)
            reasons.append("numeric target with low semantic cardinality (binary)")
        elif distinct <= multiclass_cap:
            candidates.append(Task.MULTICLASS)
            reasons.append(
                f"numeric target with low semantic cardinality ({distinct} classes)"
            )
        else:
            candidates.append(Task.REGRESSION)
            reasons.append("numeric target with broad continuous support")
    else:
        ambiguities.append("target semantics not routable; requires explicit task")

    # 4) Forecasting takes precedence when time + horizon exist (rule 5).
    if time_key and horizon:
        candidates = [Task.FORECASTING, *candidates]
        reasons.append("time-indexed target plus declared horizon -> forecasting")

    task = candidates[0] if candidates else Task.AUTO
    if task is Task.AUTO and not ambiguities:
        ambiguities.append("unable to infer task from target semantics")

    # 5) Positive class / imbalance when binary (class order resolution).
    class_count: Optional[int] = None
    positive_cls: Optional[Any] = None
    positive_rate: Optional[float] = None
    is_imbalanced = False
    if task is Task.BINARY:
        classes = None
        if target_values:
            classes = sorted({v for v in target_values if v is not None})
        class_count = len(classes) if classes else target_profile.distinct_estimate
        if classes and len(classes) == 2:
            pc = positive_class if positive_class is not None else default_positive(classes)
            positive_cls = pc
            pos = sum(1 for v in target_values if v == pc)
            positive_rate = pos / max(len(target_values), 1)
            is_imbalanced = positive_rate < IMBALANCE_POSITIVE_RATE

    confidence = 0.3 if task is Task.AUTO else (0.6 if ambiguities else 1.0)
    return TaskInference(
        task=task,
        confidence=confidence,
        reasons=reasons,
        ambiguities=ambiguities,
        positive_class=positive_cls,
        class_count=class_count,
        positive_rate=positive_rate,
        is_imbalanced=is_imbalanced,
    )


def _classification_validate(
    target_profile: ColumnProfile,
    multiclass_cap: int,
    ambiguities: list[str],
    reasons: list[str],
) -> None:
    distinct = target_profile.distinct_estimate
    if distinct > multiclass_cap:
        ambiguities.append(
            f"explicit classification but target has {distinct} classes (> cap {multiclass_cap})"
        )
    else:
        reasons.append(f"target supports classification ({distinct} classes)")


def route_metrics(
    task: Task,
    *,
    is_imbalanced: bool = False,
    probability_quality: bool = True,
    action_metric: Optional[str] = None,
    have_cost_spec: bool = False,
    non_negative_target: bool = False,
) -> list[MetricTemplate]:
    """Metric routing tables from doc 03."""
    out: list[MetricTemplate] = []
    if task in (Task.BINARY, Task.MULTICLASS):
        if task is Task.BINARY:
            out.append(MetricTemplate("roc_auc", Direction.MAXIMIZE, note="ranking quality"))
            if is_imbalanced:
                out.append(
                    MetricTemplate("average_precision", Direction.MAXIMIZE, note="rare positives")
                )
            if probability_quality:
                out.append(
                    MetricTemplate("log_loss", Direction.MINIMIZE, note="probability quality")
                )
                out.append(
                    MetricTemplate("brier", Direction.MINIMIZE, note="probability quality")
                )
            if action_metric == "f1":
                out.append(MetricTemplate("f1", Direction.MAXIMIZE, note="action objective"))
            if have_cost_spec:
                out.append(
                    MetricTemplate(
                        "business_utility", Direction.MAXIMIZE, note="supplied cost/utility spec"
                    )
                )
        else:
            out.append(MetricTemplate("log_loss", Direction.MINIMIZE))
            out.append(MetricTemplate("macro_f1", Direction.MAXIMIZE))
            out.append(MetricTemplate("micro_f1", Direction.MAXIMIZE))
            out.append(MetricTemplate("balanced_accuracy", Direction.MAXIMIZE))
    elif task in (Task.REGRESSION, Task.FORECASTING):
        out.append(MetricTemplate("rmse", Direction.MINIMIZE, note="primary"))
        out.append(MetricTemplate("mae", Direction.MINIMIZE))
        out.append(MetricTemplate("r2", Direction.MAXIMIZE, note="secondary"))
        if non_negative_target:
            out.append(
                MetricTemplate("rmsle", Direction.MINIMIZE, note="guarded: non-negative only")
            )
            out.append(
                MetricTemplate(
                    "mape", Direction.MINIMIZE, note="guarded: zero/near-zero handled"
                )
            )
        if task is Task.FORECASTING:
            out.append(
                MetricTemplate("mape", Direction.MINIMIZE, note="chronological validation")
            )
    return out


def make_problem(
    inference: TaskInference,
    *,
    target: str,
    group_keys: Optional[list[str]] = None,
    time_key: Optional[str] = None,
    horizon: Optional[str] = None,
    prediction_time_semantics: Optional[str] = None,
    templates: Optional[list[MetricTemplate]] = None,
) -> ProblemDefinition:
    """Build an immutable ProblemDefinition (doc 13) from an inference."""
    metrics = [
        MetricValue(
            name=t.name,
            value=0.0,  # placeholder; real value materializes at evaluation
            direction=t.direction,
            split="plan",
            aggregation="unset",
            unit=t.unit,
        )
        for t in (templates or [])
    ]
    return ProblemDefinition(
        task=inference.task,
        target=target,
        positive_class=inference.positive_class,
        prediction_time_semantics=prediction_time_semantics,
        group_keys=list(group_keys or []),
        time_key=time_key,
        horizon=horizon,
        metrics=metrics,
        ambiguities=list(inference.ambiguities),
    )
