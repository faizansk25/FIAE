"""Temporal feature operations (doc 05 entries 43-55).

Shift / lag / lead / diff / pct_change / rolling / expanding.  These operators
operate on **sorted, group-isolated** arrays: the caller is responsible for
sorting by time within each entity group.  The transforms themselves are pure
functions on aligned lists.

No-future property: every rolling window looks only at the current row and
those *before* it (insert-after-emit).  Lag/lead use explicit integer offsets.

L0 row-wise (no target access); leakage risk lives in prediction-time
availability, not in construction.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from ..contracts import FitScope, LeakageClass, TargetPermission
from .registry import FeatureOperator, register


def _f(v: Any) -> Optional[float]:
    """Coerce to float; None for missing/invalid."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return None

# --------------------------------------------------------------------------
# lag / lead / diff / pct_change
# --------------------------------------------------------------------------
def tf_lag(values: list, periods: int = 1, **_: Any) -> list:
    """Shift values *backward* by ``periods`` rows (past values)."""
    out: list[Optional[float]] = [None] * periods
    for i in range(periods, len(values)):
        out.append(_f(values[i - periods]))
    return out


def tf_lead(values: list, periods: int = 1, **_: Any) -> list:
    """Shift values *forward* by ``periods`` rows (future values)."""
    n = len(values)
    out: list[Optional[float]] = []
    for i in range(n):
        if i + periods < n:
            out.append(_f(values[i + periods]))
        else:
            out.append(None)
    return out


def tf_diff(values: list, periods: int = 1, **_: Any) -> list:
    """Period-over-period difference: x[t] - x[t-periods]."""
    out: list[Optional[float]] = [None] * periods
    for i in range(periods, len(values)):
        a, b = _f(values[i]), _f(values[i - periods])
        out.append(a - b if a is not None and b is not None else None)
    return out


def tf_pct_change(values: list, periods: int = 1, eps: float = 1e-12, **_: Any) -> list:
    """Percentage change: (x[t] - x[t-periods]) / (x[t-periods] + eps)."""
    out: list[Optional[float]] = [None] * periods
    for i in range(periods, len(values)):
        a, b = _f(values[i]), _f(values[i - periods])
        if a is not None and b is not None and abs(b + eps) > eps:
            out.append((a - b) / (b + eps))
        else:
            out.append(None)
    return out


# --------------------------------------------------------------------------
# rolling window transforms (no-future: window i..i, never i..i+future)
# --------------------------------------------------------------------------
def _rolling_window(values: list, window: int, i: int) -> list:
    """Return the backward-looking window for position *i* (inclusive)."""
    start = max(0, i - window + 1)
    return [_f(values[j]) for j in range(start, i + 1)]


def tf_rolling_mean(values: list, window: int = 3, min_periods=None, **_):
    mp = min_periods or window
    out = []
    for i in range(len(values)):
        w = [x for x in _rolling_window(values, window, i) if x is not None]
        if len(w) >= mp:
            out.append(sum(w) / len(w))
        else:
            out.append(None)
    return out


def tf_rolling_sum(values: list, window: int = 3, min_periods=None, **_):
    mp = min_periods or window
    out = []
    for i in range(len(values)):
        w = [x for x in _rolling_window(values, window, i) if x is not None]
        if len(w) >= mp:
            out.append(sum(w))
        else:
            out.append(None)
    return out


def tf_rolling_std(values: list, window: int = 3, min_periods=None, **_):
    mp = min_periods or window
    out = []
    for i in range(len(values)):
        w = [x for x in _rolling_window(values, window, i) if x is not None]
        if len(w) >= mp:
            mean = sum(w) / len(w)
            var = sum((x - mean) ** 2 for x in w) / len(w)
            out.append(math.sqrt(var))
        else:
            out.append(None)
    return out


def tf_rolling_max(values: list, window: int = 3, min_periods=None, **_):
    mp = min_periods or window
    out = []
    for i in range(len(values)):
        w = [x for x in _rolling_window(values, window, i) if x is not None]
        if len(w) >= mp:
            out.append(max(w))
        else:
            out.append(None)
    return out


def tf_rolling_min(values: list, window: int = 3, min_periods=None, **_):
    mp = min_periods or window
    out = []
    for i in range(len(values)):
        w = [x for x in _rolling_window(values, window, i) if x is not None]
        if len(w) >= mp:
            out.append(min(w))
        else:
            out.append(None)
    return out


def tf_rolling_count(values: list, window: int = 3, **_):
    """Count of non-null values in the backward window."""
    out = []
    for i in range(len(values)):
        w = _rolling_window(values, window, i)
        out.append(sum(1 for x in w if x is not None))
    return out


# --------------------------------------------------------------------------
# rolling_unique: distinct count in backward window
# --------------------------------------------------------------------------
def tf_rolling_unique(values: list, window: int = 3, min_periods=None, **_):
    """Count of distinct non-null values in the backward window."""
    mp = min_periods or window
    out = []
    for i in range(len(values)):
        w = [x for x in _rolling_window(values, window, i) if x is not None]
        if len(w) >= mp:
            out.append(len(set(w)))
        else:
            out.append(None)
    return out


# --------------------------------------------------------------------------
# expanding_mean: mean of all prior observations
# --------------------------------------------------------------------------
def tf_expanding_mean(values: list, min_periods: int = 1, **_):
    """Cumulative mean of all prior observations (insert-after-emit)."""
    out: list[Optional[float]] = []
    running_sum = 0.0
    running_count = 0
    for v in values:
        x = _f(v)
        if x is not None:
            running_sum += x
            running_count += 1
        if running_count >= min_periods:
            out.append(running_sum / running_count)
        else:
            out.append(None)
    return out


# --------------------------------------------------------------------------
# ewma: exponential weighted moving average
# --------------------------------------------------------------------------
def tf_ewma(values: list, alpha: float = 0.3, **_):
    """Exponential weighted moving average.  s_t = alpha*x_t + (1-alpha)*s_{t-1}."""
    out: list[Optional[float]] = []
    state = None
    for v in values:
        x = _f(v)
        if x is None:
            out.append(state)
        elif state is None:
            state = x
            out.append(x)
        else:
            state = alpha * x + (1 - alpha) * state
            out.append(state)
    return out


# --------------------------------------------------------------------------
# time_since_previous: inter-event gap
# --------------------------------------------------------------------------
def tf_time_since_previous(values: list, **_):
    """Time gap between consecutive observations.

    Values are expected to be numeric timestamps or already-numeric.
    Missing for first observation.
    """
    out: list[Optional[float]] = [None]
    for i in range(1, len(values)):
        a, b = _f(values[i]), _f(values[i - 1])
        if a is not None and b is not None:
            gap = a - b
            out.append(gap if gap >= 0 else None)  # negative = disorder -> missing
        else:
            out.append(None)
    return out


# --------------------------------------------------------------------------
# time_since_first: entity age since first observation
# --------------------------------------------------------------------------
def tf_time_since_first(values: list, **_):
    """Time elapsed since the first observation.

    First row gets 0.  Subsequent rows get T_i - T_0.
    """
    out: list[Optional[float]] = []
    first_val = None
    for v in values:
        x = _f(v)
        if first_val is None and x is not None:
            first_val = x
        if x is None or first_val is None:
            out.append(None)
        else:
            out.append(x - first_val)
    return out


# --------------------------------------------------------------------------
# registrations
# --------------------------------------------------------------------------
def _temporal(name, purpose, tf, trigger, rejects, tests):
    register(
        FeatureOperator(
            name=name,
            family="temporal",
            arity="unary",
            input_types=("continuous_numeric",),
            output_type="numeric",
            purpose=purpose,
            preconditions=(
                "sorted by time within entity group",
                "group isolation enforced by caller",
            ),
            fit_scope=FitScope.NONE,
            null_policy="preserve",
            leakage_class=LeakageClass.L0,
            target_permission=TargetPermission.P0_NONE,
            cost_shape="O(n * window)",
            generation_trigger=trigger,
            rejection_conditions=rejects,
            validation="incremental CV",
            inference_requirement="same entity time-slice available",
            transform=tf,
            mandatory_tests=tests,
            time_semantics="sequential",
        )
    )


_temporal("lag", "past value at fixed offset",
          tf_lag, "time-series with sufficient history",
          ("series too short", "offset > data length"),
          ("no-future", "group isolation", "null at boundary"))
_temporal("lead", "future value at fixed offset (forecast target proxy)",
          tf_lead, "forecasting with explicit cutoff",
          ("uses future data", "series too short"),
          ("no-future boundary only", "null at boundary"))
_temporal("diff", "period-over-period change",
          tf_diff, "any numeric time-series",
          ("stationary series", "too much noise"),
          ("no-future", "null at boundary"))
_temporal("pct_change", "relative period-over-period change",
          tf_pct_change, "positive-valued time-series",
          ("near-zero denominator", "stationary"),
          ("no-future", "zero policy"))

_rolls = {
    "rolling_mean": ("backward rolling mean", tf_rolling_mean),
    "rolling_sum": ("backward rolling sum", tf_rolling_sum),
    "rolling_std": ("backward rolling std dev", tf_rolling_std),
    "rolling_max": ("backward rolling max", tf_rolling_max),
    "rolling_min": ("backward rolling min", tf_rolling_min),
    "rolling_count": ("backward window non-null count", tf_rolling_count),
}
for _name, (_purpose, _tf) in _rolls.items():
    _temporal(_name, _purpose, _tf,
              "rolling history available",
              ("window > series length", "insufficient data"),
              ("no-future", "window boundary null"))

_temporal("rolling_unique", "distinct count in backward window",
          tf_rolling_unique, "behavior diversity",
          ("memory cap", "window > series length"),
          ("no-future", "counter eviction"))
_temporal("expanding_mean", "long-term historical mean",
          tf_expanding_mean, "long-term baseline",
          ("tiny data",),
          ("insert-after-emit",))
_temporal("ewma", "recency-weighted moving average",
          tf_ewma, "recent history important",
          ("alpha search too broad",),
          ("online equals batch",))
_temporal("time_since_previous", "inter-event gap in numeric units",
          tf_time_since_previous, "event cadence",
          ("duplicate time ambiguous",),
          ("non-negative gaps",))
_temporal("time_since_first", "entity age since first seen",
          tf_time_since_first, "entity lifecycle",
          ("first time from future",),
          ("no future first",))


# --------------------------------------------------------------------------
# rolling_unique: distinct count in backward window
# --------------------------------------------------------------------------

