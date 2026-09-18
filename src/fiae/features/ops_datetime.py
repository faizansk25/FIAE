"""Datetime feature operations (doc 05 entries 33-42).

Extract calendar parts from date/datetime strings.  Null in -> null out.
Invalid parse -> explicit missing (invariant 10).  L0 row-wise operators.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any, Optional

from ..contracts import FitScope, LeakageClass, TargetPermission
from .registry import FeatureOperator, register

_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S.%f",
)


def _parse_date(v: Any) -> Optional[_dt.datetime]:
    """Parse a value to ``datetime``; return ``None`` for missing/invalid."""
    if v is None:
        return None
    if isinstance(v, _dt.datetime):
        return v
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return _dt.datetime.fromtimestamp(v, tz=_dt.timezone.utc)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        iso = s.replace("Z", "+00:00")
        try:
            return _dt.datetime.fromisoformat(iso)
        except ValueError:
            pass
        for fmt in _DATE_FORMATS:
            try:
                return _dt.datetime.strptime(s, fmt)
            except ValueError:
                continue
    return None


def _d(v: Any) -> Optional[_dt.datetime]:
    return _parse_date(v)


# --------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------
def tf_year(values: list, **_: Any) -> list:
    out: list[Optional[int]] = []
    for v in values:
        d = _d(v)
        out.append(d.year if d is not None else None)
    return out


def tf_month(values: list, **_: Any) -> list:
    out: list[Optional[int]] = []
    for v in values:
        d = _d(v)
        out.append(d.month if d is not None else None)
    return out


def tf_quarter(values: list, **_: Any) -> list:
    out: list[Optional[int]] = []
    for v in values:
        d = _d(v)
        out.append((d.month - 1) // 3 + 1 if d is not None else None)
    return out


def tf_day_of_week(values: list, **_: Any) -> list:
    out: list[Optional[int]] = []
    for v in values:
        d = _d(v)
        out.append(d.weekday() if d is not None else None)  # Mon=0..Sun=6
    return out


def tf_day_of_month(values: list, **_: Any) -> list:
    out: list[Optional[int]] = []
    for v in values:
        d = _d(v)
        out.append(d.day if d is not None else None)
    return out


def tf_hour(values: list, **_: Any) -> list:
    out: list[Optional[int]] = []
    for v in values:
        d = _d(v)
        out.append(d.hour if d is not None else None)  # 0..23
    return out


def tf_is_weekend(values: list, **_: Any) -> list:
    out: list[Optional[bool]] = []
    for v in values:
        d = _d(v)
        out.append(d.weekday() >= 5 if d is not None else None)
    return out


def tf_is_month_start(values: list, **_: Any) -> list:
    out: list[Optional[bool]] = []
    for v in values:
        d = _d(v)
        out.append(d.day == 1 if d is not None else None)
    return out


def tf_is_month_end(values: list, **_: Any) -> list:
    import calendar as _cal
    out: list[Optional[bool]] = []
    for v in values:
        d = _d(v)
        if d is not None:
            out.append(d.day == _cal.monthrange(d.year, d.month)[1])
        else:
            out.append(None)
    return out


def tf_week_of_year(values: list, **_: Any) -> list:
    """Extract ISO week of year (1-53)."""
    out: list[Optional[int]] = []
    for v in values:
        d = _d(v)
        out.append(d.isocalendar()[1] if d is not None else None)
    return out


def tf_day_of_year(values: list, **_: Any) -> list:
    """Extract day of year (1-366)."""
    out: list[Optional[int]] = []
    for v in values:
        d = _d(v)
        out.append(d.timetuple().tm_yday if d is not None else None)
    return out


def tf_minute(values: list, **_: Any) -> list:
    """Extract minute of hour (0-59)."""
    out: list[Optional[int]] = []
    for v in values:
        d = _d(v)
        out.append(d.minute if d is not None else None)
    return out


def tf_cyclical_month(values: list, **_: Any) -> list:
    """Encode month as cyclical sin/cos pair."""
    out: list[Optional[float]] = []
    for v in values:
        d = _d(v)
        if d is not None:
            m = d.month  # 1-12
            out.append(math.sin(2 * math.pi * m / 12))
        else:
            out.append(None)
    return out


def tf_cyclical_month_cos(values: list, **_: Any) -> list:
    """Cosine component of cyclical month encoding."""
    out: list[Optional[float]] = []
    for v in values:
        d = _d(v)
        if d is not None:
            m = d.month
            out.append(math.cos(2 * math.pi * m / 12))
        else:
            out.append(None)
    return out


def tf_cyclical_dow(values: list, **_: Any) -> list:
    """Encode day of week as cyclical sin/cos pair."""
    out: list[Optional[float]] = []
    for v in values:
        d = _d(v)
        if d is not None:
            dow = d.weekday()  # 0-6
            out.append(math.sin(2 * math.pi * dow / 7))
        else:
            out.append(None)
    return out


def tf_cyclical_dow_cos(values: list, **_: Any) -> list:
    """Cosine component of cyclical day-of-week encoding."""
    out: list[Optional[float]] = []
    for v in values:
        d = _d(v)
        if d is not None:
            dow = d.weekday()
            out.append(math.cos(2 * math.pi * dow / 7))
        else:
            out.append(None)
    return out


# --------------------------------------------------------------------------
# registrations
# --------------------------------------------------------------------------
def _dt_op(name: str, purpose: str, tf, trigger: str, rejects: tuple,
        tests: tuple, output_type: str = "numeric") -> None:
    register(
        FeatureOperator(
            name=name,
            family="datetime",
            arity="unary",
            input_types=("datetime",),
            output_type=output_type,
            purpose=purpose,
            preconditions=("parseable date/datetime or configured missing",),
            fit_scope=FitScope.NONE,
            null_policy="preserve",
            leakage_class=LeakageClass.L0,
            target_permission=TargetPermission.P0_NONE,
            cost_shape="O(n)",
            generation_trigger=trigger,
            rejection_conditions=rejects,
            validation="incremental CV",
            inference_requirement="input available",
            transform=tf,
            mandatory_tests=tests,
        )
    )

_dt_op('year', 'extract calendar year', tf_year,
    'any timestamp column',
    ('constant year', 'no variance'),
    ('parse correctness', 'null preservation'))
_dt_op('month', 'extract calendar month 1-12', tf_month,
    'any timestamp column',
    ('constant month', 'no variance'),
    ('range 1-12', 'null preservation'))
_dt_op('quarter', 'extract calendar quarter 1-4', tf_quarter,
    'any timestamp column',
    ('constant quarter', 'no variance'),
    ('range 1-4', 'null preservation'))
_dt_op('day_of_week', 'extract day of week (Mon=0..Sun=6)', tf_day_of_week,
    'any timestamp column',
    ('constant weekday', 'no variance'),
    ('range 0-6', 'null preservation'))
_dt_op('day_of_month', 'extract day of month 1-31', tf_day_of_month,
    'any timestamp column',
    ('constant day', 'no variance'),
    ('range 1-31', 'null preservation'))
_dt_op('hour', 'extract hour of day 0-23', tf_hour,
    'timestamp with time component',
    ('constant hour', 'no variance'),
    ('range 0-23', 'null preservation'))
_dt_op('is_weekend', 'flag Saturday/Sunday', tf_is_weekend,
    'daily or finer frequency data',
    ('weekend rare', 'no variance'),
    ('boolean output', 'null preservation'),
    output_type='boolean')
_dt_op('is_month_start', 'flag first day of month', tf_is_month_start,
    'monthly/calendar data',
    ('rare', 'no variance'),
    ('boolean output', 'null preservation'),
    output_type='boolean')
_dt_op('is_month_end', 'flag last day of month', tf_is_month_end,
    'monthly/calendar data',
    ('rare', 'no variance'),
    ('boolean output', 'null preservation'),
    output_type='boolean')
_dt_op('week_of_year', 'extract ISO week of year 1-53', tf_week_of_year,
    'any timestamp column',
    ('constant week', 'no span'),
    ('year-boundary ISO', 'null preservation'))
_dt_op('day_of_year', 'extract day of year 1-366', tf_day_of_year,
    'any timestamp column',
    ('short span', 'no variation'),
    ('leap year', 'null preservation'))
_dt_op('minute', 'extract minute of hour 0-59', tf_minute,
    'sub-hour resolution events',
    ('high noise', 'no variation'),
    ('resolution', 'null preservation'))
_dt_op('cyclical_month', 'encode month as cyclical sin component',
    tf_cyclical_month,
    'linear/distance models with monthly seasonality',
    ('tree-only', 'no gain'),
    ('norm approx 1', 'null preservation'))
_dt_op('cyclical_dow', 'encode day of week as cyclical sin component',
    tf_cyclical_dow,
    'cyclic weekly patterns',
    ('no gain',),
    ('period 7', 'null preservation'))
