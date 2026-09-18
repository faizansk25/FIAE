"""Automatic dataset report engine — Kaggle-style EDA, stdlib only.

Computes real statistics from any data source:
- per-numeric-column histograms
- Pearson correlation matrix (numeric columns)
- categorical frequency distributions
- missing-value map
- target class balance
- summary statistics (mean/std/min/quartiles/max)

All pure Python (NFR-002); results are JSON-serializable and rendered as
inline SVG by the GUI (``fiae ui`` -> Report tab).
"""

from __future__ import annotations

import math
import re as _re
from typing import Any, Optional


MAX_HIST_BINS = 16
MAX_SCATTER_ROWS = 400          # downsample cap for density-independent visuals
MAX_CORR_COLUMNS = 12           # correlation matrix cap (largest side)
MAX_DENSITY_GRID = 64           # KDE evaluation points per curve
MAX_KDE_SAMPLE = 1000           # max points fed into a KDE
MAX_SCATTER_PAIRS = 4           # strongest |r| pairs to sample
MAX_RIDGE_COLUMNS = 4           # ridgeline columns
MAX_RIDGE_GROUPS = 6            # ridgeline target classes
MAX_OUTLIER_EXAMPLES = 20       # sample outlier values kept per column


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_float_or_none(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        f = float(s)
        return None if math.isnan(f) or math.isinf(f) else f
    except ValueError:
        return None


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    sx = sy = sxx = syy = sxy = 0.0
    m = 0
    for i in range(n):
        x, y = xs[i], ys[i]
        if x is None or y is None:
            continue
        m += 1
        sx += x
        sy += y
        sxx += x * x
        syy += y * y
        sxy += x * y
    if m < 3:
        return 0.0
    cov = sxy / m - (sx / m) * (sy / m)
    vx = sxx / m - (sx / m) ** 2
    vy = syy / m - (sy / m) ** 2
    if vx <= 0 or vy <= 0:
        return 0.0
    r = cov / math.sqrt(vx * vy)
    return max(-1.0, min(1.0, r))


def _mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(var)


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _histogram(values: list[float], bins: int = MAX_HIST_BINS) -> dict:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"edges": [], "counts": []}
    lo, hi = min(clean), max(clean)
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in clean:
        idx = int((v - lo) / width)
        counts[min(idx, bins - 1)] += 1
    edges = [round(lo + i * width, 6) for i in range(bins + 1)]
    return {"edges": edges, "counts": counts}


def _kde_curve(clean: list[float], grid_n: int = MAX_DENSITY_GRID) -> dict:
    """Deterministic Gaussian KDE on a uniform grid, y normalized to [0, 1].

    Silverman bandwidth: h = 0.9 * min(std, IQR/1.349) * n^(-1/5). Points are
    subsampled on even indices (never randomly) for reproducibility.
    """
    n = len(clean)
    if n < 3:
        return {"x": [], "y": []}
    s = sorted(clean)
    _mean, std = _mean_std(clean)
    iqr = _quantile(s, 0.75) - _quantile(s, 0.25)
    sigma = min(std, iqr / 1.349) if iqr > 0 else std
    if sigma <= 0:
        sigma = std if std > 0 else 1.0
    h = 0.9 * sigma * (n ** -0.2)
    if h <= 0:
        h = 1e-9

    if n > MAX_KDE_SAMPLE:
        idxs = [int(i * (n - 1) / (MAX_KDE_SAMPLE - 1))
                for i in range(MAX_KDE_SAMPLE)]
        pts = [clean[i] for i in idxs]
    else:
        pts = clean
    m = len(pts)

    lo, hi = min(clean) - 3 * h, max(clean) + 3 * h
    step = (hi - lo) / (grid_n - 1) if grid_n > 1 else 1.0
    xs = [lo + i * step for i in range(grid_n)]
    norm = 1.0 / (m * h * 2.5066282746310002)  # 1/(m h sqrt(2pi))
    ys = []
    for x in xs:
        acc = 0.0
        for v in pts:
            z = (x - v) / h
            acc += math.exp(-0.5 * z * z)
        ys.append(acc * norm)
    ymax = max(ys) or 1.0
    return {"x": [round(x, 6) for x in xs],
            "y": [round(y / ymax, 6) for y in ys]}


def _outlier_summary(s_sorted: list[float], q25: float, q75: float) -> dict:
    """IQR-fence outlier stats. ``s_sorted`` is the sorted clean values."""
    if not s_sorted:
        return {"count": 0, "lo_fence": None, "hi_fence": None,
                "examples": []}
    iqr = q75 - q25
    lo = q25 - 1.5 * iqr
    hi = q75 + 1.5 * iqr
    outliers = [v for v in s_sorted if v < lo or v > hi]
    step = max(1, len(outliers) // MAX_OUTLIER_EXAMPLES)
    return {
        "count": len(outliers),
        "lo_fence": round(lo, 6),
        "hi_fence": round(hi, 6),
        "examples": [round(v, 6) for v in outliers[::step]
                     [:MAX_OUTLIER_EXAMPLES]],
    }


# ---------------------------------------------------------------------------
# Analyst intelligence: understand the data before choosing charts
# ---------------------------------------------------------------------------

_DATE_RE = _re.compile(
    r"^\d{4}-\d{2}(-\d{2})?([ T]\d{2}:\d{2}(:\d{2})?)?$")
_ID_NAME_RE = _re.compile(
    r"(^|_)(id|uuid|guid|key|index|idx|no|num|number)$", _re.IGNORECASE)


def _skewness(clean: list[float]) -> float:
    """Population skewness (Fisher-Pearson). 0 for n < 3 or zero variance."""
    n = len(clean)
    if n < 3:
        return 0.0
    mean = sum(clean) / n
    m2 = sum((v - mean) ** 2 for v in clean) / n
    m3 = sum((v - mean) ** 3 for v in clean) / n
    if m2 <= 0:
        return 0.0
    return m3 / (m2 ** 1.5)


def _looks_datetime(raw: list, sample: int = 50) -> bool:
    """True if >=60% of a deterministic sample of values parse as dates."""
    vals = raw[::max(1, len(raw) // sample)][:sample]
    hits = 0
    seen = 0
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()
        seen += 1
        if _DATE_RE.match(s):
            hits += 1
        else:
            try:
                float(s)
            except ValueError:
                return False if hits == 0 else hits / max(seen, 1) >= 0.6
    return seen > 0 and hits / seen >= 0.6


def _classify_column(name: str, raw: list, stats: dict, rows: int) -> tuple:
    """Return (role, reason) for one column. Roles drive chart selection."""
    n = len(raw)
    non_missing = n - stats.get("missing", 0)
    distinct = stats.get("distinct")
    kind = stats.get("kind")
    if distinct is None and kind != "numeric":
        # categorical stats always carry distinct; numeric needs a cheap count
        distinct = len({str(v) for v in raw if v is not None})\
            if n <= 50_000 else None
    missing = stats.get("missing", 0)

    if non_missing == 0:
        return "empty", "column has no values"
    if distinct is not None and distinct <= 1:
        return "constant", "every value is identical — no signal to chart"
    if kind == "numeric":
        if _ID_NAME_RE.search(name) or (distinct is not None
                                        and non_missing > 20
                                        and distinct >= 0.9 * non_missing):
            return "identifier", (f"unique/row-label values "
                                   f"({distinct}/{non_missing} distinct) — "
                                   "charting them misleads")
        if non_missing and (missing / n) > 0.6:
            return "sparse", f"{round(100*missing/n)}% missing"
        return "numeric", "continuous measure"
    # categorical-ish
    if distinct is not None and non_missing:
        if distinct / non_missing > 0.9 and non_missing > 20:
            return "identifier", ("near-unique labels "
                                   f"({distinct}/{non_missing})")
        avg_len = (sum(len(str(v)) for v in raw if v is not None)
                   / max(non_missing, 1))
        if distinct > 50 and avg_len > 20:
            return "free_text", (f"{distinct} distinct long values — "
                                 "needs NLP, not frequency charts")
        if (missing / n) > 0.6:
            return "sparse", f"{round(100*missing/n)}% missing"
        if _looks_datetime(raw):
            return "datetime", "values parse as timestamps"
        if distinct == 2:
            return "binary", "two distinct values"
        if distinct <= 20:
            return "categorical", f"{distinct} categories"
        return "high_card", f"{distinct} categories — bar charts would drown"
    return "categorical", "discrete labels"


def _mean_shift_notable(groups: list[dict], std: float) -> bool:
    """Ridgeline is only interesting if class means actually differ."""
    if len(groups) < 2 or std is None or std <= 0:
        return False
    means = []
    for g in groups:
        d = g["density"]
        if d["x"] and d["y"]:
            # approximate class mean: density-weighted center
            num = sum(x * y for x, y in zip(d["x"], d["y"]))
            den = sum(d["y"]) or 1.0
            means.append(num / den)
    if len(means) < 2:
        return False
    return (max(means) - min(means)) > 0.3 * std


def _build_plan(roles: dict, column_stats: dict, corr_columns: list,
                matrix: list) -> dict:
    """Decide which charts earn a place, with a stated reason each.

    ``roles`` maps column name -> {role, reason} (from _classify_column).
    """
    plan: dict[str, list] = {}
    excluded: dict[str, str] = {}
    for name, info in roles.items():
        role = info["role"]
        stats = column_stats[name]
        if role in ("identifier", "constant", "free_text", "sparse",
                    "empty"):
            excluded[name] = f"{role}: {info['reason']}"
            continue
        charts = []
        if role == "numeric":
            skew = stats.get("_skew", 0.0)
            charts.append({"chart": "histogram",
                           "why": "distribution shape"})
            charts.append({"chart": "violin",
                           "why": "five-number summary + outliers"})
            if abs(skew) > 1.0:
                charts.append({
                    "chart": "density",
                    "why": f"skewed ({'right' if skew > 0 else 'left'}, "
                           f"skew={skew:.2f}) — density makes the tail "
                           "visible"})
            plan[name] = charts
        elif role in ("binary", "categorical"):
            distinct = stats.get("distinct", 0)
            top = stats.get("top_values", [])
            charts.append({"chart": "bar", "why": f"{distinct} categories"})
            if 4 <= distinct <= 20 and len(top) >= 3:
                total = sum(t["count"] for t in top) or 1
                cum3 = sum(t["count"] for t in top[:3]) / total
                if cum3 >= 0.8:
                    charts.append({
                        "chart": "pareto",
                        "why": f"top-3 categories hold {round(100*cum3)}% "
                               "— classic 80/20"})
            if 2 <= distinct <= 6:
                charts.append({"chart": "donut",
                               "why": f"{distinct} parts make an honest pie"})
            plan[name] = charts
        elif role == "high_card":
            plan[name] = [{"chart": "bar",
                           "why": "top values only — full set too large "
                                  "to read"}]
        elif role == "datetime":
            plan[name] = [{"chart": "bar",
                           "why": "top periods (full temporal view needs "
                                  "timestamp parsing — roadmap)"}]

    # scatter pairs: gate hard on |r| >= 0.5 — below that it is noise art
    strong_pairs = []
    for i, a in enumerate(corr_columns):
        for j, b in enumerate(corr_columns):
            if j > i and abs(matrix[i][j]) >= 0.5:
                strong_pairs.append({"x": a, "y": b,
                                     "r": matrix[i][j]})
    strong_pairs.sort(key=lambda p: -abs(p["r"]))

    return {"plan": plan, "excluded": excluded,
            "strong_pairs": strong_pairs[:MAX_SCATTER_PAIRS]}


def _gather_insights(column_stats: dict, corr_columns: list, matrix: list,
                     target_balance: Optional[dict],
                     missing_map: list, plan: dict,
                     ridgeline: Optional[dict]) -> list:
    """Ranked narrative findings — the analyst's first three sentences."""
    insights = []

    # strongest correlation
    best = None
    for i, a in enumerate(corr_columns):
        for j, b in enumerate(corr_columns):
            if j > i and (best is None or abs(matrix[i][j]) > abs(best[0])):
                best = (matrix[i][j], a, b)
    if best and abs(best[0]) >= 0.5:
        direction = "positive" if best[0] > 0 else "negative"
        insights.append({"severity": "info",
                         "text": f"`{best[1]}` and `{best[2]}` move together "
                                 f"(r={best[0]:.2f}, {direction}) — one may "
                                 "be redundant for modeling."})

    # skew
    for name, s in column_stats.items():
        if s["kind"] != "numeric" or name not in plan:
            continue
        sk = s.get("_skew", 0.0)
        if abs(sk) > 1.0:
            insights.append({"severity": "info",
                             "text": f"`{name}` is heavily "
                                     f"{'right' if sk > 0 else 'left'}-skewed "
                                     f"(skew={sk:.2f}) — consider a log or "
                                     "rank transform before modeling."})

    # outliers
    for name, s in column_stats.items():
        if s["kind"] == "numeric" and s.get("outliers", {}).get("count", 0):
            of = s["outliers"]
            pct = round(100 * of["count"] / max(s["count"], 1), 1)
            if pct >= 1.0:
                insights.append({"severity": "warning",
                                 "text": f"`{name}` has {of['count']} outliers "
                                         f"({pct}% of values) beyond the IQR "
                                         "fences — check data entry errors or "
                                         "heavy tails."})

    # missing data
    missy = [m for m in missing_map if m["null_fraction"] > 0.2]
    if missy:
        worst = max(missy, key=lambda m: m["null_fraction"])
        insights.append({"severity": "warning",
                         "text": f"`{worst['column']}` is "
                                 f"{round(100*worst['null_fraction'])}% empty — "
                                 "imputation or column drop needed."})

    # junk columns
    junk = plan.get("excluded", {})
    if junk:
        names = ", ".join(f"`{n}`" for n in list(junk)[:5])
        insights.append({"severity": "info",
                         "text": f"{len(junk)} column(s) excluded from charts "
                                 f"({names}) — identifiers/constants/text "
                                 "carry no chartable signal."})

    # imbalance
    if target_balance and target_balance.get("imbalance_ratio", 1.0) >= 1.5:
        tb = target_balance
        insights.append({"severity": "warning",
                         "text": f"Target `{tb['column']}` is imbalanced "
                                 f"({tb['imbalance_ratio']}:1) — use stratified "
                                 "splits and balanced metrics."})

    # class mean shifts
    if ridgeline:
        for col, rc in ridgeline["columns"].items():
            stats = column_stats.get(col, {})
            if _mean_shift_notable(rc["groups"], stats.get("std")):
                insights.append({"severity": "info",
                                 "text": f"`{col}` distributions shift across "
                                         f"`{ridgeline['target']}` classes — "
                                         "a promising predictor."})
    return insights[:12]


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------


def build_report(source: str, target: Optional[str] = None,
                 table: Optional[str] = None,
                 max_rows: int = 20_000) -> dict:
    """Compute the full EDA report for a data source.

    Works with any source the intake layer supports (25+ kinds). The
    ``table`` argument is forwarded for database sources; ``target`` adds a
    class-balance section when the column exists.
    """
    from .intake import auto_adapter, ProfileConfig, ProfileMode, profile_source

    kwargs = {"table": table} if table else {}
    adapter = auto_adapter(source, **kwargs)
    profile = profile_source(adapter, ProfileConfig(mode=ProfileMode.FAST))

    # ---- scan up to max_rows ------------------------------------------------
    columns: dict[str, list] = {}
    rows_read = 0
    for batch in adapter.scan():
        n = max((len(v) for v in batch.columns.values()), default=0)
        for name, vals in batch.columns.items():
            columns.setdefault(name, []).extend(vals)
        rows_read += n
        if rows_read >= max_rows:
            break
    for name in columns:
        columns[name] = columns[name][:max_rows]

    column_meta = {c.name: c for c in profile.columns}
    numeric: dict[str, list[Optional[float]]] = {}
    categorical: dict[str, list] = {}
    for name, raw in columns.items():
        meta = column_meta.get(name)
        sem = getattr(meta, "semantic_type", "") if meta else ""
        sem_val = getattr(sem, "value", sem)  # enum -> str
        sem_str = str(sem_val)
        if ("categorical" in sem_str or "boolean" in sem_str
                or "text" in sem_str or "identifier" in sem_str):
            categorical[name] = raw
        else:
            numeric[name] = [_to_float_or_none(v) for v in raw]

    # ---- per-column summaries ----------------------------------------------
    column_stats: dict[str, dict] = {}
    for name, vals in numeric.items():
        clean = [v for v in vals if v is not None]
        s = sorted(clean)
        mean, std = _mean_std(clean)
        column_stats[name] = {
            "kind": "numeric",
            "count": len(clean),
            "missing": len(vals) - len(clean),
            "mean": round(mean, 6) if clean else None,
            "std": round(std, 6) if clean else None,
            "min": round(s[0], 6) if s else None,
            "q25": round(_quantile(s, 0.25), 6) if s else None,
            "median": round(_quantile(s, 0.5), 6) if s else None,
            "q75": round(_quantile(s, 0.75), 6) if s else None,
            "max": round(s[-1], 6) if s else None,
            "histogram": _histogram(vals),
            "density": _kde_curve(clean) if clean else {"x": [], "y": []},
            "outliers": _outlier_summary(
                s,
                _quantile(s, 0.25) if s else 0.0,
                _quantile(s, 0.75) if s else 0.0),
        }
        # keep skew internal (used for planning; exposed via insights)
        column_stats[name]["_skew"] = round(_skewness(clean), 3)

    for name, raw in categorical.items():
        counts: dict[str, int] = {}
        missing = 0
        for v in raw:
            if v is None or str(v).strip() == "":
                missing += 1
                continue
            key = str(v)
            counts[key] = counts.get(key, 0) + 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:12]
        column_stats[name] = {
            "kind": "categorical",
            "count": len(raw) - missing,
            "missing": missing,
            "distinct": len(counts),
            "top_values": [{"value": k, "count": c} for k, c in top],
        }

    # ---- correlation matrix (numeric columns) -------------------------------
    corr_columns = [n for n, vals in numeric.items()
                    if any(v is not None for v in vals)][:MAX_CORR_COLUMNS]
    matrix: list[list[float]] = []
    for a in corr_columns:
        row = []
        for b in corr_columns:
            if a == b:
                row.append(1.0)
            else:
                row.append(round(_pearson(numeric[a], numeric[b]), 3))
        matrix.append(row)

    # ---- missing map ---------------------------------------------------------
    missing_map = []
    for name in columns:
        meta = column_meta.get(name)
        mf = getattr(meta, "null_fraction", None)
        if mf is None:
            vals = columns[name]
            mf = (sum(1 for v in vals if v is None) / len(vals)) if vals else 0.0
        missing_map.append({"column": name, "null_fraction": round(mf, 4)})

    # ---- target balance -------------------------------------------------------
    target_balance: Optional[dict] = None
    if target and target in columns:
        counts: dict[str, int] = {}
        for v in columns[target]:
            key = "?" if v is None else str(v)
            counts[key] = counts.get(key, 0) + 1
        total = sum(counts.values()) or 1
        target_balance = {
            "column": target,
            "classes": [{"value": k, "count": c,
                         "pct": round(100 * c / total, 1)}
                        for k, c in sorted(counts.items(),
                                           key=lambda kv: -kv[1])[:12]],
            "imbalance_ratio": (round(max(counts.values())
                                      / max(min(counts.values()), 1), 2)
                                if len(counts) > 1 else 1.0),
        }

    # ---- scatter pairs (strongest |r|, downsampled points) ------------------
    scatter_pairs: list[dict] = []
    pair_rank: list[tuple[float, str, str]] = []
    for i, a in enumerate(corr_columns):
        for j, b in enumerate(corr_columns):
            if j > i and abs(matrix[i][j]) > 0.05:
                pair_rank.append((abs(matrix[i][j]), a, b))
    pair_rank.sort(reverse=True)
    for _, a, b in pair_rank[:MAX_SCATTER_PAIRS]:
        xa, xb = numeric[a], numeric[b]
        valid = [(p, q) for p, q in zip(xa, xb)
                 if p is not None and q is not None]
        if len(valid) < 5:
            continue
        stride = max(1, len(valid) // MAX_SCATTER_ROWS)
        pts = [[round(p, 6), round(q, 6)] for p, q in valid[::stride]]
        r = round(matrix[corr_columns.index(a)][corr_columns.index(b)], 3)
        scatter_pairs.append({"x": a, "y": b, "r": r, "points": pts})

    # ---- ridgeline (class-conditional densities) ----------------------------
    ridgeline: Optional[dict] = None
    if target and target in columns:
        tvals = columns[target]
        tcounts: dict[str, int] = {}
        for v in tvals:
            key = "?" if v is None else str(v)
            tcounts[key] = tcounts.get(key, 0) + 1
        if 2 <= len(tcounts) <= 12:
            top_groups = [k for k, _ in
                          sorted(tcounts.items(), key=lambda kv: -kv[1])
                          [:MAX_RIDGE_GROUPS]]
            group_set = set(top_groups)
            ridge_cols: dict[str, dict] = {}
            for name, vals in list(numeric.items())[:MAX_RIDGE_COLUMNS]:
                if name == target:
                    continue
                buckets: dict[str, list[float]] = {g: [] for g in top_groups}
                for v, t in zip(vals, tvals):
                    if v is None:
                        continue
                    g = "?" if t is None else str(t)
                    if g in group_set:
                        buckets[g].append(v)
                groups_out = []
                for g in top_groups:
                    if len(buckets[g]) >= 3:
                        groups_out.append({
                            "value": g, "n": len(buckets[g]),
                            "density": _kde_curve(buckets[g]),
                        })
                if len(groups_out) >= 2:
                    ridge_cols[name] = {"groups": groups_out}
            if ridge_cols:
                ridgeline = {"target": target, "columns": ridge_cols}

    # ---- analyst layer: classify -> plan -> gate -> narrate -----------------
    roles: dict[str, dict] = {}
    for name in columns:
        role, reason = _classify_column(
            name, columns[name], column_stats.get(name, {}), rows_read)
        roles[name] = {"role": role, "reason": reason}

    plan = _build_plan(roles, column_stats, corr_columns, matrix)

    # ridgeline: keep only columns where class means actually separate
    if ridgeline:
        kept = {}
        for col, rc in ridgeline["columns"].items():
            if _mean_shift_notable(rc["groups"],
                                   column_stats.get(col, {}).get("std")):
                kept[col] = rc
        ridgeline["columns"] = kept
        if not kept:
            ridgeline = None

    insights = _gather_insights(
        column_stats, corr_columns, matrix, target_balance,
        missing_map, plan, ridgeline)

    return {
        "source_id": adapter.source_id(),
        "fingerprint": profile.dataset_fingerprint,
        "rows_scanned": rows_read,
        "n_columns": len(columns),
        "column_stats": column_stats,
        "correlation": {"columns": corr_columns, "matrix": matrix},
        "missing_map": missing_map,
        "target_balance": target_balance,
        "scatter_pairs": scatter_pairs,
        "ridgeline": ridgeline,
        "chart_plan": {"roles": roles, **plan},
        "insights": insights,
    }
