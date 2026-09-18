"""Fitted-state pipeline for L1 operators (doc 04, doc 05).

Handles the fit/transform lifecycle for operators that learn parameters
from training data (fit_scope == TRAINING_FOLD).  L0 operators pass through
unchanged.  L2/L3 operators are excluded (they need cross-fit or temporal
routing handled elsewhere).

Design:
- ``FittedPipeline`` stores per-operator fit state keyed by feature ID.
- ``fit()`` runs all L0 transforms plus fits L1 operators on training data.
- ``transform()`` applies the fitted state to new data.
- State is serializable for inference persistence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from .contracts import FitScope
from .features.registry import FeatureOperator, get_operator
from .search.triggers import FeatureProposal


# Track which operators need fit state
_FIT_REQUIRED_SCOPES = frozenset({FitScope.TRAINING_FOLD, FitScope.DEVELOPMENT})


def _needs_fit(op: FeatureOperator) -> bool:
    """Check if an operator requires a fit step."""
    return op.fit_scope in _FIT_REQUIRED_SCOPES


def _coerce_to_floats(values: list) -> list:
    """Coerce raw values to float list, treating non-numeric as None."""
    out = []
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out.append(None)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(float(v))
        elif isinstance(v, bool):
            out.append(1.0 if v else 0.0)
        else:
            out.append(None)
    return out


@dataclass
class FittedState:
    """Fit state for one operator application."""

    op_name: str
    feature_id: str
    params: dict[str, Any] = field(default_factory=dict)
    fit_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class FittedPipeline:
    """Manages fit/transform lifecycle for a set of feature proposals.

    Usage::

        pipe = FittedPipeline()
        # On training data:
        outputs = pipe.fit(proposals, columnar_data)
        # On dev/test data:
        outputs = pipe.transform(proposals, columnar_data)
    """

    states: dict[str, FittedState] = field(default_factory=dict)

    def fit(
        self,
        proposals: list[FeatureProposal],
        columnar_data: dict[str, list],
    ) -> dict[str, list]:
        """Fit all L1 operators on training data, transform all L0 operators.

        Returns dict mapping feature_name -> transformed values.
        """
        results: dict[str, list] = {}

        for proposal in proposals:
            feat_name = proposal.op + "(" + "_".join(
                inp[4:] if inp.startswith("raw:") else inp
                for inp in proposal.inputs
            ) + ")"

            op = get_operator(proposal.op)
            input_arrays = self._get_inputs(proposal, columnar_data)
            if input_arrays is None:
                continue

            n = min(len(a) for a in input_arrays)
            trimmed = [a[:n] for a in input_arrays]

            if _needs_fit(op):
                # L1 operator: fit on training data, then transform
                try:
                    if hasattr(op, 'fit_fn') and op.fit_fn:
                        fit_state = op.fit_fn(*trimmed, **proposal.params)
                    else:
                        # For operators registered with fit/transform pairs,
                        # the transform function expects state as a parameter
                        fit_state = self._fit_operator(op, trimmed, proposal.params)
                except Exception:
                    continue

                # Store fit state
                state = FittedState(
                    op_name=op.name, feature_id=feat_name,
                    params=dict(proposal.params), fit_state=fit_state,
                )
                self.states[feat_name] = state

                # Transform using fitted state
                try:
                    output = self._transform_with_state(
                        op, trimmed, fit_state, proposal.params
                    )
                except Exception:
                    continue
            else:
                # L0 operator: just transform
                try:
                    if op.arity == "unary":
                        output = op.transform(trimmed[0], **proposal.params)
                    elif op.arity == "binary" and len(trimmed) >= 2:
                        output = op.transform(trimmed[0], trimmed[1], **proposal.params)
                    else:
                        continue
                except Exception:
                    continue

            results[feat_name] = self._to_float_list(output, n)

        return results

    def fit_transform(
        self,
        proposals: list[FeatureProposal],
        columnar_data: dict[str, list],
    ) -> dict[str, list]:
        """Fit all operators on the data, then transform it in one call."""
        self.fit(proposals, columnar_data)
        return self.transform(proposals, columnar_data)

    def transform(
        self,
        proposals: list[FeatureProposal],
        columnar_data: dict[str, list],
    ) -> dict[str, list]:
        """Apply all operators using previously fitted state.

        L0 operators are re-computed directly.
        L1 operators use stored fit state.
        """
        results: dict[str, list] = {}

        for proposal in proposals:
            feat_name = proposal.op + "(" + "_".join(
                inp[4:] if inp.startswith("raw:") else inp
                for inp in proposal.inputs
            ) + ")"

            op = get_operator(proposal.op)
            input_arrays = self._get_inputs(proposal, columnar_data)
            if input_arrays is None:
                continue

            n = min(len(a) for a in input_arrays)
            trimmed = [a[:n] for a in input_arrays]

            if _needs_fit(op):
                # Use stored fit state
                stored = self.states.get(feat_name)
                if stored is None:
                    continue  # not fitted
                try:
                    output = self._transform_with_state(
                        op, trimmed, stored.fit_state, proposal.params
                    )
                except Exception:
                    continue
            else:
                try:
                    if op.arity == "unary":
                        output = op.transform(trimmed[0], **proposal.params)
                    elif op.arity == "binary" and len(trimmed) >= 2:
                        output = op.transform(trimmed[0], trimmed[1], **proposal.params)
                    else:
                        continue
                except Exception:
                    continue

            results[feat_name] = self._to_float_list(output, n)

        return results

    def get_state_dict(self) -> dict[str, Any]:
        """Serialize all fit states for persistence."""
        result = {}
        for fid, state in self.states.items():
            result[fid] = {
                "op_name": state.op_name,
                "params": state.params,
                "fit_state": state.fit_state,
            }
        return result

    def load_state_dict(self, data: dict[str, Any]) -> None:
        """Load fit states from serialized dict."""
        for fid, entry in data.items():
            self.states[fid] = FittedState(
                op_name=entry["op_name"],
                feature_id=fid,
                params=entry.get("params", {}),
                fit_state=entry.get("fit_state", {}),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_inputs(
        self, proposal: FeatureProposal, columnar_data: dict[str, list]
    ) -> Optional[list[list]]:
        """Extract input arrays for a proposal from columnar data."""
        arrays = []
        for fid in proposal.inputs:
            name = fid[4:] if fid.startswith("raw:") else fid
            if name in columnar_data:
                arrays.append(columnar_data[name])
            else:
                return None
        return arrays if arrays else None

    def _fit_operator(
        self, op: FeatureOperator, inputs: list[list], params: dict
    ) -> dict[str, Any]:
        """Dispatch fit to the correct function based on operator name."""
        name = op.name
        # Map operator names to their fit functions
        fit_map = {
            "standardize": self._fit_standardize,
            "robust_scale": self._fit_robust_scale,
            "minmax_scale": self._fit_minmax_scale,
            "winsorize": self._fit_winsorize,
            "quantile_normal": self._fit_quantile_normal,
            "one_hot": self._fit_one_hot,
            "ordinal_encode": self._fit_ordinal_encode,
            "frequency_encode": self._fit_frequency_encode,
            "count_encode": self._fit_count_encode,
            "rare_group": self._fit_rare_group,
        }
        if name in fit_map:
            return fit_map[name](inputs, params)
        # Fallback: no fit needed or unknown
        return {}

    def _transform_with_state(
        self, op: FeatureOperator, inputs: list[list],
        fit_state: dict, params: dict
    ) -> list:
        """Dispatch transform with state based on operator name."""
        name = op.name
        state_key_map = {
            "standardize": ("mean", "std"),
            "robust_scale": ("median", "iqr"),
            "minmax_scale": ("min", "max"),
            "winsorize": ("lower", "upper"),
        }
        if name in state_key_map:
            keys = state_key_map[name]
            state = {k: fit_state.get(k, 0.0) for k in keys}
            if name == "standardize":
                mean, std = state["mean"], state["std"]
                return self._apply_standardize(inputs[0], mean, std)
            if name == "robust_scale":
                median, iqr = state["median"], state["iqr"]
                return self._apply_robust_scale(inputs[0], median, iqr)
            if name == "minmax_scale":
                lo, hi = state["min"], state["max"]
                return self._apply_minmax_scale(inputs[0], lo, hi)
            if name == "winsorize":
                lower, upper = state["lower"], state["upper"]
                return self._apply_winsorize(inputs[0], lower, upper)

        if name == "quantile_normal":
            values = fit_state.get("values", [])
            return self._apply_quantile_normal(inputs[0], values)
        if name == "one_hot":
            categories = fit_state.get("categories", [])
            return self._apply_one_hot(inputs[0], categories)
        if name == "ordinal_encode":
            mapping = fit_state.get("mapping", {})
            return self._apply_ordinal_encode(inputs[0], mapping)
        if name == "frequency_encode":
            freq_map = fit_state.get("frequency_map", {})
            return self._apply_frequency_encode(inputs[0], freq_map)
        if name == "count_encode":
            count_map = fit_state.get("count_map", {})
            return self._apply_count_encode(inputs[0], count_map)
        if name == "rare_group":
            retained = fit_state.get("retained", set())
            return self._apply_rare_group(inputs[0], retained)

        # Fallback: use operator's transform directly (for L1 group/text/etc)
        if op.arity == "unary":
            return op.transform(inputs[0], **params)
        if op.arity == "binary" and len(inputs) >= 2:
            return op.transform(inputs[0], inputs[1], **params)
        return []

    # ------------------------------------------------------------------
    # Fit implementations
    # ------------------------------------------------------------------
    def _fit_standardize(self, inputs: list[list], params: dict) -> dict:
        valid = [x for x in _coerce_to_floats(inputs[0]) if x is not None]
        if len(valid) < 2:
            return {"mean": 0.0, "std": 1.0}
        mean = sum(valid) / len(valid)
        var = sum((x - mean) ** 2 for x in valid) / len(valid)
        return {"mean": mean, "std": max(math.sqrt(var), 1e-12)}

    def _fit_robust_scale(self, inputs: list[list], params: dict) -> dict:
        valid = sorted(x for x in _coerce_to_floats(inputs[0]) if x is not None)
        if not valid:
            return {"median": 0.0, "iqr": 1.0}
        n = len(valid)
        median = valid[n // 2] if n % 2 == 1 else (valid[n // 2 - 1] + valid[n // 2]) / 2
        q1 = valid[n // 4]
        q3 = valid[min((3 * n) // 4, n - 1)]
        iqr = max(q3 - q1, 1e-12)
        return {"median": median, "iqr": iqr}

    def _fit_minmax_scale(self, inputs: list[list], params: dict) -> dict:
        valid = sorted(x for x in _coerce_to_floats(inputs[0]) if x is not None)
        if not valid:
            return {"min": 0.0, "max": 1.0}
        return {"min": valid[0], "max": valid[-1]}

    def _fit_winsorize(self, inputs: list[list], params: dict) -> dict:
        lo_q = params.get("lower_quantile", 0.01)
        hi_q = params.get("upper_quantile", 0.99)
        valid = sorted(x for x in _coerce_to_floats(inputs[0]) if x is not None)
        if not valid:
            return {"lower": 0.0, "upper": 0.0}
        n = len(valid)
        return {
            "lower": valid[max(0, int(lo_q * (n - 1)))],
            "upper": valid[min(n - 1, int(hi_q * (n - 1)))],
        }

    def _fit_quantile_normal(self, inputs: list[list], params: dict) -> dict:
        valid = sorted(x for x in _coerce_to_floats(inputs[0]) if x is not None)
        return {"values": valid, "n": len(valid)}

    def _fit_one_hot(self, inputs: list[list], params: dict) -> dict:
        from collections import Counter
        counts = Counter(str(v) for v in inputs[0] if v is not None and str(v).strip())
        max_cats = params.get("max_categories", 100)
        categories = [c for c, _ in counts.most_common(max_cats)]
        return {"categories": categories}

    def _fit_ordinal_encode(self, inputs: list[list], params: dict) -> dict:
        from collections import Counter
        counts = Counter(str(v) for v in inputs[0] if v is not None and str(v).strip())
        categories = [c for c, _ in counts.most_common()]
        mapping = {c: i for i, c in enumerate(categories)}
        return {"mapping": mapping}

    def _fit_frequency_encode(self, inputs: list[list], params: dict) -> dict:
        from collections import Counter
        n = len(inputs[0])
        counts = Counter(str(v) for v in inputs[0] if v is not None and str(v).strip())
        freq = {c: cnt / max(n, 1) for c, cnt in counts.items()}
        return {"frequency_map": freq}

    def _fit_count_encode(self, inputs: list[list], params: dict) -> dict:
        from collections import Counter
        counts = Counter(str(v) for v in inputs[0] if v is not None and str(v).strip())
        return {"count_map": dict(counts)}

    def _fit_rare_group(self, inputs: list[list], params: dict) -> dict:
        from collections import Counter
        min_count = params.get("min_count", 5)
        counts = Counter(str(v) for v in inputs[0] if v is not None and str(v).strip())
        retained = {c for c, cnt in counts.items() if cnt >= min_count}
        return {"retained": retained}

    # ------------------------------------------------------------------
    # Transform implementations
    # ------------------------------------------------------------------
    def _apply_standardize(self, values: list, mean: float, std: float) -> list:
        return [
            (x - mean) / std if (x := _coerce_to_floats([v])[0]) is not None else None
            for v in values
        ]

    def _apply_robust_scale(self, values: list, median: float, iqr: float) -> list:
        return [
            (x - median) / iqr if (x := _coerce_to_floats([v])[0]) is not None else None
            for v in values
        ]

    def _apply_minmax_scale(self, values: list, lo: float, hi: float) -> list:
        span = hi - lo
        return [
            (x - lo) / span if (x := _coerce_to_floats([v])[0]) is not None and span > 0 else
            (0.5 if span == 0 and (x := _coerce_to_floats([v])[0]) is not None else None)
            for v in values
        ]

    def _apply_winsorize(self, values: list, lower: float, upper: float) -> list:
        return [
            max(lower, min(upper, x))
            if (x := _coerce_to_floats([v])[0]) is not None else None
            for v in values
        ]

    def _apply_quantile_normal(self, values: list, ref_values: list) -> list:
        n = len(ref_values)
        if n < 3:
            return [None] * len(values)
        out = []
        for v in values:
            x_list = _coerce_to_floats([v])
            x = x_list[0] if x_list else None
            if x is None:
                out.append(None)
                continue
            lo, hi = 0, n
            while lo < hi:
                mid = (lo + hi) // 2
                if ref_values[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid
            p = (lo + 0.5) / n
            # Rational approximation to inverse normal CDF
            if p <= 0:
                out.append(-6.0)
            elif p >= 1:
                out.append(6.0)
            elif p > 0.5:
                t = math.sqrt(-2.0 * math.log(1.0 - p))
                c0, c1, c2 = 2.515517, 0.802853, 0.010328
                d1, d2, d3 = 1.432788, 0.189269, 0.001308
                out.append(t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t))
            else:
                t = math.sqrt(-2.0 * math.log(p))
                c0, c1, c2 = 2.515517, 0.802853, 0.010328
                d1, d2, d3 = 1.432788, 0.189269, 0.001308
                out.append(-(t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)))
        return out

    def _apply_one_hot(self, values: list, categories: list) -> list:
        cat_to_idx = {c: i for i, c in enumerate(categories)}
        out = []
        for v in values:
            s = str(v) if v is not None else ""
            row = [False] * len(categories)
            if s in cat_to_idx:
                row[cat_to_idx[s]] = True
            out.append(row)
        return out

    def _apply_ordinal_encode(self, values: list, mapping: dict) -> list:
        out = []
        for v in values:
            s = str(v) if v is not None else ""
            out.append(mapping.get(s, -1))
        return out

    def _apply_frequency_encode(self, values: list, freq_map: dict) -> list:
        return [freq_map.get(str(v) if v is not None else "", 0.0) for v in values]

    def _apply_count_encode(self, values: list, count_map: dict) -> list:
        return [count_map.get(str(v) if v is not None else "", 0) for v in values]

    def _apply_rare_group(self, values: list, retained: set) -> list:
        return [str(v) if v is not None and str(v) in retained else "__RARE__" for v in values]

    def _to_float_list(self, output: list, n: int) -> list:
        """Convert operator output to a float list for the probe/portfolio."""
        floats = []
        for v in output[:n]:
            if v is None:
                floats.append(0.0)
            elif isinstance(v, bool):
                floats.append(1.0 if v else 0.0)
            elif isinstance(v, (int, float)):
                floats.append(float(v))
            elif isinstance(v, list):
                # One-hot or multi-output: sum as proxy
                floats.append(sum(1.0 if x else 0.0 for x in v))
            else:
                floats.append(0.0)
        return floats
