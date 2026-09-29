"""M33 regression tests: W-2 value-level parity gate + canonical wiring.

Before: ``gate_feature_parity`` only checked that expected node ids exist
in the IR — wrong values, wrong lengths, wrong null masks, and NaN all
passed. No caller ever passed expectations, so the gate was always
vacuously green. These tests pin the fixed, executed parity contract.
"""

import sys
import tempfile
import os


from fiae.codegen.compiler import gate_feature_parity
from fiae.codegen.pipeline_ir import (
    build_ir_from_proposals,
    generate_python_code,
)


def _log1p_ir_and_code():
    class P:
        op = "log1p"
        inputs = ["raw:x"]
        params = {}

    ir = build_ir_from_proposals([P()])
    code = generate_python_code(ir)
    return ir, code


_GOOD = [0.0, 0.6931471805599453, 2.3978952727983707, None, 4.61512051684126]
_DATA = {"x": [0.0, 1.0, 10.0, None, 100.0]}


class TestGateValueParity:
    def test_correct_values_pass(self):
        ir, code = _log1p_ir_and_code()
        r = gate_feature_parity(
            ir, expected_features={"n_0": _GOOD},
            exported_code=code, input_data=_DATA,
        )
        assert r.passed, r.message

    def test_wrong_value_fails(self):
        ir, code = _log1p_ir_and_code()
        bad = [0.0, 0.7, 2.3978952727983707, None, 4.61512051684126]
        r = gate_feature_parity(
            ir, expected_features={"n_0": bad},
            exported_code=code, input_data=_DATA,
        )
        assert not r.passed
        assert "value mismatch" in r.message

    def test_wrong_row_count_fails(self):
        ir, code = _log1p_ir_and_code()
        r = gate_feature_parity(
            ir, expected_features={"n_0": _GOOD[:3]},
            exported_code=code, input_data=_DATA,
        )
        assert not r.passed
        assert "row count mismatch" in r.message

    def test_wrong_null_mask_fails(self):
        ir, code = _log1p_ir_and_code()
        bad = [0.0, 0.6931471805599453, 2.3978952727983707, 1.0, 4.61512051684126]
        r = gate_feature_parity(
            ir, expected_features={"n_0": bad},
            exported_code=code, input_data=_DATA,
        )
        assert not r.passed
        assert "null-mask mismatch" in r.message

    def test_exported_code_raising_fails(self):
        ir, code = _log1p_ir_and_code()
        broken = code.replace("math.log1p(v)", "1/0 if v else math.log1p(v)")
        r = gate_feature_parity(
            ir, expected_features={"n_0": _GOOD},
            exported_code=broken, input_data=_DATA,
        )
        assert not r.passed
        assert "raised" in r.message or "mismatch" in r.message

    def test_vacuous_pass_is_flagged(self):
        """Gate without expectations still passes structurally but the
        message must say parity was not exercised (no silent vacuity)."""
        ir, _ = _log1p_ir_and_code()
        r = gate_feature_parity(ir, expected_features=None)
        assert r.passed
        assert "not exercised" in r.message


class TestCanonicalParityWiring:
    def test_sabotaged_fitted_values_fail_export(self):
        """End-to-end: if the fitted runtime's values are corrupted at
        codegen time, the parity gate must fail (11 -> 10 gates)."""
        sys.path.insert(0, os.path.join(
            os.path.dirname(__file__), "..", "examples", "churn"))
        try:
            from run_api import generate_dataset  # type: ignore
        finally:
            sys.path.pop(0)


        tmp = tempfile.mkdtemp()
        csv_path = os.path.join(tmp, "c.csv")
        generate_dataset(csv_path)

        from fiae.codegen.compiler import gate_feature_parity as real_gate
        calls = {"n": 0}

        def corrupting_gate(ir, expected_features=None, **kw):
            calls["n"] += 1
            if expected_features:
                first_key = next(iter(expected_features))
                expected_features[first_key] = [
                    (v + 123.0) if (v is not None and i % 5 == 0) else v
                    for i, v in enumerate(expected_features[first_key])
                ]
            return real_gate(ir, expected_features=expected_features, **kw)

        # patch the local-import target used by phase_codegen
        import fiae.codegen.compiler as cc
        cc.gate_feature_parity = corrupting_gate
        try:
            from fiae.pipeline.canonical import run_canonical_pipeline

            result = run_canonical_pipeline(csv_path, "is_returned")
            total = result.summary()["codegen_gates"]
            assert calls["n"] >= 1, "parity gate must be invoked by codegen"
            assert total == "10/11", (
                f"sabotaged parity must fail the export gate, got {total}"
            )
        finally:
            cc.gate_feature_parity = real_gate

    def test_normal_run_stays_11_of_11(self):
        sys.path.insert(0, os.path.join(
            os.path.dirname(__file__), "..", "examples", "churn"))
        try:
            from run_api import generate_dataset  # type: ignore
        finally:
            sys.path.pop(0)

        from fiae.pipeline.canonical import run_canonical_pipeline

        tmp = tempfile.mkdtemp()
        csv_path = os.path.join(tmp, "c.csv")
        generate_dataset(csv_path)
        result = run_canonical_pipeline(csv_path, "is_returned")
        assert result.summary()["codegen_gates"] == "11/11"
