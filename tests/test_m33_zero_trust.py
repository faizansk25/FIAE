"""M33 regression tests: zero-trust audit fixes (W-8, W-9).

Pins the adversarial-probe findings: security validators must catch
obfuscated/aliased attacks, and DAG verification must report duplicate
node_ids precisely instead of a misleading cycle error.
"""



from fiae.security.sandbox import validate_input_code
from fiae.codegen.pipeline_ir import PipelineIR, IRNode


class TestW9ValidatorHardening:
    def test_getattr_escape_blocked(self):
        assert validate_input_code('x = getattr(builtins, "ev"+"al")("1+1")')

    def test_builtin_aliasing_blocked(self):
        assert validate_input_code('f = open; f("/etc/passwd")')
        assert validate_input_code("g = globals; g()['__builtins__']")
        assert validate_input_code("h = getattr")

    def test_input_call_blocked(self):
        assert validate_input_code('x = input("password: ")')

    def test_submodule_import_blocked(self):
        assert validate_input_code("import os.path")
        assert validate_input_code("import urllib.request")

    def test_benign_code_still_passes(self):
        assert validate_input_code("x = a + b") == []
        assert validate_input_code("y = max(a, b) + len(c)") == []
        assert validate_input_code("z = [v for v in items if v > 0]") == []


class TestW8DuplicateNodeIds:
    def test_duplicate_ids_reported_precisely(self):
        ir = PipelineIR()
        ir.add_node(IRNode(node_id="same", operator="log1p", inputs=["raw:x"]))
        ir.add_node(IRNode(node_id="same", operator="abs", inputs=["raw:x"]))
        errs = ir.verify_dag()
        assert any("Duplicate node_id" in e for e in errs), errs

    def test_real_cycle_still_detected(self):
        ir = PipelineIR()
        ir.add_node(IRNode(node_id="a", operator="log1p", inputs=["raw:x", "b"]))
        ir.add_node(IRNode(node_id="b", operator="abs", inputs=["a"]))
        errs = ir.verify_dag()
        assert any("Cycle detected" in e for e in errs), errs

    def test_clean_dag_has_no_errors(self):
        ir = PipelineIR()
        ir.add_node(IRNode(node_id="a", operator="log1p", inputs=["raw:x"]))
        ir.add_node(IRNode(node_id="b", operator="abs", inputs=["a"]))
        assert ir.verify_dag() == []


class TestW2GateStillEnforced:
    def test_existing_node_wrong_values_fail(self):
        """The original zero-trust finding: wrong values on an existing
        node must fail the parity gate (only possible with exported_code
        + input_data; structural-only expectations remain node checks)."""
        from fiae.codegen.compiler import gate_feature_parity

        class P:
            op = "log1p"
            inputs = ["raw:x"]
            params = {}

        ir = build_ir([P()])
        code = generate_code(ir)
        data = {"x": [0.0, 1.0, 10.0, None, 100.0]}
        expected = {"n_0": [0.0, 0.7, 2.3978952727983707, None, 4.61512051684126]}
        r = gate_feature_parity(
            ir, expected_features=expected,
            exported_code=code, input_data=data,
        )
        assert not r.passed


def build_ir(proposals):
    from fiae.codegen.pipeline_ir import build_ir_from_proposals

    return build_ir_from_proposals(proposals)


def generate_code(ir):
    from fiae.codegen.pipeline_ir import generate_python_code

    return generate_python_code(ir)
