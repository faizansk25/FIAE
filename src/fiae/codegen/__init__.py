"""Codegen subsystem — doc 09 (Architecture, Codegen, Verification)."""
from .auto_tests import generate_operator_test, generate_all_scaffolds, verify_operator_contract, verify_all_contracts
from .pipeline_ir import PipelineIR, IRNode, build_ir_from_proposals, generate_python_code
from .compiler import compile_pipeline, generate_sklearn_project, CompilerReport
