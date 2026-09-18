"""Canonical pipeline module (doc 15)."""
from .canonical import (
    run_canonical_pipeline,
    CanonicalResult,
    RunContext,
    phase_bootstrap,
    phase_intake,
    phase_validate,
    phase_splits,
    phase_generate,
    phase_funnel,
    phase_hpo,
    phase_ensemble,
    phase_evaluate,
    phase_codegen,
)
