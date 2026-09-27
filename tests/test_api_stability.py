"""Public API stability tests (audit program M2).

Pins the documented public entry points and their signatures. A change here
is a breaking change and must ship with a major-version bump -- if this test
fails, either update the pin deliberately (and document the break) or restore
the signature.
"""

import inspect

from fiae.ids import canonical_json, content_hash, feature_id_for
from fiae.intake.adapter_factory import auto_adapter, list_supported_sources
from fiae.pipeline.canonical import run_canonical_pipeline
from fiae.report import build_report
from fiae.security.sandbox import (
    run_in_sandbox,
    validate_column_name,
    validate_input_code,
)
from fiae.features.registry import all_operators, get_operator

# Entry point -> expected signature string (canonical inspect rendering).
EXPECTED_SIGNATURES = {
    all_operators: "() -> 'list[FeatureOperator]'",
    get_operator: "(name: 'str') -> 'FeatureOperator'",
    run_canonical_pipeline: "(source_path: 'str', target: 'str', config: 'dict | None' = None) -> 'CanonicalResult'",
    content_hash: "(obj: 'Any', *, exclude_keys: 'Optional[frozenset[str]]' = None) -> 'str'",
    canonical_json: "(obj: 'Any', *, exclude_keys: 'Optional[frozenset[str]]' = None) -> 'str'",
    feature_id_for: "(node_material: 'Any') -> 'str'",
    auto_adapter: "(source: 'Any', *, target: 'Optional[str]' = None, **kwargs: 'Any') -> 'DataSourceAdapter'",
    list_supported_sources: "() -> 'dict[str, list[str]]'",
    build_report: "(source: 'str', target: 'Optional[str]' = None, table: 'Optional[str]' = None, max_rows: 'int' = 20000) -> 'dict'",
    run_in_sandbox: "(fn: 'Callable', args: 'tuple' = (), kwargs: 'Optional[dict]' = None, policy: 'Optional[SandboxPolicy]' = None) -> 'SandboxResult'",
    validate_column_name: "(name: 'str') -> 'list[str]'",
    validate_input_code: "(code: 'str', policy: 'Optional[SandboxPolicy]' = None) -> 'list[str]'",
}


class TestPublicApiSurface:
    def test_entry_points_importable(self):
        # Import side-effect alone proves the modules exist and are wired.
        assert all_operators() and len(all_operators()) == 95

    def test_signatures_are_stable(self):
        broken = []
        for fn, expected in EXPECTED_SIGNATURES.items():
            actual = str(inspect.signature(fn))
            if actual != expected:
                broken.append(f"{fn.__name__}: {actual} (expected {expected})")
        assert broken == [], (
            "Public API signature drift detected:\n  " + "\n  ".join(broken)
        )

    def test_version_is_semver_like(self):
        import fiae

        parts = fiae.__version__.split(".")
        assert len(parts) == 3 and all(p.isdigit() for p in parts), (
            f"version {fiae.__version__!r} must be MAJOR.MINOR.PATCH"
        )

    def test_result_contracts_keep_documented_keys(self):
        """summary() keys are consumed by CLI/GUI/docs -- removing one is
        a breaking change."""
        from fiae.pipeline.canonical import CanonicalResult

        keys = set(CanonicalResult().summary())
        assert {
            "run_id", "phases_completed", "total_time_s", "portfolio_size",
            "best_model", "best_score", "codegen_gates", "primary_metric",
            "primary_value", "export_path", "errors",
        } <= keys
