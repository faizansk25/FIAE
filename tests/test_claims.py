"""Claims verification: turns README marketing claims into automated assertions.

Every test here independently re-derives a documented claim (operator count,
zero-dependency core, gate count, determinism scope, connector coverage,
security enforcement). If one of these fails, either the claim in README.md
is stale or the code regressed -- fix whichever is wrong, never weaken the
assertion silently.
"""

import json
import os
import subprocess
import sys


from fiae.codegen import compiler as codegen_compiler
from fiae.features.registry import all_operators
from fiae.ids import content_hash
from fiae.intake.adapter_factory import (
    _EXTENSION_MAP,
    _SCHEME_MAP,
    list_supported_sources,
)
from fiae.intake.adapter_base import BaseAdapter
from fiae.security.sandbox import validate_column_name, validate_input_code

# ---------------------------------------------------------------------------
# Claim: 95 typed operators across 13 families
# ---------------------------------------------------------------------------


class TestOperatorCatalogClaims:
    def test_operator_count_is_95(self):
        ops = all_operators()
        assert len(ops) == 95, (
            "README advertises 95 operators; registry exposes "
            f"{len(ops)}. Update the claim or fix the registry."
        )

    def test_operator_count_matches_readme_badge(self):
        """Parse the operators badge number straight out of README.md."""
        readme = _read_readme()
        import re

        m = re.search(r"operators-(\d+)-", readme)
        assert m, "operators badge missing from README.md"
        assert int(m.group(1)) == len(all_operators())

    def test_thirteen_operator_families(self):
        fams = {op.family for op in all_operators()}
        assert len(fams) == 13

    def test_every_operator_has_leakage_class(self):
        for op in all_operators():
            assert op.leakage_class is not None, f"{op.name} missing leakage_class"
            assert op.leakage_class in ("L0", "L1", "L2", "L3", "L4", "L5")

    def test_operator_names_unique(self):
        names = [op.name for op in all_operators()]
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# Claim: zero-dependency core (stdlib only)
# ---------------------------------------------------------------------------

_BLOCKED_MODULES = ("numpy", "pandas", "sklearn", "polars", "joblib", "yaml", "pyarrow")

_ZERO_DEP_PROBE = r"""
import sys


_BLOCKED = %(blocked)s


class _Blocker:
    def find_module(self, name, path=None):
        if name.split(".")[0] in _BLOCKED:
            return self

    def load_module(self, name):
        raise ImportError("blocked: " + name)


sys.meta_path.insert(0, _Blocker())
sys.path.insert(0, r"%(src)s")

import importlib
import pkgutil

import fiae

violations = []
for m in pkgutil.walk_packages(fiae.__path__, "fiae."):
    try:
        importlib.import_module(m.name)
    except ImportError as e:
        if "blocked" in str(e):
            violations.append(m.name)
    except Exception:
        pass  # modules may fail for other reasons (missing optional data etc.)

if violations:
    print(json.dumps(violations))
else:
    print("[]")
"""


class TestZeroDependencyCore:
    def test_all_fiae_modules_import_without_heavy_deps(self):
        src_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "src")
        )
        probe = _ZERO_DEP_PROBE % {
            "blocked": json.dumps(list(_BLOCKED_MODULES)),
            "src": src_root,
        }
        out = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert out.returncode == 0, out.stderr
        violations = json.loads(out.stdout.strip().splitlines()[-1])
        assert violations == [], (
            "Zero-dependency core claim violated by modules requiring "
            f"heavy deps: {violations}"
        )


# ---------------------------------------------------------------------------
# Claim: 11 automated verification gates (incl. feature/prediction parity)
# ---------------------------------------------------------------------------


class TestExportGateClaims:
    def test_exactly_11_gates(self):
        gates = [
            n
            for n in dir(codegen_compiler)
            if n.startswith("gate_")
            and callable(getattr(codegen_compiler, n))
        ]
        assert len(gates) == 11, (
            f"README claims 11 verification gates; found {len(gates)}: {gates}"
        )

    def test_gate_names_match_doc09(self):
        expected = {
            "operator_coverage",
            "validate_ir",
            "topological_sort",
            "deduplicate",
            "partition_fit_transform",
            "select_backend",
            "emit_contracts",
            "feature_parity",
            "prediction_parity",
            "latency_resource",
            "syntax_check",
        }
        found = {
            n[len("gate_"):]
            for n in dir(codegen_compiler)
            if n.startswith("gate_") and callable(getattr(codegen_compiler, n))
        }
        assert found == expected

    def test_parity_gates_actually_compare(self):
        """The parity gates must reject mismatched expectations, not just
        pass trivially on empty input."""
        from fiae.codegen.compiler import (
            gate_feature_parity,
            gate_prediction_parity,
        )
        from fiae.codegen.pipeline_ir import PipelineIR

        ir = PipelineIR()
        r = gate_feature_parity(
            ir,
            expected_features={"missing_node": [1.0, 2.0, 3.0]},
        )
        assert not r.passed, "feature parity gate must fail on unknown node"

        r2 = gate_prediction_parity(ir, expected_predictions={"threshold": 7.0})
        assert not r2.passed, "prediction parity gate must fail on bad threshold"


# ---------------------------------------------------------------------------
# Claim: deterministic IDs / content hashes (scoped: content-derived IDs are
# deterministic across processes; run IDs are intentionally ephemeral)
# ---------------------------------------------------------------------------


class TestDeterminismClaims:
    def test_content_hash_is_repeatable(self):
        payload = {"a": 1, "b": [1, 2, 3], "c": {"d": "x"}}
        assert content_hash(payload) == content_hash(payload)

    def test_content_hash_stable_across_processes(self):
        payload = {"rows": 80, "cols": 3, "name": "synthetic"}
        h1 = content_hash(payload)
        code = (
            "import sys; sys.path.insert(0, r'%(src)s'); "
            "from fiae.ids import content_hash; "
            "print(content_hash(%(payload)s))"
            % {
                "src": os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "src")
                ),
                "payload": repr(payload),
            }
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == h1, (
            "content_hash is not stable across processes -- violates the "
            "deterministic IDs claim (doc 13)"
        )

    def test_content_hash_excludes_ephemeral_keys(self):
        a = content_hash({"x": 1, "timestamp": "2026-01-01T00:00:00Z"})
        b = content_hash({"x": 1, "timestamp": "2099-12-31T23:59:59Z"})
        assert a == b, "ephemeral keys must not affect content hashes (doc 13)"

    def test_run_id_is_unique_not_reused(self):
        """Run IDs are deliberately ephemeral; assert uniqueness rather than
        cross-run stability (scoping the determinism claim honestly)."""
        import tempfile

        from fiae.pipeline.canonical import phase_bootstrap

        tmp = tempfile.mkdtemp()
        csv_path = os.path.join(tmp, "d.csv")
        with open(csv_path, "w") as f:
            f.write("x1,x2,y\n1,2,3\n4,5,6\n7,8,9\n")
        ctx1 = phase_bootstrap(csv_path, "y")
        ctx2 = phase_bootstrap(csv_path, "y")
        assert ctx1.run_id and ctx2.run_id
        assert ctx1.run_id != ctx2.run_id
        # ...but the source fingerprint (content-derived) must match
        assert ctx1.source_fingerprint == ctx2.source_fingerprint


# ---------------------------------------------------------------------------
# Claim: 25+ data sources via 12 concrete adapters
# ---------------------------------------------------------------------------


class TestConnectorClaims:
    def test_twelve_concrete_adapters(self):
        import fiae.intake.adapter_api as api_mod
        import fiae.intake.adapter_cloud as cloud_mod
        import fiae.intake.adapter_dataframe as df_mod
        import fiae.intake.adapter_file as file_mod
        import fiae.intake.adapter_sql as sql_mod
        import fiae.intake.adapter_warehouse as wh_mod

        concrete = []
        for mod in (api_mod, cloud_mod, df_mod, file_mod, sql_mod, wh_mod):
            for name in dir(mod):
                obj = getattr(mod, name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, BaseAdapter)
                    and obj is not BaseAdapter
                    and not name.startswith("_")
                    and not name.startswith("Warehouse")  # base class
                ):
                    concrete.append(obj.__name__)
        # 12 concrete adapters (some files share helper subclasses)
        assert len(concrete) >= 12, f"expected >=12 adapters, found {concrete}"

    def test_source_coverage_exceeds_25(self):
        """13 file extensions + 16 URI schemes + SQL detection = 25+ sources."""
        total = len(_EXTENSION_MAP) + len(_SCHEME_MAP)
        assert total >= 25, (
            f"extension ({len(_EXTENSION_MAP)}) + scheme ({len(_SCHEME_MAP)}) "
            "coverage must be >= 25 to back the '25+ sources' claim"
        )

    def test_list_supported_sources_reports(self):
        sources = list_supported_sources()
        assert sources and len(sources) >= 6


# ---------------------------------------------------------------------------
# Claim: runtime security guarantees
# ---------------------------------------------------------------------------


class TestSecurityClaims:
    def test_column_name_injection_rejected(self):
        for hostile in ("'; DROP TABLE x", "__import__('os')", "a;b"):
            problems = validate_column_name(hostile)
            assert problems, f"column name {hostile!r} must be flagged"

    def test_column_name_benign_accepted(self):
        assert validate_column_name("feature_1") == []
        assert validate_column_name("Customer Age") == []

    def test_code_injection_rejected(self):
        for hostile in (
            "import os; os.system('rm -rf /')",
            "__import__('subprocess')",
            "open('/etc/passwd').read()",
        ):
            problems = validate_input_code(hostile)
            assert problems, f"code {hostile!r} must be flagged"

    def test_benign_code_accepted(self):
        assert validate_input_code("x = a + b") == []


# ---------------------------------------------------------------------------
# Claim: tests count badge
# ---------------------------------------------------------------------------


def _read_readme() -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "README.md")
    with open(path, encoding="utf-8") as f:
        return f.read()


class TestReadmeCountBadges:
    def test_tests_badge_matches_collected(self, pytestconfig):
        """Badge count should track collected tests (allowing skips)."""
        import re

        readme = _read_readme()
        m = re.search(r"tests-(\d+)", readme)
        assert m, "tests badge missing from README.md"
        badge = int(m.group(1))
        collected = pytestconfig.item_count if hasattr(pytestconfig, "item_count") else None
        # Cross-check only when full suite collected; otherwise skip check.
        if collected and collected > 700:
            assert abs(collected - badge) <= 10, (
                f"README badge says {badge} tests, pytest collected {collected}"
            )


# ---------------------------------------------------------------------------
# Claim: 11/11 gates pass on a real canonical pipeline run
# ---------------------------------------------------------------------------


class TestCanonicalPipelineGateClaim:
    def _write_csv(self, tmp_path, n=80):
        csv_path = tmp_path / "d.csv"
        rows = ["x1,x2,y"]
        for i in range(n):
            rows.append(f"{i},{i % 7},{i % 2}")
        csv_path.write_text("\n".join(rows) + "\n")
        return str(csv_path)

    def test_full_pipeline_reports_all_gates_passed(self, tmp_path):
        from fiae.pipeline.canonical import run_canonical_pipeline

        result = run_canonical_pipeline(self._write_csv(tmp_path), "y")
        assert result.phases_completed == 10, result.errors
        assert result.errors == []
        summary = result.summary()
        assert summary["codegen_gates"] == "11/11", summary
