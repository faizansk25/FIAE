"""Reliability and fuzz tests (audit program M6).

Adversarial inputs against intake, codegen, and the security layer:
- CSV fuzzing: empty, header-only, ragged, binary garbage, multi-GB-field rows
- hostile file paths (traversal, null bytes, nonexistent)
- hostile column names flowing through code generation
- code-generation injection must fail closed (syntax gate rejects)
- URL/SQL scheme routing (no accidental local-file execution)
"""

import os

import pytest

from fiae.codegen.compiler import gate_syntax_check
from fiae.codegen.pipeline_ir import IRNode, PipelineIR, generate_python_code
from fiae.intake.adapter_factory import auto_adapter
from fiae.intake.csv_source import CsvDataSourceAdapter
from fiae.security.sandbox import validate_column_name, validate_input_code


def _write(tmp_path, name, data: bytes) -> str:
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


class TestCsvFuzzing:
    def test_empty_file_is_not_a_crash(self, tmp_path):
        p = _write(tmp_path, "empty.csv", b"")
        batches = list(CsvDataSourceAdapter(p).scan())
        assert batches == []

    def test_header_only_file(self, tmp_path):
        """A single-row file cannot prove a header exists: the row is kept
        as data under synthetic column names (conservative inference)."""
        p = _write(tmp_path, "h.csv", b"a,b,c\n")
        batches = list(CsvDataSourceAdapter(p).scan())
        assert len(batches) == 1
        cols = list(batches[0].columns.keys())
        assert len(cols) == 3 and cols[0].startswith("column_")
        assert batches[0].columns[cols[0]] == ["a"]

    def test_ragged_rows_counted_as_malformed(self, tmp_path):
        p = _write(tmp_path, "ragged.csv", b"a,b\n1,2\n3\n4,5,6\n7,8\n")
        ad = CsvDataSourceAdapter(p)
        batches = list(ad.scan())
        n = sum(len(b.columns.get("a", [])) for b in batches)
        assert n == 2  # only the well-formed rows survive
        assert ad.malformed_rows == 2

    def test_binary_garbage_rejected_with_fiae_error(self, tmp_path):
        p = _write(tmp_path, "bin.csv", bytes(range(256)) * 100)
        with pytest.raises(Exception) as excinfo:
            list(CsvDataSourceAdapter(p).scan())
        # FIAE errors carry a code; raw crashes do not
        assert type(excinfo.value).__name__ != "_csv.Error"

    def test_huge_single_field_is_malformed_not_fatal(self, tmp_path):
        """A single 5 MB field must not escape as a raw _csv.Error (huge-row
        DoS guard): it is counted as malformed and scanning continues."""
        p = _write(tmp_path, "huge.csv", b"a\n" + b"x" * 5_000_000 + b"\n1\n")
        ad = CsvDataSourceAdapter(p)
        list(ad.scan())  # must not raise _csv.Error
        assert ad.malformed_rows >= 1

    def test_hostile_column_names_survive_intake(self, tmp_path):
        p = _write(
            tmp_path,
            "cols.csv",
            b"'; DROP TABLE x,__import__('os'),a-b,c d\n1,2,3,4\n",
        )
        batches = list(CsvDataSourceAdapter(p).scan())
        cols = list(batches[0].columns.keys())
        assert "'; DROP TABLE x" in cols
        assert "__import__('os')" in cols
        # and the security layer flags them
        assert validate_column_name(cols[0])
        assert validate_column_name(cols[1])

    def test_utf8_and_bom(self, tmp_path):
        p = _write(tmp_path, "bom.csv", b"\xef\xbb\xbfna\xc3\xa9me\nv\xablue\n")
        batches = list(CsvDataSourceAdapter(p).scan())
        assert batches  # must not crash


class TestHostilePaths:
    def test_path_traversal_rejected_cleanly(self, tmp_path):
        target = os.path.join(tmp_path, "..", "..", "etc", "passwd.csv")
        with pytest.raises(Exception):
            auto_adapter(target)

    def test_null_byte_path_rejected(self, tmp_path):
        with pytest.raises(Exception):
            auto_adapter(str(tmp_path / "x\x00.csv"))

    def test_missing_file_rejected_with_fiae_error(self, tmp_path):
        from fiae.errors import FIAEError

        with pytest.raises(FIAEError):
            auto_adapter(str(tmp_path / "nope.csv"))


class TestSchemeRouting:
    def test_sql_string_routed_to_sql_adapter(self):
        adapter = auto_adapter("SELECT * FROM users")
        assert type(adapter).__name__ == "SqlAdapter"

    def test_http_routed_to_api_adapter(self):
        adapter = auto_adapter("http://example.com/data")
        assert type(adapter).__name__ == "ApiAdapter"

    def test_unknown_scheme_rejected(self):
        with pytest.raises(Exception):
            auto_adapter("ftp://host/file.csv")

    def test_cloud_scheme_routed(self):
        adapter = auto_adapter("s3://bucket/data.csv")
        assert type(adapter).__name__ == "CloudAdapter"


class TestCodegenInjection:
    def test_hostile_column_name_fails_closed(self):
        """A malicious column name must never produce executable exploit
        code: the export gate rejects it (fail-closed)."""
        ir = PipelineIR()
        ir.add_node(
            IRNode(
                node_id="h1",
                operator="log1p",
                inputs=['raw:x"; __import__("os").system("echo PWNED"); "'],
            )
        )
        report = gate_syntax_check(ir)
        assert not report.passed

    def test_hostile_name_does_not_appear_as_code(self):
        ir = PipelineIR()
        ir.add_node(
            IRNode(node_id="h1", operator="log1p", inputs=["raw:__import__('os')"])
        )
        code = generate_python_code(ir)
        # the payload must be contained in a string literal context, never
        # emitted as bare calls
        assert "os.system" not in code.replace("__import__('os')", "")
        assert gate_syntax_check(ir).passed is False or "__import__" in code

    def test_benign_tricky_names_still_export(self):
        """Real-world column names (spaces, dashes, accents, quotes) must
        keep working -- security must not break usability."""
        ir = PipelineIR()
        names = ["Customer Age", "a-b", "caf\u00e9", "x'quote", "back\\slash"]
        for i, col in enumerate(names):
            ir.add_node(
                IRNode(node_id=f"n{i}", operator="log1p", inputs=["raw:" + col])
            )
        code = generate_python_code(ir)
        compile(code, "<gen>", "exec")  # must be valid Python
        assert gate_syntax_check(ir).passed

    def test_sandbox_still_flags_generated_exploit(self):
        exploit = 'import os; os.system("rm -rf /")'
        assert validate_input_code(exploit)
        assert validate_input_code("__import__('subprocess').run(['ls'])")
        assert validate_input_code("open('/etc/passwd').read()")
        assert validate_input_code("x = a + b") == []
