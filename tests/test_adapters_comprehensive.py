"""Comprehensive adapter tests — SQL, DataFrame, file formats, API, factory.

Complements test_concurrency_warehouse.py (warehouse + concurrency) with:
- SqlAdapter: URI parsing, engines, queries, pushdown, errors
- DataFrameAdapter: pandas (if installed), fallback object
- File adapters: JSON, NDJSON round-trips (Excel/Parquet/Feather if libs)
- ApiAdapter: JSON/text extraction, HTML detection, pagination logic (offline)
- Factory: scheme routing, extension mapping, passthrough
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from fiae.intake import (
    auto_adapter,
    list_supported_sources,
    JsonAdapter,
    NdjsonAdapter,
)
from fiae.intake.adapter_sql import SqlAdapter
from fiae.intake.adapter_api import ApiAdapter
from fiae.intake.adapter_factory import (
    _get_scheme,
    _is_dataframe,
    _looks_like_sql,
    _looks_like_sql_query,
)


# ---------------------------------------------------------------------------
# Factory internals
# ---------------------------------------------------------------------------


class TestFactoryDetection:
    def test_scheme_extraction(self):
        assert _get_scheme("s3://b/k") == "s3"
        assert _get_scheme("HTTPS://x.com") == "https"
        assert _get_scheme("/plain/path.csv") == ""

    def test_sql_detection(self):
        assert _looks_like_sql("sqlite:///a.db")
        assert _looks_like_sql("postgresql://u:p@h/d")
        assert _looks_like_sql("mysql://u:p@h/d")
        assert not _looks_like_sql("s3://bucket/k")
        assert not _looks_like_sql("data.csv")

    def test_sql_query_detection(self):
        assert _looks_like_sql_query("SELECT * FROM t")
        assert _looks_like_sql_query("  select 1")
        assert not _looks_like_sql_query("SELECT".join([]))  # empty
        assert not _looks_like_sql_query("data.csv")

    def test_dataframe_duck_typing(self):
        class FakeDF:
            columns = ["a"]
            shape = (1, 1)

        assert _is_dataframe(FakeDF())
        assert not _is_dataframe("csv")
        assert not _is_dataframe(None)

    def test_list_supported_sources_has_all_categories(self):
        cats = list_supported_sources()
        for key in ("File formats", "Cloud storage", "Databases",
                    "Data warehouses", "DataFrames", "APIs"):
            assert key in cats, f"missing category: {key}"
            assert cats[key], f"empty category: {key}"

    def test_factory_passthrough_adapter(self, tmp_path):
        from fiae.intake import CsvDataSourceAdapter

        p = tmp_path / "pass.csv"
        p.write_text("a\n1\n")
        original = CsvDataSourceAdapter(str(p))
        assert auto_adapter(original) is original


# ---------------------------------------------------------------------------
# SqlAdapter (SQLite — real, no external deps)
# ---------------------------------------------------------------------------


@pytest.fixture()
def sql_db(tmp_path):
    db_path = tmp_path / "adv.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE emp (id INTEGER, name TEXT, salary REAL)")
    conn.executemany(
        "INSERT INTO emp VALUES (?, ?, ?)",
        [(i, f"e{i}", 100.0 * i) for i in range(30)],
    )
    conn.commit()
    conn.close()
    return str(db_path)


class TestSqlAdapter:
    def test_engine_detection(self):
        assert SqlAdapter("sqlite:///x.db")._engine == "sqlite"
        assert SqlAdapter("postgresql://u:p@h/d")._engine == "postgresql"
        assert SqlAdapter("postgres://u:p@h/d")._engine == "postgresql"
        assert SqlAdapter("mysql://u:p@h/d")._engine == "mysql"
        assert SqlAdapter("mssql://u:p@h/d")._engine == "mssql"
        assert SqlAdapter("weird://x")._engine == "unknown"

    def test_uri_credential_parsing(self):
        cs = "postgresql://alice:secret@db.host.com:5432/analytics"
        assert SqlAdapter(cs)._parse_user(cs) == "alice"
        assert SqlAdapter(cs)._parse_password(cs) == "secret"
        assert SqlAdapter(cs)._parse_host(cs) == "db.host.com"
        assert SqlAdapter(cs)._parse_database(cs) == "analytics"

    def test_scan_table(self, sql_db):
        adapter = SqlAdapter(f"sqlite:///{sql_db}", table="emp")
        total = sum(len(b.columns["id"]) for b in adapter.scan())
        assert total == 30

    def test_scan_projection(self, sql_db):
        adapter = SqlAdapter(f"sqlite:///{sql_db}", table="emp")
        batches = list(adapter.scan(projection=["id"]))
        assert set(batches[0].columns.keys()) == {"id"}

    def test_scan_query(self, sql_db):
        adapter = SqlAdapter(
            f"sqlite:///{sql_db}", query="SELECT id, name FROM emp WHERE id < 10")
        total = sum(len(b.columns["id"]) for b in adapter.scan())
        assert total == 10

    def test_estimate_rows(self, sql_db):
        adapter = SqlAdapter(f"sqlite:///{sql_db}", table="emp")
        assert adapter.estimate_rows() == 30

    def test_requires_table_or_query(self, sql_db):
        adapter = SqlAdapter(f"sqlite:///{sql_db}")
        # Single-table DB: table auto-resolves ergonomically
        assert adapter._resolve_table() == "emp"
        list(adapter.scan())  # must not raise

    def test_multi_table_requires_table(self, tmp_path):
        db_path = tmp_path / "multi.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE a (x INTEGER)")
        conn.execute("CREATE TABLE b (x INTEGER)")
        conn.commit()
        conn.close()
        adapter = SqlAdapter(f"sqlite:///{db_path}")
        with pytest.raises(ValueError):
            list(adapter.scan())

    def test_missing_table_raises(self, sql_db):
        adapter = SqlAdapter(f"sqlite:///{sql_db}", table="nope")
        with pytest.raises(Exception):
            list(adapter.scan())

    def test_pushdown_supported(self):
        assert SqlAdapter("sqlite:///x.db").supports_pushdown()

    def test_fingerprint_table_vs_query(self, sql_db):
        a = SqlAdapter(f"sqlite:///{sql_db}", table="emp")
        b = SqlAdapter(f"sqlite:///{sql_db}", query="SELECT * FROM emp")
        assert a.fingerprint_material() != b.fingerprint_material()


# ---------------------------------------------------------------------------
# DataFrame adapter (pandas if available, else duck-typed)
# ---------------------------------------------------------------------------


class TestDataFrameAdapter:
    def test_pandas_roundtrip(self):
        pd = pytest.importorskip("pandas")
        from fiae.intake import DataFrameAdapter

        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        adapter = DataFrameAdapter(df)
        batches = list(adapter.scan())
        total_a = sum(len(b.columns["a"]) for b in batches)
        assert total_a == 3
        assert batches[0].columns["b"] == ["x", "y", "z"]

    def test_factory_routes_dataframe(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"a": [1]})
        adapter = auto_adapter(df)
        assert type(adapter).__name__ == "DataFrameAdapter"


# ---------------------------------------------------------------------------
# File adapters: JSON / NDJSON
# ---------------------------------------------------------------------------


class TestFileAdapters:
    def test_json_array(self, tmp_path):
        p = tmp_path / "d.json"
        p.write_text(json.dumps([
            {"id": 1, "name": "a"}, {"id": 2, "name": "b"},
        ]))
        adapter = JsonAdapter(str(p))
        batches = list(adapter.scan())
        total = sum(len(b.columns["id"]) for b in batches)
        assert total == 2

    def test_ndjson(self, tmp_path):
        p = tmp_path / "d.ndjson"
        p.write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n')
        adapter = NdjsonAdapter(str(p))
        total = sum(len(b.columns["a"]) for b in adapter.scan())
        assert total == 3

    def test_factory_routes_json(self, tmp_path):
        p = tmp_path / "d.json"
        p.write_text('[{"a": 1}]')
        assert type(auto_adapter(str(p))).__name__ == "JsonAdapter"

    def test_factory_routes_ndjson(self, tmp_path):
        p = tmp_path / "d.jsonl"
        p.write_text('{"a": 1}\n')
        assert type(auto_adapter(str(p))).__name__ == "NdjsonAdapter"

    def test_json_null_and_nested_values(self, tmp_path):
        p = tmp_path / "d.json"
        p.write_text(json.dumps([
            {"a": None, "b": [1, 2], "c": {"k": "v"}},
        ]))
        batch = next(iter(JsonAdapter(str(p)).scan()))
        assert batch.columns["a"] == [None]  # null preserved


# ---------------------------------------------------------------------------
# ApiAdapter: offline logic tests (no network)
# ---------------------------------------------------------------------------


class TestApiAdapterOffline:
    def test_json_path_extraction(self):
        a = ApiAdapter("https://x/api", json_path="data.users")
        data = {"data": {"users": [{"id": 1}, {"id": 2}]}}
        assert len(a._extract_from_json(data)) == 2

    def test_json_path_missing_returns_empty(self):
        a = ApiAdapter("https://x/api", json_path="data.missing")
        assert a._extract_from_json({"data": {}}) == []

    def test_json_path_index(self):
        a = ApiAdapter("https://x/api", json_path="results.0")
        data = {"results": [{"id": 7}]}
        assert a._extract_from_json(data) == [{"id": 7}]

    def test_top_level_list_extraction(self):
        a = ApiAdapter("https://x/api")
        assert len(a._extract_from_json([{"id": 1}, {"id": 2}])) == 2

    def test_cursor_extraction_variants(self):
        a = ApiAdapter("https://x/api")
        assert a._extract_cursor({"next_cursor": "abc"}) == "abc"
        assert a._extract_cursor({"nextPageToken": "tok"}) == "tok"
        assert a._extract_cursor({"cursor": "c"}) == "c"
        assert a._extract_cursor({}) is None

    def test_text_extraction_csv(self):
        a = ApiAdapter("https://x/api")
        records = a._extract_from_text("id,name\n1,alpha\n2,beta\n")
        assert records == [
            {"id": "1", "name": "alpha"},
            {"id": "2", "name": "beta"},
        ]

    def test_text_extraction_headerless(self):
        a = ApiAdapter("https://x/api")
        records = a._extract_from_text("1,2\n3,4\n")
        assert records == [{"col_0": "1", "col_1": "2"},
                           {"col_0": "3", "col_1": "4"}]

    def test_source_id_and_fingerprint(self):
        a1 = ApiAdapter("https://x/api", params={"p": 1})
        a2 = ApiAdapter("https://x/api", params={"p": 1})
        a3 = ApiAdapter("https://x/api", params={"p": 2})
        assert a1.source_id() == "api|https://x/api"
        assert a1.fingerprint_material() == a2.fingerprint_material()
        assert a1.fingerprint_material() != a3.fingerprint_material()

    def test_scan_projection(self, monkeypatch):
        a = ApiAdapter("https://x/api")
        monkeypatch.setattr(
            a, "_fetch_all_records",
            lambda: [{"id": 1, "secret": "x"}, {"id": 2, "secret": "y"}])
        batches = list(a.scan(projection=["id"]))
        assert set(batches[0].columns.keys()) == {"id"}
        assert batches[0].columns["id"] == ["1", "2"]


# ---------------------------------------------------------------------------
# Cloud adapter unit logic (no network)
# ---------------------------------------------------------------------------


class TestCloudAdapterOffline:
    def test_provider_detection(self):
        from fiae.intake import CloudAdapter

        assert CloudAdapter("s3://b/k")._provider == "s3"
        assert CloudAdapter("gs://b/k")._provider == "gcs"
        assert CloudAdapter("az://c/k")._provider == "azure"
        assert CloudAdapter("abfss://c/k")._provider == "azure"
        assert CloudAdapter("wasbs://c/k")._provider == "azure"

    def test_bucket_key_parsing(self):
        from fiae.intake import CloudAdapter

        a = CloudAdapter("s3://my-bucket/path/to/file.csv")
        assert a._bucket == "my-bucket"
        assert a._key == "path/to/file.csv"


# ---------------------------------------------------------------------------
# Excel / Parquet / Feather (optional deps — skip if absent)
# ---------------------------------------------------------------------------


class TestOptionalFileFormats:
    def test_parquet_roundtrip(self, tmp_path):
        pa = pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq
        from fiae.intake import ParquetAdapter

        p = tmp_path / "d.parquet"
        table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        pq.write_table(table, str(p))
        batches = list(ParquetAdapter(str(p)).scan())
        total = sum(len(b.columns["a"]) for b in batches)
        assert total == 3

    def test_factory_routes_parquet(self, tmp_path):
        pa = pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq

        p = tmp_path / "d.parquet"
        pq.write_table(pa.table({"a": [1]}), str(p))
        assert type(auto_adapter(str(p))).__name__ == "ParquetAdapter"

    def test_feather_roundtrip(self, tmp_path):
        pa = pytest.importorskip("pyarrow")
        import pyarrow.feather as feather
        from fiae.intake import FeatherAdapter

        p = tmp_path / "d.feather"
        feather.write_feather(pa.table({"v": [1.0, 2.0]}), str(p))
        total = sum(
            len(b.columns["v"]) for b in FeatherAdapter(str(p)).scan())
        assert total == 2
