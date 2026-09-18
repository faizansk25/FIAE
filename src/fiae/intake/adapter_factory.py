"""Data source adapter factory with auto-detection.

Given any supported data source (file path, URI, DataFrame, connection string),
automatically detect the right adapter and return a configured instance.

Usage:
    from fiae.intake import auto_adapter

    # Auto-detect from path
    adapter = auto_adapter("data.csv")
    adapter = auto_adapter("s3://bucket/data.parquet")
    adapter = auto_adapter("postgresql://user:pass@localhost/db")
    adapter = auto_adapter(pandas_df)
    adapter = auto_adapter("https://api.example.com/data")

    # List supported sources
    supported = list_supported_sources()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .base import DataSourceAdapter


# ── Registry of supported sources ───────────────────────────────────────

_EXTENSION_MAP = {
    ".csv": "csv",
    ".tsv": "csv",
    ".txt": "csv",
    ".xlsx": "excel",
    ".xls": "excel",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".json": "json",
    ".jsonl": "ndjson",
    ".ndjson": "ndjson",
    ".feather": "feather",
    ".arrow": "feather",
    ".ipc": "feather",
}

_SCHEME_MAP = {
    "s3": "cloud",
    "gs": "cloud",
    "az": "cloud",
    "wasbs": "cloud",
    "abfss": "cloud",
    "sqlite": "sql",
    "postgresql": "sql",
    "postgres": "sql",
    "mysql": "sql",
    "mssql": "sql",
    "sqlserver": "sql",
    "bigquery": "warehouse",
    "snowflake": "warehouse",
    "redshift": "warehouse",
    "http": "api",
    "https": "api",
}

_SQL_KEYWORDS = {"SELECT", "FROM", "WHERE", "INSERT", "UPDATE", "DELETE", "CREATE"}


def auto_adapter(
    source: Any,
    *,
    target: Optional[str] = None,
    **kwargs: Any,
) -> DataSourceAdapter:
    """Auto-detect and create the right adapter for any data source.

    Parameters
    ----------
    source : str | Path | DataFrame | DataSourceAdapter
        The data source to wrap.
    target : str, optional
        Target column name (passed to the adapter).
    **kwargs
        Additional arguments passed to the specific adapter.

    Returns
    -------
    DataSourceAdapter
        A configured adapter instance.
    """
    # Already an adapter
    if isinstance(source, DataSourceAdapter):
        return source

    # DataFrame
    if _is_dataframe(source):
        from .adapter_dataframe import DataFrameAdapter
        return DataFrameAdapter(source, **kwargs)

    # String sources
    source_str = str(source)

    # SQL connection string
    if _looks_like_sql(source_str):
        from .adapter_sql import SqlAdapter
        return SqlAdapter(source_str, **kwargs)

    # Warehouse connection string
    scheme = _get_scheme(source_str)
    if scheme in ("bigquery", "snowflake", "redshift"):
        from .adapter_warehouse import (
            BigQueryAdapter, SnowflakeAdapter, RedshiftAdapter)
        if scheme == "bigquery":
            return BigQueryAdapter(source_str, **kwargs)
        if scheme == "snowflake":
            return SnowflakeAdapter(source_str, **kwargs)
        return RedshiftAdapter(source_str, **kwargs)

    # SQL query
    if _looks_like_sql_query(source_str):
        from .adapter_sql import SqlAdapter
        return SqlAdapter(
            kwargs.pop("connection_string", "sqlite:///:memory:"),
            query=source_str,
            **kwargs,
        )

    # Cloud URI
    scheme = _get_scheme(source_str)
    if scheme in _SCHEME_MAP and _SCHEME_MAP[scheme] == "cloud":
        from .adapter_cloud import CloudAdapter
        return CloudAdapter(source_str, **kwargs)

    # API URL
    if scheme in ("http", "https"):
        from .adapter_api import ApiAdapter
        return ApiAdapter(source_str, **kwargs)

    # File path
    path = Path(source_str)
    if path.exists() or path.is_file():
        ext = path.suffix.lower()
        adapter_type = _EXTENSION_MAP.get(ext)

        if adapter_type == "csv" or ext in ("", ".data", ".dat"):
            from .csv_source import CsvDataSourceAdapter
            return CsvDataSourceAdapter(source_str, **kwargs)
        if adapter_type == "excel":
            from .adapter_file import ExcelAdapter
            return ExcelAdapter(source_str, **kwargs)
        if adapter_type == "parquet":
            from .adapter_file import ParquetAdapter
            return ParquetAdapter(source_str, **kwargs)
        if adapter_type == "json":
            from .adapter_file import JsonAdapter
            return JsonAdapter(source_str, **kwargs)
        if adapter_type == "ndjson":
            from .adapter_file import NdjsonAdapter
            return NdjsonAdapter(source_str, **kwargs)
        if adapter_type == "feather":
            from .adapter_file import FeatherAdapter
            return FeatherAdapter(source_str, **kwargs)
        # Default: try CSV
        from .csv_source import CsvDataSourceAdapter
        return CsvDataSourceAdapter(source_str, **kwargs)

    # Fallback: treat as CSV path
    from .csv_source import CsvDataSourceAdapter
    return CsvDataSourceAdapter(source_str, **kwargs)


def _is_dataframe(obj: Any) -> bool:
    """Check if object is a DataFrame-like."""
    if obj is None:
        return False
    type_name = type(obj).__module__
    if "pandas" in type_name or "polars" in type_name or "pyspark" in type_name:
        return True
    # Duck-typing: has .columns and .shape or .height
    return bool(hasattr(obj, "columns") and (hasattr(obj, "shape") or hasattr(obj, "height")))


def _looks_like_sql(source: str) -> bool:
    """Check if string looks like a SQL connection string."""
    lower = source.lower().strip()
    for prefix in ("sqlite://", "postgresql://", "postgres://", "mysql://",
                    "mssql://", "sqlserver://"):
        if lower.startswith(prefix):
            return True
    return False


def _looks_like_sql_query(source: str) -> bool:
    """Check if string looks like a SQL query."""
    upper = source.strip().upper()
    return any(upper.startswith(kw) for kw in _SQL_KEYWORDS)


def _get_scheme(source: str) -> str:
    """Extract URI scheme."""
    if "://" in source:
        return source.split("://")[0].lower()
    return ""


def list_supported_sources() -> dict[str, list[str]]:
    """List all supported data source types and their extensions/schemes."""
    return {
        "File formats": [
            "CSV (.csv, .tsv, .txt)",
            "Excel (.xlsx, .xls) [requires openpyxl]",
            "Parquet (.parquet, .pq) [requires pyarrow]",
            "JSON (.json)",
            "NDJSON (.jsonl, .ndjson)",
            "Feather (.feather, .arrow, .ipc) [requires pyarrow]",
        ],
        "Cloud storage": [
            "AWS S3 (s3://bucket/key) [requires boto3]",
            "Google Cloud Storage (gs://bucket/key) [requires google-cloud-storage]",
            "Azure Blob (az://container/blob) [requires azure-storage-blob]",
        ],
        "Databases": [
            "SQLite (sqlite:///path)",
            "PostgreSQL (postgresql://user:pass@host/db) [requires psycopg2]",
            "MySQL (mysql://user:pass@host/db) [requires pymysql]",
            "Any SQLAlchemy database (requires sqlalchemy)",
        ],
        "Data warehouses": [
            "BigQuery (bigquery://project/dataset) [requires google-cloud-bigquery]",
            "Snowflake (snowflake://user:pass@account/db/schema) [requires snowflake-connector-python]",
            "Redshift (redshift://user:pass@host:5439/db) [requires redshift-connector or psycopg2]",
        ],
        "DataFrames": [
            "pandas DataFrame",
            "polars DataFrame",
            "PySpark DataFrame",
        ],
        "APIs": [
            "REST API (https://api.example.com/endpoint) [requires requests]",
            "GraphQL (https://api.example.com/graphql) [requires requests]",
        ],
    }
