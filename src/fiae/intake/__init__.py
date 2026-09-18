"""Data intake, schema, and profiling subsystem (doc 02).

Supports 20+ data source types through the DataSourceAdapter protocol:

Core adapters (zero dependencies):
    - CsvDataSourceAdapter — CSV/TSV with multi-region sampling
    - DataFrameAdapter — pandas, polars, PySpark DataFrames

Optional adapters (require additional packages):
    - SqlAdapter — PostgreSQL, MySQL, SQLite
    - CloudAdapter — S3, GCS, Azure Blob
    - ExcelAdapter — Excel files
    - ParquetAdapter — Parquet files
    - JsonAdapter — JSON arrays
    - NdjsonAdapter — JSON Lines / NDJSON
    - FeatherAdapter — Feather/Arrow IPC
    - ApiAdapter — REST and GraphQL APIs

Factory:
    - auto_adapter() — Auto-detect and create the right adapter

Usage:
    from fiae.intake import auto_adapter

    # Auto-detect from any source
    adapter = auto_adapter("data.csv")
    adapter = auto_adapter("s3://bucket/data.parquet")
    adapter = auto_adapter("postgresql://user:pass@localhost/db")
    adapter = auto_adapter(pandas_df)
    adapter = auto_adapter("https://api.example.com/data")

    # Profile the source
    from fiae.intake import profile_source, ProfileConfig
    profile = profile_source(adapter, ProfileConfig())
"""

from .base import DataSourceAdapter, RowBatch, SamplePlan
from .csv_source import CsvDataSourceAdapter
from .profiler import ProfileConfig, ProfileMode, profile_source
from .typing_engine import infer_physical_type, infer_semantic_type
from .stats import Welford, CardinalitySketch
from .adapter_dataframe import DataFrameAdapter
from .adapter_sql import SqlAdapter
from .adapter_cloud import CloudAdapter
from .adapter_file import (
    ExcelAdapter,
    ParquetAdapter,
    JsonAdapter,
    NdjsonAdapter,
    FeatherAdapter,
)
from .adapter_api import ApiAdapter
from .adapter_warehouse import (
    BigQueryAdapter,
    SnowflakeAdapter,
    RedshiftAdapter,
)
from .adapter_factory import auto_adapter, list_supported_sources

__all__ = [
    # Core protocol
    "DataSourceAdapter",
    "RowBatch",
    "SamplePlan",
    # CSV adapter
    "CsvDataSourceAdapter",
    # Profiling
    "ProfileConfig",
    "ProfileMode",
    "profile_source",
    # Type inference
    "infer_physical_type",
    "infer_semantic_type",
    # Statistics
    "Welford",
    "CardinalitySketch",
    # DataFrame adapter
    "DataFrameAdapter",
    # SQL adapter
    "SqlAdapter",
    # Cloud adapter
    "CloudAdapter",
    # File format adapters
    "ExcelAdapter",
    "ParquetAdapter",
    "JsonAdapter",
    "NdjsonAdapter",
    "FeatherAdapter",
    # API adapter
    "ApiAdapter",
    # Warehouse adapters
    "BigQueryAdapter",
    "SnowflakeAdapter",
    "RedshiftAdapter",
    # Factory
    "auto_adapter",
    "list_supported_sources",
]
