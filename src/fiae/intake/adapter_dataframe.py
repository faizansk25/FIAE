"""DataFrame adapter — pandas, polars, PySpark.

Wraps any DataFrame-like object into the FIAE DataSourceAdapter protocol.
This is the primary integration path for users who already have data in
DataFrames.

Usage:
    import pandas as pd
    df = pd.read_csv("data.csv")
    adapter = DataFrameAdapter(df)

    import polars as pl
    df = pl.read_parquet("data.parquet")
    adapter = DataFrameAdapter(df)
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterator, Optional

from .adapter_base import BaseAdapter
from .base import RowBatch


class DataFrameAdapter(BaseAdapter):
    """Adapter for pandas, polars, or PySpark DataFrames.

    Parameters
    ----------
    df : DataFrame
        A pandas, polars, or PySpark DataFrame.
    name : str, optional
        Name for this source (default: "dataframe").
    batch_rows : int
        Rows per batch (default 2048).
    """

    def __init__(
        self,
        df: Any,
        *,
        name: str = "dataframe",
        batch_rows: int = 2048,
    ) -> None:
        super().__init__("dataframe", name=name)
        self._df = df
        self._name = name
        self._batch_rows = batch_rows
        self._columns: Optional[list[str]] = None
        self._n_rows: Optional[int] = None
        self._df_type = self._detect_type(df)

    def _detect_type(self, df: Any) -> str:
        """Detect DataFrame library."""
        type_name = type(df).__module__
        if "pandas" in type_name:
            return "pandas"
        if "polars" in type_name:
            return "polars"
        if "pyspark" in type_name:
            return "spark"
        return "unknown"

    @property
    def columns(self) -> list[str]:
        if self._columns is None:
            if self._df_type == "pandas":
                self._columns = list(self._df.columns)
            elif self._df_type == "polars" or self._df_type == "spark":
                self._columns = self._df.columns
            else:
                self._columns = []
        return self._columns

    @property
    def n_rows(self) -> int:
        if self._n_rows is None:
            if self._df_type == "pandas":
                self._n_rows = len(self._df)
            elif self._df_type == "polars":
                self._n_rows = self._df.height
            elif self._df_type == "spark":
                self._n_rows = self._df.count()
            else:
                self._n_rows = 0
        return self._n_rows

    def source_id(self) -> str:
        return f"dataframe|{self._df_type}|{self._name}"

    def schema_hint(self) -> Optional[dict[str, Any]]:
        """Provide schema from DataFrame dtypes."""
        if self._df_type == "pandas":
            return {col: str(dtype) for col, dtype in self._df.dtypes.items()}
        if self._df_type == "polars":
            return {col: str(dtype) for col, dtype in zip(
                self._df.columns, self._df.dtypes
            )}
        return None

    def estimate_rows(self) -> Optional[int]:
        return self.n_rows

    def estimate_bytes(self) -> Optional[int]:
        """Estimate byte size (rough)."""
        if self._df_type == "pandas":
            try:
                return int(self._df.memory_usage(deep=True).sum())
            except Exception:
                return None
        return None

    def scan(
        self,
        projection: Optional[list[str]] = None,
        predicate: Optional[Any] = None,
    ) -> Iterator[RowBatch]:
        """Scan the DataFrame in batches."""
        cols_to_use = projection or self.columns
        n = self.n_rows

        for start in range(0, n, self._batch_rows):
            end = min(start + self._batch_rows, n)
            batch_data: dict[str, list] = {}

            for col in cols_to_use:
                if self._df_type == "pandas":
                    series = self._df[col].iloc[start:end]
                    batch_data[col] = [
                        None if v != v else (str(v) if v is not None else None)
                        for v in series.tolist()
                    ]
                elif self._df_type == "polars":
                    # Polars slice
                    chunk = self._df[col][start:end]
                    batch_data[col] = [
                        None if v != v else (str(v) if v is not None else None)
                        for v in chunk.to_list()
                    ]
                elif self._df_type == "spark":
                    # Spark: collect in batches
                    rows = self._df.select(col).tail(end)[start:end]
                    batch_data[col] = [str(r[0]) if r[0] is not None else None for r in rows]

            if batch_data:
                yield RowBatch(columns=batch_data)

    def supports_seek_sampling(self) -> bool:
        return self._df_type in ("pandas", "polars")

    def fingerprint_material(self) -> bytes:
        """Fingerprint from DataFrame hash."""
        if self._df_type == "pandas":
            try:
                return hashlib.sha256(
                    self._df.values.tobytes()[:8192]
                ).digest()
            except Exception:
                pass
        return hashlib.sha256(
            f"df|{self._df_type}|{self._name}|{self.n_rows}".encode()
        ).digest()

    def to_pandas(self) -> Any:
        """Convert to pandas DataFrame if not already."""
        if self._df_type == "pandas":
            return self._df
        if self._df_type == "polars":
            return self._df.to_pandas()
        if self._df_type == "spark":
            return self._df.toPandas()
        raise ValueError(f"Cannot convert {self._df_type} to pandas")

    def to_polars(self) -> Any:
        """Convert to polars DataFrame if not already."""
        if self._df_type == "polars":
            return self._df
        if self._df_type == "pandas":
            import polars as pl
            return pl.from_pandas(self._df)
        raise ValueError(f"Cannot convert {self._df_type} to polars")
