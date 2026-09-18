"""File format adapters — Excel, Parquet, JSON/NDJSON, Feather, Arrow.

Additional file format support beyond CSV. Each adapter wraps a specific
file format into the FIAE DataSourceAdapter protocol.

Usage:
    adapter = ExcelAdapter("data.xlsx", sheet="Sheet1")
    adapter = ParquetAdapter("data.parquet", columns=["col1", "col2"])
    adapter = JsonAdapter("data.json")
    adapter = NdjsonAdapter("data.ndjson")
    adapter = FeatherAdapter("data.feather")
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Iterator, Optional

from .adapter_base import BaseAdapter
from .base import RowBatch


# ── Excel Adapter ───────────────────────────────────────────────────────

class ExcelAdapter(BaseAdapter):
    """Adapter for Excel files (.xlsx, .xls).

    Parameters
    ----------
    path : str | Path
        Path to the Excel file.
    sheet : str, optional
        Sheet name (default: first sheet).
    batch_rows : int
        Rows per batch (default 2048).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        sheet: Optional[str] = None,
        batch_rows: int = 2048,
    ) -> None:
        super().__init__("excel")
        self._path = Path(path)
        self._sheet = sheet
        self._batch_rows = batch_rows
        self._size = self._path.stat().st_size if self._path.exists() else 0

    def source_id(self) -> str:
        return f"excel|{self._path.resolve().as_posix()}"

    def estimate_rows(self) -> Optional[int]:
        return None  # Need to read first sheet

    def estimate_bytes(self) -> Optional[int]:
        return self._size

    def scan(
        self,
        projection: Optional[list[str]] = None,
        predicate: Optional[Any] = None,
    ) -> Iterator[RowBatch]:
        """Read Excel file in batches."""
        try:
            openpyxl = importlib.import_module("openpyxl")
            wb = openpyxl.load_workbook(str(self._path), read_only=True)
            ws = wb[self._sheet] if self._sheet else wb.active

            header = None
            columns: dict[str, list] = {}
            count = 0

            for i, row in enumerate(ws.iter_rows(values_only=True)):
                if i == 0:
                    header = [str(c) if c is not None else f"col_{j}" for j, c in enumerate(row)]
                    columns = {col: [] for col in header}
                    continue

                for j, val in enumerate(row):
                    if j < len(header):
                        columns[header[j]].append(
                            None if val is None else str(val)
                        )
                count += 1

                if count >= self._batch_rows:
                    if projection:
                        columns = {k: v for k, v in columns.items() if k in projection}
                    if any(columns.values()):
                        yield RowBatch(columns=columns)
                    columns = {col: [] for col in (header or [])}
                    count = 0

            if count > 0 and any(columns.values()):
                if projection:
                    columns = {k: v for k, v in columns.items() if k in projection}
                yield RowBatch(columns=columns)

            wb.close()
        except ImportError as err:
            raise ImportError(
                "Excel support requires openpyxl: pip install openpyxl") from err

    def fingerprint_material(self) -> bytes:
        return hashlib.sha256(
            f"excel|{self._path.resolve()}|{self._size}".encode()
        ).digest()


# ── Parquet Adapter ─────────────────────────────────────────────────────

class ParquetAdapter(BaseAdapter):
    """Adapter for Parquet files.

    Parameters
    ----------
    path : str | Path
        Path to the Parquet file.
    columns : list[str], optional
        Columns to read (default: all).
    batch_rows : int
        Rows per batch (default 2048).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        columns: Optional[list[str]] = None,
        batch_rows: int = 2048,
    ) -> None:
        super().__init__("parquet")
        self._path = Path(path)
        self._columns = columns
        self._batch_rows = batch_rows
        self._size = self._path.stat().st_size if self._path.exists() else 0

    def source_id(self) -> str:
        return f"parquet|{self._path.resolve().as_posix()}"

    def estimate_rows(self) -> Optional[int]:
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(str(self._path))
            return pf.metadata.num_rows
        except ImportError:
            return None
        except Exception:
            return None

    def estimate_bytes(self) -> Optional[int]:
        return self._size

    def scan(
        self,
        projection: Optional[list[str]] = None,
        predicate: Optional[Any] = None,
    ) -> Iterator[RowBatch]:
        """Read Parquet file in batches."""
        cols = projection or self._columns
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(str(self._path))
            for batch in pf.iter_batches(
                batch_size=self._batch_rows,
                columns=cols,
            ):
                col_names = batch.schema.names if hasattr(batch, 'schema') else cols
                data = {}
                for i, name in enumerate(col_names):
                    data[name] = [
                        None if v is None else str(v)
                        for v in batch.column(i).to_pylist()
                    ]
                if any(data.values()):
                    yield RowBatch(columns=data)
        except ImportError:
            try:
                import pandas as pd
                df = pd.read_parquet(str(self._path), columns=cols)
                data = {}
                for col in df.columns:
                    data[col] = [
                        None if v != v else (str(v) if v is not None else None)
                        for v in df[col].tolist()
                    ]
                if data:
                    yield RowBatch(columns=data)
            except ImportError as err:
                raise ImportError(
                    "Parquet support requires pyarrow or pandas: pip install pyarrow"
                ) from err

    def fingerprint_material(self) -> bytes:
        return hashlib.sha256(
            f"parquet|{self._path.resolve()}|{self._size}".encode()
        ).digest()


# ── JSON / NDJSON Adapter ───────────────────────────────────────────────

class JsonAdapter(BaseAdapter):
    """Adapter for JSON files (array of objects).

    Parameters
    ----------
    path : str | Path
        Path to the JSON file.
    batch_rows : int
        Rows per batch (default 2048).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        batch_rows: int = 2048,
    ) -> None:
        super().__init__("json")
        self._path = Path(path)
        self._batch_rows = batch_rows
        self._size = self._path.stat().st_size if self._path.exists() else 0

    def source_id(self) -> str:
        return f"json|{self._path.resolve().as_posix()}"

    def estimate_bytes(self) -> Optional[int]:
        return self._size

    def scan(
        self,
        projection: Optional[list[str]] = None,
        predicate: Optional[Any] = None,
    ) -> Iterator[RowBatch]:
        """Read JSON array of objects."""
        with open(self._path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            data = [data]

        columns: dict[str, list] = {}
        count = 0

        for obj in data:
            if not isinstance(obj, dict):
                continue
            for k, v in obj.items():
                columns.setdefault(k, []).append(
                    None if v is None else str(v)
                )
            count += 1

            if count >= self._batch_rows:
                if projection:
                    columns = {k: v for k, v in columns.items() if k in projection}
                if any(columns.values()):
                    yield RowBatch(columns=columns)
                columns = {}
                count = 0

        if count > 0 and any(columns.values()):
            if projection:
                columns = {k: v for k, v in columns.items() if k in projection}
            yield RowBatch(columns=columns)

    def fingerprint_material(self) -> bytes:
        return hashlib.sha256(
            f"json|{self._path.resolve()}|{self._size}".encode()
        ).digest()


class NdjsonAdapter(BaseAdapter):
    """Adapter for NDJSON / JSON Lines files.

    Parameters
    ----------
    path : str | Path
        Path to the NDJSON file.
    batch_rows : int
        Rows per batch (default 2048).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        batch_rows: int = 2048,
    ) -> None:
        super().__init__("ndjson")
        self._path = Path(path)
        self._batch_rows = batch_rows
        self._size = self._path.stat().st_size if self._path.exists() else 0

    def source_id(self) -> str:
        return f"ndjson|{self._path.resolve().as_posix()}"

    def estimate_bytes(self) -> Optional[int]:
        return self._size

    def scan(
        self,
        projection: Optional[list[str]] = None,
        predicate: Optional[Any] = None,
    ) -> Iterator[RowBatch]:
        """Read NDJSON line by line."""
        columns: dict[str, list] = {}
        count = 0

        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                for k, v in obj.items():
                    columns.setdefault(k, []).append(
                        None if v is None else str(v)
                    )
                count += 1

                if count >= self._batch_rows:
                    if projection:
                        columns = {k: v for k, v in columns.items() if k in projection}
                    if any(columns.values()):
                        yield RowBatch(columns=columns)
                    columns = {}
                    count = 0

        if count > 0 and any(columns.values()):
            if projection:
                columns = {k: v for k, v in columns.items() if k in projection}
            yield RowBatch(columns=columns)

    def fingerprint_material(self) -> bytes:
        return hashlib.sha256(
            f"ndjson|{self._path.resolve()}|{self._size}".encode()
        ).digest()


# ── Feather / Arrow Adapter ─────────────────────────────────────────────

class FeatherAdapter(BaseAdapter):
    """Adapter for Feather / Arrow IPC files.

    Parameters
    ----------
    path : str | Path
        Path to the Feather file.
    columns : list[str], optional
        Columns to read.
    batch_rows : int
        Rows per batch (default 2048).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        columns: Optional[list[str]] = None,
        batch_rows: int = 2048,
    ) -> None:
        super().__init__("feather")
        self._path = Path(path)
        self._columns = columns
        self._batch_rows = batch_rows
        self._size = self._path.stat().st_size if self._path.exists() else 0

    def source_id(self) -> str:
        return f"feather|{self._path.resolve().as_posix()}"

    def estimate_bytes(self) -> Optional[int]:
        return self._size

    def scan(
        self,
        projection: Optional[list[str]] = None,
        predicate: Optional[Any] = None,
    ) -> Iterator[RowBatch]:
        """Read Feather file."""
        cols = projection or self._columns
        try:
            import pyarrow.feather as feather
            table = feather.read_table(str(self._path), columns=cols)
            column_names = table.column_names

            for start in range(0, len(table), self._batch_rows):
                end = min(start + self._batch_rows, len(table))
                batch = table.slice(start, end - start)
                data = {}
                for i, name in enumerate(column_names):
                    data[name] = [
                        None if v is None else str(v)
                        for v in batch.column(i).to_pylist()
                    ]
                if any(data.values()):
                    yield RowBatch(columns=data)
        except ImportError as err:
            raise ImportError(
                "Feather support requires pyarrow: pip install pyarrow") from err

    def fingerprint_material(self) -> bytes:
        return hashlib.sha256(
            f"feather|{self._path.resolve()}|{self._size}".encode()
        ).digest()
