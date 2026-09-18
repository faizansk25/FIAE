"""Cloud storage adapter — AWS S3, Google Cloud Storage, Azure Blob.

Provides streaming access to files in cloud object stores. Supports CSV,
Parquet, JSON, and other formats stored in cloud buckets.

Usage:
    adapter = CloudAdapter("s3://bucket/data.csv")
    adapter = CloudAdapter("gs://bucket/data.parquet")
    adapter = CloudAdapter("az://container/data.csv")
    adapter = CloudAdapter("s3://bucket/data.csv", profile="production")
"""

from __future__ import annotations

import hashlib
import importlib
from typing import Any, Iterator, Optional
from urllib.parse import urlparse

from .adapter_base import BaseAdapter
from .base import RowBatch


class CloudAdapter(BaseAdapter):
    """Cloud object storage adapter for S3, GCS, Azure Blob.

    Parameters
    ----------
    uri : str
        Cloud URI: s3://, gs://, az://, or wasbs://
    profile : str, optional
        AWS profile name (for S3).
    project : str, optional
        GCP project (for GCS).
    credential : str, optional
        Path to credential file or connection string.
    batch_rows : int
        Rows per batch (default 2048).
    """

    def __init__(
        self,
        uri: str,
        *,
        profile: Optional[str] = None,
        project: Optional[str] = None,
        credential: Optional[str] = None,
        batch_rows: int = 2048,
    ) -> None:
        super().__init__("cloud", uri=uri)
        self._uri = uri
        self._profile = profile
        self._project = project
        self._credential = credential
        self._batch_rows = batch_rows
        self._provider = self._detect_provider(uri)
        self._parsed = urlparse(uri)
        self._size: Optional[int] = None

    def _detect_provider(self, uri: str) -> str:
        """Detect cloud provider from URI scheme."""
        lower = uri.lower()
        if lower.startswith("s3://"):
            return "s3"
        if lower.startswith("gs://"):
            return "gcs"
        if lower.startswith("az://") or lower.startswith("wasbs://"):
            return "azure"
        if lower.startswith("abfss://"):
            return "azure"
        return "unknown"

    @property
    def _bucket(self) -> str:
        return self._parsed.hostname or self._parsed.path.split("/")[1]

    @property
    def _key(self) -> str:
        return self._parsed.path.lstrip("/")

    def source_id(self) -> str:
        return f"cloud|{self._provider}|{self._uri}"

    def estimate_rows(self) -> Optional[int]:
        return None  # Unknown without reading

    def estimate_bytes(self) -> Optional[int]:
        if self._size is not None:
            return self._size
        try:
            if self._provider == "s3":
                s3 = importlib.import_module("boto3").client("s3")
                resp = s3.head_object(Bucket=self._bucket, Key=self._key)
                self._size = resp["ContentLength"]
                return self._size
            if self._provider == "gcs":
                gcs = importlib.import_module("google.cloud.storage").Client(
                    project=self._project
                )
                bucket = gcs.bucket(self._bucket)
                blob = bucket.blob(self._key)
                blob.reload()
                self._size = blob.size
                return self._size
            if self._provider == "azure":
                blob_service = importlib.import_module(
                    "azure.storage.blob"
                ).BlobServiceClient.from_connection_string(self._credential)
                container_client = blob_service.get_container_client(self._bucket)
                blob_client = container_client.get_blob_client(self._key)
                props = blob_client.get_blob_properties()
                self._size = props.size
                return self._size
        except Exception:
            pass
        return None

    def _iter_chunks(self, chunk_size: int = 1024 * 256) -> Iterator[bytes]:
        """Stream the object as byte chunks (never loads whole file in memory).

        Shows a download progress bar with ETA when total size is known,
        matching the ApiAdapter UX.
        """
        import sys
        import time

        body = None  # raw streaming body supporting .read(chunk)
        total = self.estimate_bytes()

        if self._provider == "s3":
            s3 = importlib.import_module("boto3").client("s3")
            resp = s3.get_object(Bucket=self._bucket, Key=self._key)
            body = resp["Body"]  # boto3 StreamingBody — streams over the wire
        elif self._provider == "gcs":
            gcs = importlib.import_module("google.cloud.storage").Client(
                project=self._project
            )
            blob = gcs.bucket(self._bucket).blob(self._key)
            body = blob.open("rb")  # chunked resumable reader
        elif self._provider == "azure":
            blob_service = importlib.import_module(
                "azure.storage.blob"
            ).BlobServiceClient.from_connection_string(self._credential)
            container_client = blob_service.get_container_client(self._bucket)
            blob_client = container_client.get_blob_client(self._key)
            body = blob_client.download_blob().chunks()
        else:
            raise ImportError(f"Cloud provider {self._provider} SDK not installed")

        downloaded = 0
        start = time.time()
        use_bar = sys.stdout.isatty() and bool(total)
        while True:
            chunk = body.read(chunk_size)
            if not chunk:
                break
            downloaded += len(chunk)
            if use_bar:
                pct = min(100.0, downloaded * 100.0 / total)
                speed = downloaded / max(1e-9, time.time() - start)
                eta = max(0, (total - downloaded) / speed) if speed > 0 else 0
                bar = "#" * int(pct // 4) + "-" * (25 - int(pct // 4))
                sys.stdout.write(
                    f"\r  ↓ Downloading: [{bar}] {pct:5.1f}% "
                    f"{downloaded / 1e6:.1f}/{total / 1e6:.1f} MB "
                    f"{speed / 1e6:.1f} MB/s ETA: {eta:.0f}s"
                )
                sys.stdout.flush()
            yield chunk
        if use_bar:
            sys.stdout.write("\r  ↓ Downloading: [" + "#" * 25 + "] 100.0%\n")
            sys.stdout.flush()

    def _get_text_stream(self):
        """Return a text-mode incremental stream over the cloud object."""
        import io

        class _TextStream(io.RawIOBase):
            def __init__(self, gen):
                self._gen = gen
                self._buf = b""

            def readable(self):
                return True

            def readinto(self, b):
                while len(self._buf) < len(b):
                    try:
                        self._buf += next(self._gen)
                    except StopIteration:
                        break
                n = min(len(b), len(self._buf))
                b[:n] = self._buf[:n]
                self._buf = self._buf[n:]
                return n

        return io.TextIOWrapper(
            io.BufferedReader(_TextStream(self._iter_chunks()), buffer_size=256 * 1024),
            encoding="utf-8",
        )

    def scan(
        self,
        projection: Optional[list[str]] = None,
        predicate: Optional[Any] = None,
    ) -> Iterator[RowBatch]:
        """Stream data from cloud file."""
        # Delegate to the appropriate format adapter
        lower = self._uri.lower()
        if lower.endswith(".csv") or lower.endswith(".tsv") or lower.endswith(".txt"):
            yield from self._scan_csv()
        elif lower.endswith(".json") or lower.endswith(".jsonl") or lower.endswith(".ndjson"):
            yield from self._scan_jsonl()
        elif lower.endswith(".parquet"):
            yield from self._scan_parquet(projection)
        else:
            # Default: try CSV
            yield from self._scan_csv()

    def _scan_csv(self) -> Iterator[RowBatch]:
        """Scan CSV from cloud incrementally (constant memory, any file size)."""
        import csv

        stream = self._get_text_stream()
        reader = csv.reader(stream)
        header = None
        columns: dict[str, list] = {}
        nrows = 0

        for i, row in enumerate(reader):
            if i == 0:
                header = row
                columns = {col: [] for col in header}
                continue
            if header:
                for j, col in enumerate(header):
                    val = row[j] if j < len(row) else None
                    columns[col].append(val if val != "" else None)
            else:
                # No header: discover columns on first data row
                if not columns:
                    columns = {f"col_{j}": [] for j in range(len(row))}
                for j, col in enumerate(columns):
                    val = row[j] if j < len(row) else None
                    columns[col].append(val if val != "" else None)
            nrows += 1

            if nrows >= self._batch_rows:
                if columns and any(len(v) for v in columns.values()):
                    yield RowBatch(columns=columns)
                columns = {col: [] for col in (header or columns.keys())}
                nrows = 0

        if columns and any(len(v) for v in columns.values()):
            yield RowBatch(columns=columns)

    def _scan_jsonl(self) -> Iterator[RowBatch]:
        """Scan NDJSON from cloud incrementally (constant memory)."""
        import json

        stream = self._get_text_stream()
        columns: dict[str, list] = {}
        count = 0

        for line in stream:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for k, v in obj.items():
                columns.setdefault(k, []).append(
                    None if v is None else str(v)
                )
            count += 1
            if count >= self._batch_rows:
                if columns:
                    yield RowBatch(columns=columns)
                columns = {}
                count = 0

        if columns:
            yield RowBatch(columns=columns)

    def _get_bytes_fileobj(self):
        """Parquet needs random access (seek), so buffer the object in memory."""
        import io

        return io.BytesIO(b"".join(self._iter_chunks()))

    def _scan_parquet(self, projection: Optional[list[str]] = None) -> Iterator[RowBatch]:
        """Scan Parquet from cloud."""
        try:
            import pyarrow.parquet as pq
            fileobj = self._get_bytes_fileobj()
            pf = pq.ParquetFile(fileobj)
            columns = projection or pf.schema.names
            for batch in pf.iter_batches(batch_size=self._batch_rows, columns=columns):
                cols = {col: batch.column(i).to_pylist() for i, col in enumerate(columns)}
                # Convert to strings
                str_cols = {}
                for k, v in cols.items():
                    str_cols[k] = [
                        None if val is None else str(val) for val in v
                    ]
                if any(str_cols.values()):
                    yield RowBatch(columns=str_cols)
        except ImportError:
            # Fallback: try pandas
            try:
                import pandas as pd
                fileobj = self._get_bytes_fileobj()
                df = pd.read_parquet(fileobj, columns=projection)
                cols = {}
                for col in df.columns:
                    cols[col] = [
                        None if v != v else (str(v) if v is not None else None)
                        for v in df[col].tolist()
                    ]
                if cols:
                    yield RowBatch(columns=cols)
            except ImportError as err:
                raise ImportError(
                    "Parquet support requires pyarrow or pandas with pyarrow"
                ) from err

    def fingerprint_material(self) -> bytes:
        """Fingerprint from URI + size."""
        size = self.estimate_bytes() or 0
        material = f"{self._uri}|{size}"
        return hashlib.sha256(material.encode()).digest()

    def close(self) -> None:
        """No persistent connections to close."""
        pass
