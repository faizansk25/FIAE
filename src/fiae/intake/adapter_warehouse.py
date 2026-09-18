"""Data warehouse adapters — BigQuery, Snowflake, Redshift.

Each adapter lazy-loads its SDK driver so core FIAE stays dependency-free
(NFR-002). All scans are streaming (fetchmany batches) with constant memory,
and connections are **thread-local** so one adapter instance can be scanned
from multiple worker threads without sharing a cursor across threads.

Usage:
    adapter = BigQueryAdapter("bigquery://my-project/my_dataset", table="events")
    adapter = SnowflakeAdapter("snowflake://user:pass@account/db/schema", table="orders")
    adapter = RedshiftAdapter("redshift://user:pass@cluster.abc.us-east-1.redshift.amazonaws.com:5439/dev", table="sales")

Or via the factory:
    adapter = auto_adapter("bigquery://proj/ds", table="t")
    adapter = auto_adapter("snowflake://u:p@acct/db/s", table="t")
    adapter = auto_adapter("redshift://u:p@host:5439/db", table="t")
"""

from __future__ import annotations

import contextlib
import hashlib
import threading
from typing import Any, Iterator, Optional

from .adapter_base import BaseAdapter
from .base import RowBatch


class _WarehouseAdapterBase(BaseAdapter):
    """Shared streaming scan + thread-local connection machinery.

    Subclasses implement ``_connect()`` (return a DB-API connection) and
    optionally ``_quote_ident()`` for engine-specific identifier quoting.
    """

    def __init__(self, uri: str, *, table: Optional[str] = None,
                 query: Optional[str] = None, batch_rows: int = 2048,
                 **connect_kwargs: Any) -> None:
        super().__init__("warehouse", uri=uri)
        self._uri = uri
        self._table = table
        self._query = query
        self._batch_rows = batch_rows
        self._connect_kwargs = connect_kwargs
        self._local = threading.local()
        if not table and not query:
            raise ValueError("Warehouse adapters require 'table' or 'query'")

    # ── Subclass hooks ────────────────────────────────────────────────────

    def _connect(self):
        """Return a new DB-API connection (called per thread)."""
        raise NotImplementedError

    def _quote_ident(self, name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    # ── Thread-local connection management ───────────────────────────────

    def _get_connection(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    # ── Introspection ─────────────────────────────────────────────────────

    def source_id(self) -> str:
        label = self._table or "query"
        return f"{self._engine}|{self._uri}|{label}"

    def estimate_rows(self) -> Optional[int]:
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute(self._count_sql())
            row = cur.fetchone()
            cur.close()
            return int(row[0]) if row else None
        except Exception:
            return None

    def _count_sql(self) -> str:
        target = self._table or f"({self._query})"
        return f"SELECT COUNT(*) FROM {target}"

    # ── Scan ──────────────────────────────────────────────────────────────

    def scan(
        self,
        projection: Optional[list[str]] = None,
        predicate: Optional[Any] = None,
    ) -> Iterator[RowBatch]:
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            if self._query:
                sql = self._query
            else:
                cols = ", ".join(self._quote_ident(c) for c in projection) \
                    if projection else "*"
                sql = f"SELECT {cols} FROM {self._quote_ident(self._table)}"
                if predicate:
                    sql += f" WHERE {predicate}"
            cur.execute(sql)
            columns = [d[0] for d in (cur.description or [])]
            if projection:
                columns = projection
            while True:
                rows = cur.fetchmany(self._batch_rows)
                if not rows:
                    break
                batch_cols: dict[str, list] = {c: [] for c in columns}
                for row in rows:
                    for i, c in enumerate(columns):
                        v = row[i] if i < len(row) else None
                        batch_cols[c].append(v)
                yield RowBatch(columns=batch_cols)
        finally:
            cur.close()

    def supports_pushdown(self) -> bool:
        return True

    def fingerprint_material(self) -> bytes:
        material = f"{self._engine}|{self._uri}|{self._table or self._query}"
        return hashlib.sha256(material.encode()).digest()

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()
            self._local.conn = None

    def __del__(self) -> None:
        self.close()


class BigQueryAdapter(_WarehouseAdapterBase):
    """Google BigQuery adapter (driver: ``google-cloud-bigquery``).

    Uses the BigQuery DB-API (``google.cloud.bigquery.dbapi``) which supports
    fetchmany-based streaming via page-size limits.
    """

    _engine = "bigquery"

    def __init__(self, uri: str, *, table: Optional[str] = None,
                 query: Optional[str] = None, batch_rows: int = 2048,
                 project: Optional[str] = None,
                 credentials: Optional[str] = None,
                 location: Optional[str] = None) -> None:
        super().__init__(uri, table=table, query=query, batch_rows=batch_rows,
                         project=project, credentials=credentials,
                         location=location)
        # Parse project/dataset from URI when not given explicitly:
        # bigquery://project/dataset[/table]
        if project is None:
            rest = uri.split("//", 1)[-1]
            path_parts = rest.split("/")
            if path_parts and path_parts[0]:
                self._project = path_parts[0]
            else:
                self._project = None
        else:
            self._project = project
        self._credentials = credentials
        self._location = location

    def _connect(self):
        try:
            from google.cloud import bigquery
            from google.cloud.bigquery.dbapi import connect
        except ImportError as exc:
            raise ImportError(
                "BigQuery adapter requires google-cloud-bigquery: "
                "pip install google-cloud-bigquery"
            ) from exc
        client_kwargs: dict[str, Any] = {}
        if self._project:
            client_kwargs["project"] = self._project
        if self._credentials:
            client_kwargs["credentials"] = self._credentials
        client = bigquery.Client(**client_kwargs)
        if self._location:
            client.location = self._location
        return connect(client)

    def _quote_ident(self, name: str) -> str:
        return "`" + name.replace("`", "\\`") + "`"

    def _count_sql(self) -> str:
        if self._table:
            return f"SELECT COUNT(*) FROM {self._quote_ident(self._table)}"
        return f"SELECT COUNT(*) FROM ({self._query})"


class SnowflakeAdapter(_WarehouseAdapterBase):
    """Snowflake adapter (driver: ``snowflake-connector-python``).

    Connection string format:
        snowflake://user:password@account/db/schema?warehouse=WH
    """

    _engine = "snowflake"

    def __init__(self, uri: str, *, table: Optional[str] = None,
                 query: Optional[str] = None, batch_rows: int = 2048,
                 warehouse: Optional[str] = None,
                 role: Optional[str] = None,
                 account: Optional[str] = None) -> None:
        super().__init__(uri, table=table, query=query, batch_rows=batch_rows,
                         warehouse=warehouse, role=role, account=account)
        self._warehouse = warehouse
        self._role = role
        self._account = account

    def _connect(self):
        try:
            import snowflake.connector
        except ImportError as exc:
            raise ImportError(
                "Snowflake adapter requires snowflake-connector-python: "
                "pip install snowflake-connector-python"
            ) from exc
        kwargs: dict[str, Any] = dict(self._connect_kwargs)
        kwargs["account"] = self._account or self._parse_account()
        user, password, database, schema = self._parse_uri()
        if user:
            kwargs["user"] = user
        if password:
            kwargs["password"] = password
        if database:
            kwargs["database"] = database
        if schema:
            kwargs["schema"] = schema
        if self._warehouse:
            kwargs["warehouse"] = self._warehouse
        if self._role:
            kwargs["role"] = self._role
        return snowflake.connector.connect(**kwargs)

    def _parse_uri(self) -> tuple[str, str, str, str]:
        """Parse snowflake://user:pass@account/db/schema."""
        from urllib.parse import urlparse
        parsed = urlparse(self._uri)
        user = parsed.username or ""
        password = parsed.password or ""
        parts = [p for p in parsed.path.split("/") if p]
        database = parts[0] if parts else ""
        schema = parts[1] if len(parts) > 1 else ""
        return user, password, database, schema

    def _parse_account(self) -> str:
        from urllib.parse import urlparse
        return urlparse(self._uri).hostname or ""


class RedshiftAdapter(_WarehouseAdapterBase):
    """Amazon Redshift adapter (driver: ``redshift_connector`` or psycopg2).

    Connection string format:
        redshift://user:pass@cluster.xxx.redshift.amazonaws.com:5439/db
    """

    _engine = "redshift"

    def __init__(self, uri: str, *, table: Optional[str] = None,
                 query: Optional[str] = None, batch_rows: int = 2048,
                 iam_profile: Optional[str] = None) -> None:
        super().__init__(uri, table=table, query=query, batch_rows=batch_rows,
                         iam_profile=iam_profile)
        self._iam_profile = iam_profile

    def _connect(self):
        try:
            import redshift_connector
        except ImportError:
            redshift_connector = None
        if redshift_connector is not None:
            from urllib.parse import urlparse
            parsed = urlparse(self._uri)
            kwargs: dict[str, Any] = {
                "host": parsed.hostname,
                "port": parsed.port or 5439,
                "database": (parsed.path or "/").lstrip("/") or "dev",
                "user": parsed.username,
                "password": parsed.password,
            }
            if self._iam_profile:
                kwargs["profile"] = self._iam_profile
            return redshift_connector.connect(**kwargs)
        # Fallback: psycopg2 (Redshift is postgres-compatible)
        try:
            import psycopg2
        except ImportError as exc:
            raise ImportError(
                "Redshift adapter requires redshift-connector or psycopg2: "
                "pip install redshift-connector-python  |  pip install psycopg2-binary"
            ) from exc
        return psycopg2.connect(self._uri.replace("redshift://", "postgresql://", 1))

    def _quote_ident(self, name: str) -> str:
        return '"' + name.replace('"', '""') + '"'
