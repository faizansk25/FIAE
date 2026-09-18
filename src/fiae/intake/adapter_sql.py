"""SQL database adapter — PostgreSQL, MySQL, SQLite.

Supports any database with a DB-API 2.0 interface. Lazy-loads the
appropriate driver so core FIAE stays dependency-free.

Usage:
    adapter = SqlAdapter("sqlite:///data.db", table="users")
    adapter = SqlAdapter("postgresql://user:pass@host/db", query="SELECT * FROM users")
    adapter = SqlAdapter("mysql+pymysql://user:pass@host/db", table="orders")
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import threading
from typing import Any, Iterator, Optional

from .adapter_base import BaseAdapter
from .base import RowBatch


class SqlAdapter(BaseAdapter):
    """SQL database adapter implementing DataSourceAdapter protocol.

    Parameters
    ----------
    connection_string : str
        SQLAlchemy-style connection string or raw DSN.
        Examples:
            - "sqlite:///path/to/db.sqlite"
            - "postgresql://user:pass@localhost:5432/mydb"
            - "mysql://user:pass@localhost:3306/mydb"
            - "sqlite:///:memory:"
    table : str, optional
        Table name to query. Mutually exclusive with `query`.
    query : str, optional
        Custom SQL query. Mutually exclusive with `table`.
    schema : str, optional
        Database schema (for PostgreSQL/MySQL).
    batch_rows : int
        Rows per batch (default 2048).
    """

    def __init__(
        self,
        connection_string: str,
        *,
        table: Optional[str] = None,
        query: Optional[str] = None,
        schema: Optional[str] = None,
        batch_rows: int = 2048,
    ) -> None:
        super().__init__("sql", connection_string=connection_string)
        self._conn_str = connection_string
        self._table = table
        self._query = query
        self._schema = schema
        self._batch_rows = batch_rows
        # Thread-local connections: one adapter instance can be scanned from
        # multiple threads concurrently without sharing cursors (doc 08).
        self._local = threading.local()
        self._columns: Optional[list[str]] = None
        self._row_count: Optional[int] = None

        # Auto-detect engine
        self._engine = self._detect_engine(connection_string)

    def _detect_engine(self, conn_str: str) -> str:
        """Detect database engine from connection string."""
        lower = conn_str.lower()
        if lower.startswith("sqlite"):
            return "sqlite"
        if lower.startswith("postgresql") or lower.startswith("postgres"):
            return "postgresql"
        if lower.startswith("mysql"):
            return "mysql"
        if lower.startswith("mssql") or lower.startswith("sqlserver"):
            return "mssql"
        return "unknown"

    def _get_connection(self):
        """Return a thread-local database connection (lazy, per thread)."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn

        # Try sqlalchemy first
        try:
            sqlalchemy = importlib.import_module("sqlalchemy")
            engine = sqlalchemy.create_engine(self._conn_str)
            conn = engine.connect()
            self._local.conn = conn
            self._local.cursor = None
            self._connected = True
            return conn
        except ImportError:
            pass

        # Fall back to raw DB-API drivers
        if self._engine == "sqlite":
            sqlite3 = importlib.import_module("sqlite3")
            # Extract path from sqlite:///path
            path = self._conn_str.replace("sqlite:///", "").replace("sqlite://", "")
            conn = sqlite3.connect(path or ":memory:")
            self._local.conn = conn
            self._local.cursor = conn.cursor()
            self._connected = True
            return conn
        if self._engine == "postgresql":
            psycopg2 = importlib.import_module("psycopg2")
            conn = psycopg2.connect(self._conn_str)
            self._local.conn = conn
            self._local.cursor = conn.cursor()
            self._connected = True
            return conn
        if self._engine == "mysql":
            mysql_driver = None
            for mod in ("pymysql", "mysqlclient", "mysql.connector"):
                try:
                    mysql_driver = importlib.import_module(mod)
                    break
                except ImportError:
                    continue
            if mysql_driver is None:
                raise ImportError(
                    "MySQL adapter requires one of: pymysql, mysqlclient, mysql-connector-python"
                )
            # Parse connection string
            conn = mysql_driver.connect(
                host=self._parse_host(self._conn_str),
                user=self._parse_user(self._conn_str),
                password=self._parse_password(self._conn_str),
                database=self._parse_database(self._conn_str),
            )
            self._local.conn = conn
            self._local.cursor = conn.cursor()
            self._connected = True
            return conn
        raise ImportError(
            f"Unsupported SQL engine: {self._engine}. "
            "Install sqlalchemy or the appropriate driver."
        )

    @staticmethod
    def _parse_host(conn_str: str) -> str:
        # postgresql://user:pass@host:port/db
        after_at = conn_str.split("@")[-1] if "@" in conn_str else conn_str
        host_part = after_at.split("/")[0]
        return host_part.split(":")[0]

    @staticmethod
    def _parse_user(conn_str: str) -> str:
        if "//" in conn_str and "@" in conn_str:
            between = conn_str.split("//")[1].split("@")[0]
            return between.split(":")[0]
        return ""

    @staticmethod
    def _parse_password(conn_str: str) -> str:
        if "//" in conn_str and "@" in conn_str:
            between = conn_str.split("//")[1].split("@")[0]
            parts = between.split(":", 1)
            return parts[1] if len(parts) > 1 else ""
        return ""

    @staticmethod
    def _parse_database(conn_str: str) -> str:
        after_at = conn_str.split("@")[-1] if "@" in conn_str else conn_str
        parts = after_at.split("/")
        return parts[-1] if len(parts) > 1 else ""

    def source_id(self) -> str:
        if self._table:
            return f"sql|{self._engine}|{self._table}"
        return f"sql|{self._engine}|query"

    def estimate_rows(self) -> Optional[int]:
        if self._row_count is not None:
            return self._row_count
        try:
            conn = self._get_connection()
            cur = getattr(self._local, "cursor", None)
            if cur is not None:
                table = self._table or self._detect_table()
                if table:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    self._row_count = cur.fetchone()[0]
                    return self._row_count
            else:
                # SQLAlchemy path: one-off query for counting
                result = conn.execute(f"SELECT COUNT(*) FROM {table}")
                self._row_count = result.scalar()
                return self._row_count
        except Exception:
            pass
        return None

    def _detect_table(self) -> Optional[str]:
        """Try to detect the table name from the query."""
        if self._query:
            # Simple heuristic for "SELECT ... FROM table_name"
            q = self._query.upper()
            if "FROM" in q:
                after_from = q.split("FROM")[-1].strip()
                table = after_from.split()[0].strip('"').strip("'").strip("`")
                if table and not table.startswith("("):
                    return table
        return self._table

    def _resolve_table(self) -> Optional[str]:
        """Resolve table when none given: if the DB has exactly one table,
        use it (ergonomic default for single-table SQLite databases)."""
        if self._table or self._query:
            return self._table
        if self._engine != "sqlite":
            return None
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cur.fetchall()
                      if not str(r[0]).startswith("sqlite_")]
            cur.close()
            if len(tables) == 1:
                self._table = tables[0]
                return self._table
        except Exception:
            pass
        return None

    def scan(
        self,
        projection: Optional[list[str]] = None,
        predicate: Optional[Any] = None,
    ) -> Iterator[RowBatch]:
        """Scan the SQL source, yielding columnar RowBatch objects."""
        conn = self._get_connection()
        cur = getattr(self._local, "cursor", None)

        # Build query
        if self._query:
            sql = self._query
        elif self._table or self._resolve_table():
            cols = ", ".join(projection) if projection else "*"
            sql = f"SELECT {cols} FROM {self._table}"
            if predicate:
                sql += f" WHERE {predicate}"
        else:
            raise ValueError(
                "Either table or query must be specified "
                "(database has multiple tables - pass 'table')")

        if cur is not None:
            # Raw DB-API (thread-local cursor)
            cur.execute(sql)
            columns = [desc[0] for desc in cur.description] if cur.description else []
            if projection:
                columns = projection

            while True:
                rows = cur.fetchmany(self._batch_rows)
                if not rows:
                    break
                batch_cols: dict[str, list] = {col: [] for col in columns}
                for row in rows:
                    for i, col in enumerate(columns):
                        val = row[i] if i < len(row) else None
                        batch_cols[col].append(val if val is not None else None)
                if any(batch_cols.values()):
                    yield RowBatch(columns=batch_cols)
        else:
            # SQLAlchemy connection (thread-local)
            result = conn.execute(sql)
            columns = list(result.keys()) if hasattr(result, 'keys') else []

            batch_cols = {col: [] for col in columns}
            count = 0
            for row in result:
                for i, col in enumerate(columns):
                    val = row[i] if i < len(row) else None
                    batch_cols[col].append(val)
                count += 1
                if count >= self._batch_rows:
                    yield RowBatch(columns=batch_cols)
                    batch_cols = {col: [] for col in columns}
                    count = 0
            if count > 0:
                yield RowBatch(columns=batch_cols)

    def supports_pushdown(self) -> bool:
        return True  # SQL supports WHERE clause pushdown

    def fingerprint_material(self) -> bytes:
        """Fingerprint from connection string + table/query."""
        material = f"{self._engine}|{self._conn_str}|{self._table or self._query}"
        return hashlib.sha256(material.encode()).digest()

    def close(self) -> None:
        """Close this thread's database connection."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()
            self._local.conn = None
            self._local.cursor = None
            self._connected = False

    def __del__(self) -> None:
        self.close()
