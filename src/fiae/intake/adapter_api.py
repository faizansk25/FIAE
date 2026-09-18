"""API adapter — REST endpoints and GraphQL.

Fetch data from HTTP APIs and convert to the FIAE adapter protocol.
Supports paginated REST APIs, GraphQL queries, and streaming endpoints.

Usage:
    adapter = ApiAdapter("https://api.example.com/users")
    adapter = ApiAdapter(
        "https://api.example.com/graphql",
        query="{ users { id name email age } }",
        json_path="data.users",
    )
    adapter = ApiAdapter(
        "https://api.example.com/data",
        headers={"Authorization": "Bearer token"},
        pagination="offset",
        page_size=100,
    )
"""

from __future__ import annotations

import hashlib
import importlib
import json
from typing import Any, Iterator, Optional

from .adapter_base import BaseAdapter
from .base import RowBatch
from ..errors import ErrorCode, FIAEError


class ApiAdapter(BaseAdapter):
    """REST/GraphQL API adapter.

    Parameters
    ----------
    url : str
        API endpoint URL.
    method : str
        HTTP method (default: GET).
    headers : dict, optional
        HTTP headers (e.g., Authorization).
    params : dict, optional
        Query parameters.
    body : dict, optional
        Request body (for POST).
    query : str, optional
        GraphQL query (triggers GraphQL mode).
    json_path : str, optional
        Dot-separated path to extract records from JSON response.
        E.g., "data.users" or "results.items".
    pagination : str, optional
        Pagination strategy: "offset", "cursor", "link", or None.
    page_size : int
        Records per page for pagination (default: 100).
    max_pages : int
        Maximum pages to fetch (default: 100).
    batch_rows : int
        Rows per batch (default 2048).
    """

    def __init__(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: Optional[dict[str, str]] = None,
        params: Optional[dict[str, Any]] = None,
        body: Optional[dict[str, Any]] = None,
        query: Optional[str] = None,
        json_path: Optional[str] = None,
        pagination: Optional[str] = None,
        page_size: int = 100,
        max_pages: int = 100,
        batch_rows: int = 2048,
    ) -> None:
        super().__init__("api", url=url)
        self._url = url
        self._method = method
        self._headers = headers or {}
        self._params = params or {}
        self._body = body
        self._query = query
        self._json_path = json_path
        self._pagination = pagination
        self._page_size = page_size
        self._max_pages = max_pages
        self._batch_rows = batch_rows
        self._records: Optional[list[dict]] = None

    def source_id(self) -> str:
        return f"api|{self._url}"

    def estimate_rows(self) -> Optional[int]:
        return None  # Unknown without fetching

    def _fetch_all_records(self) -> list[dict]:
        """Fetch all records from the API."""
        if self._records is not None:
            return self._records

        try:
            requests = importlib.import_module("requests")
        except ImportError as err:
            raise ImportError(
                "API adapter requires requests: pip install requests") from err

        records: list[dict] = []

        if self._query:
            # GraphQL mode
            records = self._fetch_graphql(requests)
        else:
            # REST mode
            records = self._fetch_rest(requests)

        self._records = records
        return records

    def _fetch_graphql(self, requests: Any) -> list[dict]:
        """Fetch data via GraphQL query."""
        payload = {"query": self._query}
        resp = requests.post(
            self._url,
            json=payload,
            headers=self._headers,
        )
        resp.raise_for_status()
        data = resp.json()
        return self._extract_from_json(data)

    def _fetch_rest(self, requests: Any) -> list[dict]:
        """Fetch data via REST with optional pagination."""
        records: list[dict] = []

        if self._pagination == "offset":
            for page in range(self._max_pages):
                params = {**self._params, "offset": page * self._page_size, "limit": self._page_size}
                resp = requests.get(self._url, headers=self._headers, params=params)
                resp.raise_for_status()
                data = resp.json()
                page_records = self._extract_from_json(data)
                if not page_records:
                    break
                records.extend(page_records)
                if len(page_records) < self._page_size:
                    break
        elif self._pagination == "cursor":
            cursor = None
            for _ in range(self._max_pages):
                params = {**self._params, "limit": self._page_size}
                if cursor:
                    params["cursor"] = cursor
                resp = requests.get(self._url, headers=self._headers, params=params)
                resp.raise_for_status()
                data = resp.json()
                page_records = self._extract_from_json(data)
                if not page_records:
                    break
                records.extend(page_records)
                cursor = self._extract_cursor(data)
                if not cursor or len(page_records) < self._page_size:
                    break
        else:
            # No pagination: single request
            if self._method == "POST":
                resp = requests.post(
                    self._url, headers=self._headers,
                    params=self._params, json=self._body,
                )
            else:
                resp = requests.get(
                    self._url, headers=self._headers, params=self._params,
                    stream=True,
                )
            resp.raise_for_status()

            # Download with progress indicator
            content = self._download_with_progress(resp)

            # Check if response is HTML (e.g., GitHub blob page)
            content_stripped = content.strip()[:200].lower()
            if content_stripped.startswith('<!doctype html') or content_stripped.startswith('<html'):
                raise FIAEError(
                    code=ErrorCode.DATA_FORMAT_ERROR,
                    safe_message=(
                        "URL returned HTML page instead of raw data. "
                        "For GitHub files, use the raw file URL: "
                        "https://raw.githubusercontent.com/..."
                    ),
                    component="api_adapter",
                    evidence={"url": self._url}
                )

            # Try to parse as JSON first
            try:
                data = json.loads(content)
                records = self._extract_from_json(data)
            except (ValueError, json.JSONDecodeError):
                # Not JSON — treat as text (e.g., CSV file served over HTTP)
                records = self._extract_from_text(content)

        return records

    def _download_with_progress(self, resp) -> str:
        """Download response content with progress indicator and ETA."""
        import time
        import sys

        total = int(resp.headers.get('content-length', 0))
        downloaded = 0
        chunks = []
        start_time = time.time()
        last_update = start_time

        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                chunks.append(chunk)
                downloaded += len(chunk)

                # Update progress every 0.5 seconds
                now = time.time()
                if now - last_update >= 0.5 or (total > 0 and downloaded >= total):
                    last_update = now
                    elapsed = now - start_time
                    speed = downloaded / elapsed if elapsed > 0 else 0

                    if total > 0:
                        # Cap percentage at 100% and handle cases where downloaded > total
                        percent = min((downloaded / total) * 100, 100.0)
                        eta = max((total - downloaded) / speed, 0) if speed > 0 else 0
                        bar_len = 30
                        filled = min(int(bar_len * downloaded / total), bar_len)
                        bar = '█' * filled + '░' * (bar_len - filled)
                        sys.stderr.write(
                            f'\r  ↓ Downloading: [{bar}] {percent:.1f}% '
                            f'{downloaded/1024/1024:.1f}/{total/1024/1024:.1f} MB '
                            f'{speed/1024/1024:.1f} MB/s ETA: {eta:.0f}s'
                        )
                    else:
                        sys.stderr.write(
                            f'\r  ↓ Downloading: {downloaded/1024/1024:.1f} MB '
                            f'{speed/1024/1024:.1f} MB/s'
                        )

        sys.stderr.write('\n')
        sys.stderr.flush()
        return b''.join(chunks).decode('utf-8', errors='replace')

    def dialect_report(self) -> dict[str, Any]:
        return {
            "encoding": "utf-8",
            "delimiter": ",",
            "has_header": True,
            "modal_width": 0,
            "malformed_fraction": 0.0,
        }

    def _extract_from_text(self, text: str) -> list[dict]:
        """Extract records from plain text (CSV/TSV) response."""
        import csv
        import io
        import sys

        text = text.strip()
        if not text:
            return []

        # Detect delimiter
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(text[:1024])
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","

        # Remove CSV field size limit
        try:
            csv.field_size_limit(sys.maxsize)
        except OverflowError:
            csv.field_size_limit(2**31 - 1)

        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        rows = list(reader)

        if not rows:
            return []

        # Use first row as header if it looks like headers
        header = rows[0]
        data_rows = rows[1:]

        # Check if first row looks like headers (contains non-numeric values)
        has_header = False
        if data_rows:
            try:
                [float(v) for v in data_rows[0]]
                # If first data row is all numeric, assume no header
                has_header = False
            except ValueError:
                has_header = True

        if not has_header:
            # Generate column names
            header = [f"col_{i}" for i in range(len(rows[0]))]
            data_rows = rows

        records = []
        for row in data_rows:
            record = {}
            for i, val in enumerate(row):
                col_name = header[i] if i < len(header) else f"col_{i}"
                record[col_name] = val.strip() if val else None
            records.append(record)

        return records

    def _extract_from_json(self, data: Any) -> list[dict]:
        """Extract records from JSON response using json_path."""
        if self._json_path:
            parts = self._json_path.split(".")
            current = data
            for part in parts:
                if isinstance(current, dict):
                    current = current.get(part, [])
                elif isinstance(current, list) and part.isdigit():
                    idx = int(part)
                    current = current[idx] if idx < len(current) else []
                else:
                    return []
            if isinstance(current, list):
                return [r for r in current if isinstance(r, dict)]
            if isinstance(current, dict):
                return [current]
            return []
        # Default: if top-level is a list
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        return []

    def _extract_cursor(self, data: Any) -> Optional[str]:
        """Extract pagination cursor from response."""
        if isinstance(data, dict):
            for key in ("next_cursor", "cursor", "nextPageToken", "pagination_token"):
                if key in data:
                    return str(data[key])
        return None

    def scan(
        self,
        projection: Optional[list[str]] = None,
        predicate: Optional[Any] = None,
    ) -> Iterator[RowBatch]:
        """Fetch and stream API data."""
        import time
        import sys

        records = self._fetch_all_records()

        if not records:
            return

        # Build columns
        all_keys = set()
        for r in records:
            all_keys.update(r.keys())
        keys = sorted(all_keys)
        if projection:
            keys = [k for k in keys if k in projection]

        columns: dict[str, list] = {k: [] for k in keys}
        count = 0
        total = len(records)
        start_time = time.time()
        last_update = start_time

        for record in records:
            for k in keys:
                val = record.get(k)
                columns[k].append(None if val is None else str(val))
            count += 1

            now = time.time()
            if now - last_update >= 0.5 or count == total:
                last_update = now
                elapsed = now - start_time
                speed = count / elapsed if elapsed > 0 else 0
                eta = min((total - count) / speed, 9999) if speed > 0 else 9999
                percent = min((count / total) * 100, 100.0) if total > 0 else 0.0
                bar_len = 30
                filled = min(int(bar_len * count / total), bar_len) if total > 0 else 0
                bar = '█' * filled + '░' * (bar_len - filled)
                try:
                    sys.stderr.write(
                        f'\r  ⚙ Processing: [{bar}] {percent:.1f}% '
                        f'{count}/{total} rows '
                        f'{speed:.0f} rows/s ETA: {eta:.0f}s'
                    )
                    sys.stderr.flush()
                except Exception:
                    pass

            if count >= self._batch_rows:
                if any(columns.values()):
                    yield RowBatch(columns=columns)
                columns = {k: [] for k in keys}
                count = 0

        sys.stderr.write('\n')
        sys.stderr.flush()

        if count > 0 and any(columns.values()):
            yield RowBatch(columns=columns)

    def supports_pushdown(self) -> bool:
        return True  # APIs can filter server-side

    def fingerprint_material(self) -> bytes:
        material = f"{self._url}|{self._method}|{json.dumps(self._params, sort_keys=True)}"
        return hashlib.sha256(material.encode()).digest()
