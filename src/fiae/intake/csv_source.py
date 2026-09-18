"""CSV data source adapter (doc 02 CSV parsing algorithm).

Robust bounded sniffing: encoding/BOM, delimiter, header, physical row width
stability. Seekable regular files are sampled at multiple positions, never
head-only (DuckDB-style, doc 14). Malformed-row accounting is explicit; a
malformed fraction above policy aborts or lowers confidence (step 0011).
"""

from __future__ import annotations

import csv
import hashlib
import io
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Optional

from ..errors import ErrorCode, FIAEError
from .base import RowBatch, SamplePlan

CANDIDATE_DELIMITERS: tuple[str, ...] = (",", "\t", ";", "|")
PROBE_BYTES = 65_536
DEFAULT_BATCH_ROWS = 2048


def detect_encoding(raw: bytes) -> str:
    """BOM-aware encoding detection; prefer UTF-8 (step 0011, rule 2)."""
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def _modal_widths(rows: list[list[str]]) -> tuple[int, float]:
    widths = [len(r) for r in rows]
    if not widths:
        return 0, 0.0
    modal, count = Counter(widths).most_common(1)[0]
    return modal, count / len(widths)


def detect_dialect(
    text: str, override: Optional[str] = None
) -> dict[str, Any]:
    """Evaluate delimiter candidates; respect quoting; require stable width."""
    candidates = (override,) if override else CANDIDATE_DELIMITERS
    best: Optional[dict[str, Any]] = None
    for delim in candidates:
        try:
            rows = [
                r for r in csv.reader(io.StringIO(text), delimiter=delim) if r
            ]
        except csv.Error:
            continue
        if len(rows) < 2:
            continue
        modal, stability = _modal_widths(rows)
        if modal == 0:
            continue
        # Row-width stability is the confidence signal; a wider modal width is
        # a tiebreak so single-column files are not spuriously ambiguous.
        score = stability * 10.0 + (1.0 if modal > 1 else 0.0)
        if best is None or score > best["score"] + 1e-9:
            best = {
                "delimiter": delim,
                "score": score,
                "modal_width": modal,
                "width_stability": stability,
                "confidence": round(min(1.0, stability), 4),
            }
    if best is None:
        raise FIAEError(
            code=ErrorCode.SCHEMA_AMBIGUITY,
            safe_message="CSV dialect detection failed: no stable delimiter found.",
            component="csv_source",
        )
    best.pop("score")
    return best


def _looks_typed(value: str) -> bool:
    from .typing_engine import parse_scalar_candidates

    cands = parse_scalar_candidates(value)
    return any(c in ("integer", "float", "date", "timestamp", "boolean") for c in cands)


def detect_header(rows: list[list[str]]) -> tuple[bool, float]:
    """Infer header only with adequate confidence (step 0011, rule 6)."""
    if len(rows) < 2:
        return False, 0.0
    first, second = rows[0], rows[1]
    if len(first) != len(second) or not first:
        return False, 0.0
    if any(_looks_typed(v) for v in first):
        # First row already looks like data (typed values).
        return False, 0.9
    typed_in_second = any(_looks_typed(v) for v in second)
    confidence = 0.6
    if typed_in_second:
        confidence = 0.95
    if len(set(first)) < len(first):
        confidence -= 0.4  # duplicate header names weaken the hypothesis
    return confidence >= 0.5, max(0.0, min(1.0, confidence))


class CsvDataSourceAdapter:
    """CSV adapter implementing the DataSourceAdapter contract (doc 02)."""

    def __init__(
        self,
        path: str | Path,
        *,
        delimiter: Optional[str] = None,
        has_header: Optional[bool] = None,
        encoding: Optional[str] = None,
        batch_rows: int = DEFAULT_BATCH_ROWS,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FIAEError(
                code=ErrorCode.DATA_FORMAT_ERROR,
                safe_message="Source is not a readable file.",
                component="csv_source",
                evidence={"path": self.path.name},
            )
        self._size = self.path.stat().st_size
        self._start_size = self._size
        with open(self.path, "rb") as fh:
            probe = fh.read(PROBE_BYTES)
        self.encoding = encoding or detect_encoding(probe)
        text = probe.decode(self.encoding, errors="replace")
        self._probe_bytes = len(probe)
        self._probe_lines = max(text.count("\n"), 0)
        try:
            dialect = detect_dialect(text, override=delimiter)
        except FIAEError:
            # Tiny/degenerate file: fail later with a precise error, not here.
            if text.count("\n") < 2:
                first_row = next(csv.reader(io.StringIO(text)), [])
                dialect = {
                    "delimiter": delimiter or ",",
                    "confidence": 0.0,
                    "modal_width": len(first_row),
                    "width_stability": 0.0,
                }
            else:
                raise
        self.delimiter = dialect["delimiter"]
        self._dialect_confidence = dialect["confidence"]
        self._modal_width = dialect["modal_width"]
        if has_header is not None:
            # User declaration wins (step 0011, rule 6).
            self.has_header = has_header
            self._header_confidence = 1.0
        else:
            self.has_header, self._header_confidence = detect_header(
                self._parse_text(text)
            )
        self.batch_rows = batch_rows
        self.column_names: list[str] = self._resolve_columns(self._parse_text(text))
        self.malformed_rows = 0
        self.parsed_rows = 0

    # -- metadata ------------------------------------------------------------
    def source_id(self) -> str:
        return f"csv|{self.path.resolve().as_posix()}"

    def schema_hint(self) -> Optional[dict[str, Any]]:
        return None  # CSV is not self-describing (doc 08)

    def estimate_rows(self) -> Optional[int]:
        """Estimate from average line length over the bounded probe."""
        if self._probe_lines == 0:
            return None
        avg_line = self._probe_bytes / self._probe_lines
        return int(self._size / avg_line)

    def estimate_bytes(self) -> Optional[int]:
        return self._size

    def supports_seek_sampling(self) -> bool:
        return True

    def supports_pushdown(self) -> bool:
        return False

    def dialect_report(self) -> dict[str, Any]:
        return {
            "encoding": self.encoding,
            "delimiter": self.delimiter,
            "has_header": self.has_header,
            "header_confidence": self._header_confidence,
            "dialect_confidence": self._dialect_confidence,
            "modal_width": self._modal_width,
            "malformed_fraction": self.malformed_fraction,
        }

    @property
    def malformed_fraction(self) -> float:
        return self.malformed_rows / max(self.parsed_rows, 1)

    # -- internals ------------------------------------------------------------
    def _resolve_columns(self, rows: list[list[str]]) -> list[str]:
        width = self._modal_width
        if self.has_header and rows:
            names = [(v or "").strip() for v in rows[0][:width]]
            seen: dict[str, int] = {}
            resolved = []
            for n in names:
                if not n:
                    n = "column"
                if n in seen:
                    seen[n] += 1
                    resolved.append(f"{n}_{seen[n]}")
                else:
                    seen[n] = 0
                    resolved.append(n)
            while len(resolved) < width:
                resolved.append(f"column_{len(resolved) + 1}")
            return resolved
        return [f"column_{i + 1}" for i in range(width)]

    def _parse_text(self, text: str) -> list[list[str]]:
        return [r for r in csv.reader(io.StringIO(text), delimiter=self.delimiter) if r]

    def _iter_file_rows(self, offset: int, nbytes: Optional[int]) -> Iterator[list[str]]:
        """Yield parsed data rows from a file region; skip a partial leading line."""
        with open(self.path, "rb") as fh:
            fh.seek(offset)
            if offset > 0:
                fh.readline()  # discard the possibly partial line under the seek
            data = fh.read(nbytes) if nbytes is not None else fh.read()
        text = data.decode(self.encoding, errors="replace")
        width = self._modal_width
        skip_header = offset == 0 and self.has_header
        for row in csv.reader(io.StringIO(text), delimiter=self.delimiter):
            self.parsed_rows += 1
            if len(row) != width:
                self.malformed_rows += 1
                continue
            if skip_header:
                skip_header = False
                continue
            yield row

    # -- sampling --------------------------------------------------------------
    def sample(self, plan: SamplePlan) -> Iterator[RowBatch]:
        """Multi-region deterministic sampling (doc 02 sampling plan)."""
        rng = random.Random(plan.random_seed)
        if self._size <= plan.block_bytes:
            # Whole source fits one block: read once, never duplicate regions.
            offsets = [0]
        else:
            offsets: list[int] = []
            if plan.include_head:
                offsets.append(0)
            for _ in range(plan.n_middle):
                offsets.append(max(0, self._size // 2 - plan.block_bytes // 2))
            if plan.include_tail:
                offsets.append(max(0, self._size - plan.block_bytes))
            for _ in range(plan.n_random):
                offsets.append(rng.randrange(0, self._size - plan.block_bytes))
        batch: list[list[str]] = []
        for offset in offsets:
            for row in self._iter_file_rows(offset, plan.block_bytes):
                batch.append(row)
                if len(batch) >= self.batch_rows:
                    yield self._to_batch(batch)
                    batch = []
        if batch:
            yield self._to_batch(batch)

    def scan(
        self,
        projection: Optional[list[str]] = None,
        predicate: Optional[Any] = None,
    ) -> Iterator[RowBatch]:
        batch: list[list[str]] = []
        for row in self._iter_file_rows(0, None):
            batch.append(row)
            if len(batch) >= self.batch_rows:
                b = self._to_batch(batch, projection)
                batch = []
                if b.columns:
                    yield b
        if batch:
            b = self._to_batch(batch, projection)
            if b.columns:
                yield b

    def _to_batch(
        self, rows: list[list[str]], projection: Optional[list[str]] = None
    ) -> RowBatch:
        cols: dict[str, list[str]] = {}
        for i, name in enumerate(self.column_names):
            if projection is not None and name not in projection:
                continue
            cols[name] = [("" if i >= len(r) else r[i]) for r in rows]
        return RowBatch(columns=cols)

    # -- identity / safety -------------------------------------------------------
    def fingerprint_material(self) -> bytes:
        """Bounded content hashes + size + parser metadata; path is not identity."""
        h = hashlib.sha256()
        h.update(str(self._size).encode())
        with open(self.path, "rb") as fh:
            h.update(hashlib.sha256(fh.read(PROBE_BYTES)).digest())
            if self._size > 2 * PROBE_BYTES:
                fh.seek(self._size // 2)
                h.update(hashlib.sha256(fh.read(PROBE_BYTES)).digest())
                fh.seek(-PROBE_BYTES, io.SEEK_END)
                h.update(hashlib.sha256(fh.read(PROBE_BYTES)).digest())
        h.update(self.delimiter.encode())
        h.update(self.encoding.encode())
        return h.digest()

    def verify_unchanged(self) -> None:
        current = self.path.stat().st_size
        if current != self._start_size:
            raise FIAEError(
                code=ErrorCode.SOURCE_CHANGED_DURING_RUN,
                safe_message="Source file changed during the run.",
                component="csv_source",
                evidence={"initial_bytes": self._start_size, "current_bytes": current},
            )

