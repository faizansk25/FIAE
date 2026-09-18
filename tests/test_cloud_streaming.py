"""Tests for the CloudAdapter streaming scan logic (no network required).

These tests patch the provider SDK layer so the incremental CSV/JSONL
streaming and batching behaviour can be verified deterministically.
"""

from __future__ import annotations


from fiae.intake.adapter_cloud import CloudAdapter


def _make_csv_adapter(csv_text: str, batch_rows: int = 2) -> CloudAdapter:
    """Adapter whose provider body is an in-memory byte stream."""
    adapter = CloudAdapter("s3://bucket/data.csv", batch_rows=batch_rows)

    class FakeBody:
        def __init__(self, data: bytes, chunk: int = 7) -> None:
            self._data = data
            self._chunk = chunk
            self._pos = 0

        def read(self, n: int = -1) -> bytes:
            if n is None or n < 0:
                n = self._chunk
            out = self._data[self._pos : self._pos + n]
            self._pos += len(out)
            return out

    import unittest.mock as mock

    # estimate_bytes must return the payload size for the fake provider
    adapter._size = len(csv_text.encode())

    def fake_iter_chunks(chunk_size: int = 1024 * 256):
        body = FakeBody(csv_text.encode())
        while True:
            chunk = body.read(chunk_size)
            if not chunk:
                break
            yield chunk

    adapter._iter_chunks = fake_iter_chunks  # type: ignore[method-assign]
    _ = mock  # keep import local
    return adapter


def test_streaming_csv_batches_correctly():
    """Rows are emitted in chunks of batch_rows (regression: counter never incremented)."""
    text = "a,b\n1,2\n3,4\n5,6\n7,8\n9,10\n"
    adapter = _make_csv_adapter(text, batch_rows=2)
    batches = list(adapter.scan())
    total = sum(len(b.columns["a"]) for b in batches)
    assert total == 5
    # With batch_rows=2 we expect ceil(5/2) = 3 batches
    assert len(batches) == 3


def test_streaming_csv_preserves_zero_values():
    """Regression: '0' must not be coerced to None (was `val if val else None`)."""
    text = "a,b\n0,x\n, y\n"
    adapter = _make_csv_adapter(text, batch_rows=10)
    batch = next(iter(adapter.scan()))
    assert batch.columns["a"] == ["0", None]  # '0' preserved, '' -> None
    assert batch.columns["b"] == ["x", " y"]


def test_streaming_csv_header_always_first_row():
    """First row is the header (adapter contract), data rows follow."""
    text = "1,2\n3,4\n"
    adapter = _make_csv_adapter(text, batch_rows=1)
    batches = list(adapter.scan())
    # '1,2' is the header row; data starts at '3,4'
    assert batches[0].columns["1"] == ["3"]


def test_streaming_csv_long_values_across_chunk_boundary():
    """Multi-byte reads split mid-row; csv.reader must still parse correctly."""
    text = "name,desc\n" + "abc," + "x" * 100 + "\n" + "def," + "y" * 50 + "\n"
    adapter = _make_csv_adapter(text, batch_rows=1)
    batches = list(adapter.scan())
    names = [v for b in batches for v in b.columns["name"]]
    descs = [v for b in batches for v in b.columns["desc"]]
    assert names == ["abc", "def"]
    assert descs[0] == "x" * 100


def test_streaming_jsonl_batches():
    """JSONL scan is incremental and batches by batch_rows."""
    adapter = CloudAdapter("s3://bucket/data.jsonl", batch_rows=2)
    text = '\n'.join(
        f'{{"a": {i}, "b": "v{i}"}}' for i in range(5)
    )

    class FakeBody:
        def __init__(self, data: bytes) -> None:
            self._data = data
            self._pos = 0

        def read(self, n: int = -1) -> bytes:
            step = 11
            out = self._data[self._pos : self._pos + step]
            self._pos += len(out)
            return out

    def fake_iter_chunks(chunk_size: int = 1024 * 256):
        body = FakeBody(text.encode())
        while True:
            chunk = body.read(chunk_size)
            if not chunk:
                break
            yield chunk

    adapter._iter_chunks = fake_iter_chunks  # type: ignore[method-assign]
    batches = list(adapter.scan())
    total = sum(len(b.columns["a"]) for b in batches)
    assert total == 5
    assert len(batches) == 3  # ceil(5/2)


def test_provider_detection():
    for uri, provider in [
        ("s3://b/k", "s3"),
        ("gs://b/k", "gcs"),
        ("az://c/k", "azure"),
        ("abfss://c/k", "azure"),
        ("wasbs://c/k", "azure"),
    ]:
        assert CloudAdapter(uri)._provider == provider
