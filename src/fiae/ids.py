"""Stable identifiers, canonical serialization, and content fingerprints.

Normative sources: doc 13 (canonical serialization, IDs) and doc 01 NFR-004/005.
- Canonical serialization: stable key order, stable enum values, ephemeral
  timestamps excluded from semantic hashes, normalized number representation,
  UTF-8, content hash.
- Dataset/split/feature identifiers are deterministic under their declared inputs.
Path alone is never identity (doc 02).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
import hashlib
import json
import math
import uuid
from typing import Any, Optional

# Ephemeral fields excluded from semantic hashes by default (doc 13, rule 3).
EPHEMERAL_KEYS = frozenset(
    {
        "timestamp",
        "event_id",
        "created_at",
        "decision_timestamp",
        "wall_time_s",
        "measured_at",
    }
)


def _normalize(obj: Any, exclude: frozenset[str]) -> Any:
    """Recursively normalize a value into a canonically serializable structure."""
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, enum.Enum):
        return _normalize(obj.value, exclude)
    if isinstance(obj, float):
        # Normalize permitted number representation; explicit non-finite markers.
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return repr(obj)
    if isinstance(obj, bytes):
        return obj.hex()
    if isinstance(obj, (_dt.datetime, _dt.date, _dt.time)):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return obj.hex
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out: dict[str, Any] = {"__type__": type(obj).__name__}
        for f in dataclasses.fields(obj):
            if f.name in exclude:
                continue
            out[f.name] = _normalize(getattr(obj, f.name), exclude)
        return out
    if isinstance(obj, dict):
        return {
            str(k): _normalize(v, exclude)
            for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
            if str(k) not in exclude
        }
    if isinstance(obj, (list, tuple)):
        return [_normalize(v, exclude) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(
            (_normalize(v, exclude) for v in obj), key=lambda v: json.dumps(v, default=str)
        )
    # Fallback: stable string form of the value.
    return f"<{type(obj).__name__}:{obj!s}>"


def canonical_json(obj: Any, *, exclude_keys: Optional[frozenset[str]] = None) -> str:
    """Canonical UTF-8 JSON string with stable key order and excluded ephemerals."""
    excl = EPHEMERAL_KEYS if exclude_keys is None else exclude_keys
    normalized = _normalize(obj, frozenset(excl))
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def content_hash(obj: Any, *, exclude_keys: Optional[frozenset[str]] = None) -> str:
    """SHA-256 hex digest of the canonical serialization (deterministic ID material)."""
    payload = canonical_json(obj, exclude_keys=exclude_keys).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def new_id(prefix: str) -> str:
    """Opaque, unique identifier for run/stage/trial/artifact instances.

    Note: instance IDs are unique, not content-derived; content-derived identity
    (fingerprints) uses content_hash so IDs are stable under declared inputs.
    """
    return f"{prefix}_{uuid.uuid4().hex}"


def dataset_fingerprint(material: bytes, schema_fingerprint: str, parser_config: Any) -> str:
    """Dataset identity from content material + canonical schema + parser config.

    Path alone is not identity (doc 02). Combine stable source version/content
    material, canonical schema, parser config, and bounded content hashes.
    """
    combined = {
        "material_sha256": hashlib.sha256(material).hexdigest(),
        "schema_fingerprint": schema_fingerprint,
        "parser_config": parser_config,
    }
    return "ds_" + content_hash(combined)


def split_fingerprint(validation_plan_material: Any) -> str:
    """Split identity from the declared split definition (strategy, seed, cutoffs...)."""
    return "split_" + content_hash(validation_plan_material)


def feature_id_for(node_material: Any) -> str:
    """Deterministic feature identity from canonical node content (doc 04)."""
    return "f_" + content_hash(node_material)[:24]
