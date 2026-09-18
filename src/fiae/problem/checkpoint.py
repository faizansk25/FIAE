"""Pipeline checkpointing and resume (doc 01 NFR-006).

Stage checkpoints are resumable when semantic correctness is preserved.
Each stage writes a checkpoint file; on resume, completed stages are skipped.

Normative source: doc 01 NFR-006 "Recoverability".
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from ..ids import content_hash, new_id


@dataclass
class StageCheckpoint:
    """Checkpoint for one pipeline stage."""

    stage_name: str
    stage_index: int
    status: str = "completed"  # "completed" | "failed" | "skipped"
    result: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    checksum: str = ""

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if not self.checksum:
            self.checksum = content_hash({"stage": self.stage_name, "result": self.result})


@dataclass
class PipelineCheckpoint:
    """Full pipeline checkpoint state."""

    pipeline_id: str = ""
    run_id: str = ""
    source_fingerprint: str = ""
    config_hash: str = ""
    stages: list[StageCheckpoint] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self):
        if not self.pipeline_id:
            self.pipeline_id = new_id("pipe")
        if self.created_at == 0.0:
            self.created_at = time.time()
        self.updated_at = time.time()

    def stage_completed(self, name: str) -> bool:
        return any(s.stage_name == name and s.status == "completed" for s in self.stages)

    def get_stage_result(self, name: str) -> Optional[dict]:
        for s in self.stages:
            if s.stage_name == name and s.status == "completed":
                return s.result
        return None

    def add_stage(self, checkpoint: StageCheckpoint) -> None:
        # Replace existing checkpoint for same stage
        self.stages = [s for s in self.stages if s.stage_name != checkpoint.stage_name]
        self.stages.append(checkpoint)
        self.updated_at = time.time()

    def completed_stages(self) -> list[str]:
        return [s.stage_name for s in self.stages if s.status == "completed"]

    def to_dict(self) -> dict:
        return {
            "pipeline_id": self.pipeline_id,
            "run_id": self.run_id,
            "source_fingerprint": self.source_fingerprint,
            "config_hash": self.config_hash,
            "stages": [asdict(s) for s in self.stages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PipelineCheckpoint:
        stages = [StageCheckpoint(**s) for s in d.get("stages", [])]
        return cls(
            pipeline_id=d.get("pipeline_id", ""),
            run_id=d.get("run_id", ""),
            source_fingerprint=d.get("source_fingerprint", ""),
            config_hash=d.get("config_hash", ""),
            stages=stages,
            created_at=d.get("created_at", 0.0),
            updated_at=d.get("updated_at", 0.0),
        )


class CheckpointManager:
    """Manages checkpoint persistence (doc 01 NFR-006)."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _path(self, checkpoint_id: str) -> str:
        return os.path.join(self.base_dir, f"{checkpoint_id}.json")

    def save(self, checkpoint: PipelineCheckpoint) -> str:
        """Save checkpoint to disk. Returns the file path."""
        path = self._path(checkpoint.pipeline_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)
        return path

    def load(self, checkpoint_id: str) -> Optional[PipelineCheckpoint]:
        """Load checkpoint from disk."""
        path = self._path(checkpoint_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return PipelineCheckpoint.from_dict(data)

    def list_checkpoints(self) -> list[str]:
        """List all checkpoint IDs in the base directory."""
        result = []
        for fname in os.listdir(self.base_dir):
            if fname.endswith(".json"):
                result.append(fname[:-5])
        return result

    def delete(self, checkpoint_id: str) -> bool:
        path = self._path(checkpoint_id)
        if os.path.exists(path):
            os.unlink(path)
            return True
        return False
