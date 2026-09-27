"""C-5 fix: Stage A/B leakage detectors wired into phase_validate.

Proves the real deterministic (Stage A) and statistical (Stage B) detectors
from problem/leakage.py actually fire through the canonical pipeline's
validation phase, with false-positive guards on clean data.
"""

from __future__ import annotations

from fiae.pipeline.canonical import (
    RunContext,
    phase_intake,
    phase_validate,
)


def _run_phase_on_csv(tmp_path, content: str, target: str = "y"):
    csv_path = tmp_path / "leak.csv"
    csv_path.write_text(content, encoding="utf-8")
    ctx = RunContext()
    ctx.config_snapshot["source"] = str(csv_path)
    ctx.run_dir = str(tmp_path)
    intake = phase_intake(ctx, str(csv_path), target)
    return phase_validate(ctx, intake, target)


def _clean_csv(n: int = 60) -> str:
    """Binary target with noisy features; no deterministic or statistical leak."""
    rows = ["a,b,noise,y"]
    for i in range(n):
        y = (i * 3 + (i // 4)) % 2  # irregular binary sequence, not alternating
        a = (i * 7 + 3) % 5
        b = y if (i % 3) else 1 - y  # ~33% label noise
        rows.append(f"{a},{b},{(i * 5) % 3},{y}")
    return "\n".join(rows) + "\n"


class TestStageADetectorsFireThroughPhaseValidate:
    def test_exact_target_copy_is_hard_rejected(self, tmp_path):
        rows = ["copy,y,noise"]
        for i in range(40):
            y = i % 2
            rows.append(f"{y},{y},{i}")
        validation = _run_phase_on_csv(tmp_path, "\n".join(rows) + "\n")
        leak_cols = [f for f in validation.leakage_flags if f["column"] == "copy"]
        assert leak_cols, "exact target copy must be flagged"
        assert any(
            "target_copy" in f["type"] for f in leak_cols
        ), f"expected target_copy finding, got {[f['type'] for f in leak_cols]}"
        hard = [f for f in leak_cols if f["severity"].endswith("hard_reject")]
        assert hard, "target copy must be hard_reject severity"
        assert hard[0]["action"] == "reject"

    def test_inverted_target_copy_is_hard_rejected(self, tmp_path):
        rows = ["inv,y,noise"]
        for i in range(40):
            y = i % 2
            rows.append(f"{1 - y},{y},{i}")
        validation = _run_phase_on_csv(tmp_path, "\n".join(rows) + "\n")
        flags = validation.leakage_flags
        assert any(
            "inverted_target_copy" in f["type"] for f in flags
        ), f"expected inverted_target_copy, got {[f['type'] for f in flags]}"

    def test_target_embedded_in_string_is_hard_rejected(self, tmp_path):
        rows = ["note,y,noise"]
        for i in range(40):
            y = i % 2
            rows.append(f"item_{y},{y},{i}")
        validation = _run_phase_on_csv(tmp_path, "\n".join(rows) + "\n")
        flags = validation.leakage_flags
        assert any(
            "target_embedded_in_string" in f["type"] for f in flags
        ), f"expected target_embedded_in_string, got {[f['type'] for f in flags]}"

    def test_identifier_bijection_is_hard_rejected(self, tmp_path):
        rows = ["row_id,y,feat"]
        for i in range(40):
            y = i % 2
            rows.append(f"id_{i},{y},{i % 3}")
        validation = _run_phase_on_csv(tmp_path, "\n".join(rows) + "\n")
        flags = [f for f in validation.leakage_flags if f["column"] == "row_id"]
        assert flags, "identifier bijection must be flagged"
        assert any(
            "deterministic_bijection" in f["type"] for f in flags
        ), f"expected deterministic_bijection, got {[f['type'] for f in flags]}"

    def test_boolean_renamed_copy_is_caught(self, tmp_path):
        rows = ["flag,y,noise"]
        for i in range(40):
            y = i % 2
            rows.append(f"{'yes' if y else 'no'},{y},{i}")
        validation = _run_phase_on_csv(tmp_path, "\n".join(rows) + "\n")
        flags = [f for f in validation.leakage_flags if f["column"] == "flag"]
        assert any(
            "target_copy" in f["type"] for f in flags
        ), f"expected boolean-renamed target copy as target_copy, got {[f['type'] for f in flags]}"

    def test_clean_data_raises_no_stage_ab_findings(self, tmp_path):
        validation = _run_phase_on_csv(tmp_path, _clean_csv(60))
        stage_ab = [
            f for f in validation.leakage_flags
            if f["type"] != "potential_identifier"
        ]
        assert stage_ab == [], f"clean data must raise no findings, got {stage_ab}"

    def test_heuristic_identifier_flag_still_present(self, tmp_path):
        """The legacy distinct-ratio heuristic must still run alongside the detectors."""
        rows = ["row_id,y,feat"]
        for i in range(40):
            y = i % 2
            rows.append(f"id_{i},{y},{i % 3}")
        validation = _run_phase_on_csv(tmp_path, "\n".join(rows) + "\n")
        assert any(
            f["type"] == "potential_identifier" and f["column"] == "row_id"
            for f in validation.leakage_flags
        )


class TestStageBDetectorsFireThroughPhaseValidate:
    def test_near_perfect_auc_raises_review(self, tmp_path):
        rows = ["score,y,noise"]
        for i in range(40):
            y = i % 2
            # Strong but not deterministic: positives all score 2.0, negatives
            # mostly 0.0 with a few 0.5 -> AUC = 1.0 (>= 0.98 suspicion) but
            # values never equal the target and cardinality stays low, so
            # Stage A passes and Stage B statistical suspicion fires.
            s = 2.0 if y else (0.5 if i % 10 == 0 else 0.0)
            rows.append(f"{s},{y},{i}")
        validation = _run_phase_on_csv(tmp_path, "\n".join(rows) + "\n")
        flags = [f for f in validation.leakage_flags if f["column"] == "score"]
        assert flags, "near-perfect AUC feature must be flagged"
        assert any(
            "near_perfect_auc" in f["type"] for f in flags
        ), f"expected near_perfect_auc, got {[f['type'] for f in flags]}"
        assert all(
            f["severity"] == "review_required" for f in flags
        ), "Stage B findings must be review_required, never hard reject"

    def test_missingness_determines_target(self, tmp_path):
        rows = ["m,y"]
        for i in range(40):
            if i < 20:
                rows.append(",0")  # missing -> always y=0
            else:
                rows.append(f"{i},1")  # present -> always y=1
        validation = _run_phase_on_csv(tmp_path, "\n".join(rows) + "\n")
        flags = [f for f in validation.leakage_flags if f["column"] == "m"]
        assert any(
            "missingness_determines_target" in f["type"] for f in flags
        ), f"expected missingness_determines_target, got {[f['type'] for f in flags]}"

    def test_finding_structure_is_complete(self, tmp_path):
        rows = ["copy,y"]
        for i in range(40):
            rows.append(f"{i % 2},{i % 2}")
        validation = _run_phase_on_csv(tmp_path, "\n".join(rows) + "\n")
        flags = [f for f in validation.leakage_flags if f["column"] == "copy"]
        assert flags
        for f in flags:
            assert set(f) >= {"column", "type", "leakage_class", "severity", "action", "evidence"}
            assert isinstance(f["evidence"], dict)

    def test_findings_carry_finding_id_format(self, tmp_path):
        rows = ["copy,y"]
        for i in range(40):
            rows.append(f"{i % 2},{i % 2}")
        validation = _run_phase_on_csv(tmp_path, "\n".join(rows) + "\n")
        flags = [f for f in validation.leakage_flags if f["column"] == "copy"]
        assert any(f["type"].startswith("L|") for f in flags), (
            f"finding ids must use L|kind|subject format, got {[f['type'] for f in flags]}"
        )
