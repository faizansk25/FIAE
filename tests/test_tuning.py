"""Tests for experience-driven funnel policy tuning (fiae.tuning)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fiae.experience.case import (
    CaseDiagnosis,
    CaseRecord,
    FailureTag,
)
from fiae.contracts import TrialStatus
from fiae.experience.store import ExperienceStore, StoreConfig
from fiae.funnel import FunnelPolicy
from fiae.tuning import tune_policy_from_experience

_counter = [0]


def _case(success: bool, tags: list) -> CaseRecord:
    _counter[0] += 1
    diag = CaseDiagnosis(
        status=TrialStatus.COMPLETED,
        failure_tags=[] if success else list(tags),
    )
    return CaseRecord(
        case_id=f"case_{_counter[0]}",
        dataset_fingerprint=f"ds_{_counter[0]:06d}",
        schema_fingerprint="schema_test",
        diagnosis=diag,
    )


class TestTuning(unittest.TestCase):
    def setUp(self):
        fd, self.db = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.store = ExperienceStore(StoreConfig(db_path=self.db))

    def tearDown(self):
        self.store.close()
        os.remove(self.db)

    def test_insufficient_evidence_returns_base_policy(self):
        for _i in range(3):
            self.store.write_case(_case(False, [FailureTag.REDUNDANT]))
        base = FunnelPolicy()
        report = tune_policy_from_experience(self.store, min_cases=5)
        self.assertEqual(report.cases_analyzed, 3)
        self.assertEqual(report.adjustments, [])
        self.assertEqual(report.policy, base)

    def test_redundant_tag_tightens_correlation(self):
        for _i in range(8):
            self.store.write_case(_case(False, [FailureTag.REDUNDANT]))
        base = FunnelPolicy()
        report = tune_policy_from_experience(self.store)
        fields = {a.field for a in report.adjustments}
        self.assertIn("f5_max_correlation", fields)
        self.assertLess(report.policy.f5_max_correlation,
                        base.f5_max_correlation)
        self.assertGreaterEqual(report.policy.f5_max_correlation, 0.80)

    def test_overfit_tag_reduces_depth(self):
        for _i in range(8):
            self.store.write_case(_case(False, [FailureTag.OVERFIT_HIGH_VARIANCE]))
        base = FunnelPolicy(max_depth=2)
        report = tune_policy_from_experience(self.store, base_policy=base)
        self.assertLess(report.policy.max_depth, base.max_depth)
        self.assertGreaterEqual(report.policy.max_depth, 1)

    def test_underfit_tag_increases_depth(self):
        for _i in range(8):
            self.store.write_case(_case(False, [FailureTag.UNDERFIT_HIGH_BIAS]))
        base = FunnelPolicy()
        report = tune_policy_from_experience(self.store)
        self.assertGreater(report.policy.max_depth, base.max_depth)

    def test_all_success_no_failures_relaxes(self):
        for _i in range(12):
            self.store.write_case(_case(True, []))
        base = FunnelPolicy()
        report = tune_policy_from_experience(self.store)
        self.assertGreaterEqual(report.success_rate, 0.90)
        self.assertLess(report.policy.f5_min_mutual_info_ratio,
                        base.f5_min_mutual_info_ratio)

    def test_deterministic(self):
        for _i in range(8):
            self.store.write_case(_case(False, [FailureTag.REDUNDANT,
                                            FailureTag.RESOURCE_RAM]))
        r1 = tune_policy_from_experience(self.store)
        r2 = tune_policy_from_experience(self.store)
        self.assertEqual(r1.to_dict(), r2.to_dict())

    def test_policy_roundtrip_via_json(self):
        import dataclasses
        import json
        for _i in range(8):
            self.store.write_case(_case(False, [FailureTag.OVERFIT_HIGH_VARIANCE]))
        report = tune_policy_from_experience(self.store)
        blob = json.dumps(dataclasses.asdict(report.policy))
        revived = FunnelPolicy(**json.loads(blob))
        self.assertEqual(report.policy, revived)


if __name__ == "__main__":
    unittest.main()
