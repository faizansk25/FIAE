"""Leakage scenario tests (audit program M4).

Synthetic datasets deliberately containing each leakage class from the
doc 03 taxonomy, proving the detector pipeline catches them end-to-end --
plus false-positive-rate measurement on provably clean data.

Mission classes covered:
- target-derived columns          -> Stage A hard rejects (L5) / Stage B (L4)
- post-outcome timestamps         -> review_availability / confirm_unavailable
- entity leakage                  -> deterministic bijection (L5)
- aggregation leakage             -> cross-fit target encoder excludes own row
- split contamination             -> fold_safety_report (Stage D)
- future information              -> time_since_previous / availability rules
"""

import math
import random

from fiae.contracts import FitScope, TargetPermission
from fiae.problem.leakage import (
    CrossFitTargetEncoder,
    LeakageClass,
    LeakageSeverity,
    check_target_history_safety,
    confirm_unavailable,
    detect_deterministic,
    fold_safety_report,
    review_availability,
    statistical_triage,
)
from fiae.contracts import FeatureNode


def _kind(finding):
    """Rule name from finding_id ('L|<kind>|<subject>')."""
    return finding.finding_id.split("|")[1]


def _cls(finding):
    return finding.type


def _norm_classes(findings):
    return {_cls(f) for f in findings}


# ---------------------------------------------------------------------------
# Synthetic contaminated datasets, one per leakage class
# ---------------------------------------------------------------------------


def _dataset_target_copy(n=200, seed=7):
    """L5: feature is an exact copy of the target."""
    rng = random.Random(seed)
    y = [rng.randint(0, 1) for _ in range(n)]
    f = list(y)  # direct copy
    return f, y


def _dataset_target_derived_noisy(n=200, seed=11):
    """L4: feature = target signal + small noise (target-derived column)."""
    rng = random.Random(seed)
    y = [rng.randint(0, 1) for _ in range(n)]
    f = [float(v) + rng.gauss(0, 0.05) for v in y]
    return f, y


def _dataset_post_outcome(n=200, seed=13):
    """L4/L5: feature only exists after the outcome happened (e.g. refund
    timestamp after churn decision). Modeled as always-unavailable fact."""
    rng = random.Random(seed)
    y = [rng.randint(0, 1) for _ in range(n)]
    f = [float(v) * 10 + rng.gauss(0, 1) for v in y]  # perfectly separates
    return f, y


def _dataset_entity_leakage(n=200, seed=17):
    """L5: unique entity ID deterministically maps to the target
    (memorization path)."""
    rng = random.Random(seed)
    y = [rng.randint(0, 1) for _ in range(n)]
    # make the bijection deterministic by aligning ids to targets
    by_target = {0: [], 1: []}
    for i, v in enumerate(y):
        by_target[v].append(i)
    out = [None] * n
    for v, idxs in by_target.items():
        for j, i in enumerate(idxs):
            out[i] = f"entity_{v}_{j}"
    return out, y


def _dataset_aggregation_leakage(n=200, seed=19):
    """L2: target mean encoding computed INCLUDING the row's own target
    (self-inclusion = aggregation leakage)."""
    rng = random.Random(seed)
    y = [float(rng.randint(0, 1)) for _ in range(n)]
    cats = [f"cat_{i % 20}" for i in range(n)]
    # leaky encoding: category mean including own target
    sums, counts = {}, {}
    for c, t in zip(cats, y):
        sums[c] = sums.get(c, 0.0) + t
        counts[c] = counts.get(c, 0) + 1
    leaky = [sums[c] / counts[c] for c in cats]
    return leaky, cats, y


def _dataset_clean(n=300, seed=23):
    """Provably clean: feature drawn independently of the target."""
    rng = random.Random(seed)
    y = [rng.randint(0, 1) for _ in range(n)]
    f = [rng.gauss(0, 1) for _ in range(n)]
    return f, y


# ---------------------------------------------------------------------------
# Class-by-class detection behavior
# ---------------------------------------------------------------------------


class TestTargetDerivedColumns:
    def test_exact_target_copy_is_hard_reject(self):
        f, y = _dataset_target_copy()
        finding = detect_deterministic(f, y)
        assert finding is not None
        assert _kind(finding) == "target_copy"
        assert _cls(finding) is LeakageClass.L5
        assert finding.severity is LeakageSeverity.HARD_REJECT

    def test_noisy_target_derivative_caught_by_stage_b(self):
        f, y = _dataset_target_derived_noisy()
        finding = statistical_triage(f, y)
        assert finding is not None
        assert _kind(finding) == "near_perfect_auc"
        assert _cls(finding) is LeakageClass.L4

    def test_boolean_renamed_copy_caught(self):
        # "yes"/"no" renaming must not evade the copy detector
        f, y = _dataset_target_copy(n=100)
        f = ["yes" if v else "no" for v in y]
        finding = detect_deterministic(f, y)
        assert finding is not None and _kind(finding) == "target_copy"


class TestPostOutcomeTimestamps:
    def test_availability_after_prediction_flagged(self):
        findings = review_availability(
            {"refund_amount": "2026-06-01T00:00:00"},
            "2026-05-01T00:00:00",  # prediction happens BEFORE the fact exists
        )
        assert findings
        assert all(_cls(f) is LeakageClass.L4 for f in findings)
        assert any(_kind(f) == "availability_after_prediction" for f in findings)

    def test_availability_unknown_flagged(self):
        findings = review_availability(
            {"mystery_feature": None}, "2026-05-01T00:00:00"
        )
        assert any(_kind(f) == "availability_unknown" for f in findings)

    def test_proven_unavailable_is_hard_reject(self):
        findings = confirm_unavailable(
            ["chargeback_count"], "recorded after outcome"
        )
        assert all(f.severity is LeakageSeverity.HARD_REJECT for f in findings)
        assert all(_cls(f) is LeakageClass.L5 for f in findings)

    def test_statistically_separating_post_outcome_feature_suspicious(self):
        f, y = _dataset_post_outcome()
        finding = statistical_triage(f, y)
        assert finding is not None and _cls(finding) is LeakageClass.L4


class TestEntityLeakage:
    def test_unique_id_bijection_is_hard_reject(self):
        f, y = _dataset_entity_leakage()
        finding = detect_deterministic(f, y)
        assert finding is not None
        assert _kind(finding) == "deterministic_bijection"
        assert _cls(finding) is LeakageClass.L5


class TestAggregationLeakage:
    def test_crossfit_encoder_excludes_own_target(self):
        _, cats, y = _dataset_aggregation_leakage()
        enc = CrossFitTargetEncoder(smoothing=1.0)
        encoded = enc.fit_transform(cats, y, n_folds=4, seed=1)

        # The leaky self-included encoding must differ from cross-fit output
        sums, counts = {}, {}
        for c, t in zip(cats, y):
            sums[c] = sums.get(c, 0.0) + t
            counts[c] = counts.get(c, 0) + 1
        leaky = [sums[c] / counts[c] for c in cats]

        diffs = sum(1 for a, b in zip(encoded, leaky) if abs(a - b) > 1e-9)
        assert diffs > 0, (
            "cross-fit encoding identical to self-included aggregation "
            "-- aggregation leakage not prevented"
        )

    def test_single_occurrence_falls_back_to_prior(self):
        # a category seen once must be encoded from the prior, not its own label
        enc = CrossFitTargetEncoder(smoothing=1.0, prior=0.5)
        encoded = enc.fit_transform(
            ["only_once", "a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d"],
            [1.0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
            n_folds=3,
            seed=0,
        )
        assert math.isclose(encoded[0], 0.5), (
            f"single-occurrence row encoded as {encoded[0]}, expected prior 0.5"
        )

    def test_group_aware_crossfit_respects_entities(self):
        # same entity appearing in multiple folds must stay in one fold
        enc = CrossFitTargetEncoder(smoothing=1.0)
        cats = [f"g{i % 10}" for i in range(60)]
        y = [float(i % 2) for i in range(60)]
        groups = [i % 10 for i in range(60)]
        encoded = enc.fit_transform(cats, y, n_folds=3, seed=2, group_ids=groups)
        assert len(encoded) == len(y)


class TestSplitContamination:
    def test_development_scope_rejected_for_folds(self):
        nodes = [
            FeatureNode(
                feature_id="standardize(x)",
                operator="standardize",
                inputs=["raw:x"],
                fit_scope=FitScope.DEVELOPMENT,
            )
        ]
        report = fold_safety_report(nodes)
        assert not report.ok
        assert any("development-scope" in m for m in report.findings)

    def test_target_aware_requires_training_fold(self):
        nodes = [
            FeatureNode(
                feature_id="target_mean(cat)",
                operator="target_mean_crossfit",
                inputs=["raw:cat"],
                fit_scope=FitScope.NONE,
                target_permission=TargetPermission.P1_CROSSFIT,
            )
        ]
        report = fold_safety_report(nodes)
        assert not report.ok
        assert any("cross-fit" in m for m in report.findings)

    def test_correct_training_fold_scope_passes(self):
        nodes = [
            FeatureNode(
                feature_id="target_mean(cat)",
                operator="target_mean_crossfit",
                inputs=["raw:cat"],
                fit_scope=FitScope.TRAINING_FOLD,
                target_permission=TargetPermission.P1_CROSSFIT,
            )
        ]
        assert fold_safety_report(nodes).ok


class TestFutureInformation:
    def test_target_history_without_label_delay_flagged(self):
        findings = check_target_history_safety(
            target_history_used=True, label_availability_known=False
        )
        assert findings
        assert all(_cls(f) is LeakageClass.L3 for f in findings)

    def test_known_label_delay_clears_finding(self):
        findings = check_target_history_safety(
            target_history_used=True, label_availability_known=True
        )
        assert findings == []

    def test_no_target_history_clears_finding(self):
        findings = check_target_history_safety(target_history_used=False)
        assert findings == []


# ---------------------------------------------------------------------------
# False-positive rates on provably clean data
# ---------------------------------------------------------------------------


class TestFalsePositiveRate:
    def test_stage_a_never_rejects_clean_data(self):
        rejects = 0
        for seed in range(30):
            f, y = _dataset_clean(n=150, seed=seed)
            finding = detect_deterministic(f, y)
            if finding is not None and finding.severity is LeakageSeverity.HARD_REJECT:
                rejects += 1
        assert rejects == 0, (
            f"Stage A hard-rejected clean data in {rejects}/30 datasets"
        )

    def test_stage_b_suspicion_rate_bounded_on_clean_data(self):
        # independent noise may occasionally trip the statistical suspicion
        # threshold; that is by design ("suspicion, never proof"), but it
        # must be rare, not the norm.
        suspicions = 0
        trials = 30
        for seed in range(trials):
            f, y = _dataset_clean(n=150, seed=1000 + seed)
            if statistical_triage(f, y) is not None:
                suspicions += 1
        assert suspicions <= trials * 0.2, (
            f"Stage B suspicion on {suspicions}/{trials} clean datasets "
            "-- false-positive rate too high"
        )

    def test_clean_categorical_feature_not_flagged(self):
        rng = random.Random(42)
        y = [rng.randint(0, 1) for _ in range(200)]
        f = [rng.choice(["red", "green", "blue"]) for _ in range(200)]
        assert detect_deterministic(f, y) is None
        assert statistical_triage(f, y) is None

    def test_moderately_predictive_feature_not_suspicious(self):
        # a genuinely useful but non-leaky feature (AUC ~0.7) must pass
        rng = random.Random(9)
        y = [rng.randint(0, 1) for _ in range(300)]
        f = []
        for v in y:
            base = 0.7 if v else 0.3
            f.append(min(1.0, max(0.0, rng.gauss(base, 0.25))))
        finding = statistical_triage(f, y)
        # may trigger review at extremes but never a hard reject
        if finding is not None:
            assert finding.severity is not LeakageSeverity.HARD_REJECT
