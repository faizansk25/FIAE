import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fiae.contracts import (
    FeatureNode,
    FitScope,
    LeakageClass,
    LeakageSeverity,
    TargetPermission,
)
from fiae.problem.leakage import (
    CrossFitTargetEncoder,
    check_target_history_safety,
    confirm_unavailable,
    detect_deterministic,
    fold_safety_report,
    rank_auc,
    review_availability,
    statistical_triage,
)


# --- Stage A: deterministic -------------------------------------------------
def test_exact_target_copy_hard_reject():
    f = [str(i % 2) for i in range(50)]
    t = [i % 2 for i in range(50)]
    finding = detect_deterministic(f, t, feature_name="copy")
    assert finding is not None
    assert finding.severity is LeakageSeverity.HARD_REJECT
    assert finding.type is LeakageClass.L5


def test_inverted_target_copy_hard_reject():
    f = ["yes" if i % 2 == 0 else "no" for i in range(50)]
    t = [1 if v == "no" else 0 for v in f]
    finding = detect_deterministic(f, t, feature_name="inverted")
    assert finding is not None and finding.severity is LeakageSeverity.HARD_REJECT


def test_deterministic_bijection_hard_reject():
    f = [f"row_{i}" for i in range(40)]
    t = [i % 2 for i in range(40)]
    finding = detect_deterministic(f, t, feature_name="row_number")
    assert finding is not None
    assert finding.action == "reject"


def test_non_bijective_category_is_not_rejected():
    f = ["cat_a"] * 40 + ["cat_b"] * 40
    t = [0] * 40 + [1] * 40
    finding = detect_deterministic(f, t, feature_name="segment")
    assert finding is None


def test_target_embedded_in_string_hard_reject():
    f = ["item_A", "item_B"] * 25
    t = ["A", "B"] * 25
    finding = detect_deterministic(f, t, feature_name="code")
    assert finding is not None and finding.severity is LeakageSeverity.HARD_REJECT


# --- Stage B: statistical suspicion -----------------------------------------
def test_near_perfect_auc_suspicion():
    f = [float(i % 10) for i in range(100)]
    t = [1 if v >= 5 else 0 for v in f]
    finding = statistical_triage(f, t, feature_name="score")
    assert finding is not None
    assert finding.severity is LeakageSeverity.REVIEW_REQUIRED
    assert finding.action == "review"


def test_extreme_mutual_information_suspicion():
    f = ["a"] * 100 + ["b"] * 100
    t = [1] * 100 + [0] * 99 + [1]
    finding = statistical_triage(f, t, feature_name="bucket")
    assert finding is not None
    assert finding.severity is LeakageSeverity.REVIEW_REQUIRED


def test_missingness_determines_target_suspicion():
    f = [None] * 30 + [str(i) for i in range(30)]
    t = [1] * 30 + [0] * 30
    finding = statistical_triage(f, t, feature_name="reported")
    assert finding is not None
    assert finding.severity is LeakageSeverity.REVIEW_REQUIRED


def test_weak_feature_no_suspicion():
    import random

    rng = random.Random(0)
    f = [round(rng.gauss(0, 1), 3) for _ in range(500)]
    t = [1 if rng.random() < 0.5 else 0 for _ in range(500)]
    assert statistical_triage(f, t, feature_name="noise") is None


def test_rank_auc_known_value():
    scores = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]
    y = [0, 1, 0, 1, 0, 1, 0, 1]
    auc = rank_auc(scores, y)
    assert auc is not None and auc > 0.95
# --- Stage C: prediction-time availability ----------------------------------
def test_availability_after_prediction_time_review():
    findings = review_availability(
        {"revenue_now": "2025-06-02T12:00:00"},
        "2025-06-01T00:00:00",
    )
    assert findings
    assert findings[0].severity is LeakageSeverity.REVIEW_REQUIRED


def test_availability_before_prediction_time_ok():
    findings = review_availability(
        {"revenue_now": "2025-05-01T00:00:00"},
        "2025-06-01T00:00:00",
    )
    assert findings == []


def test_confirm_unavailable_is_hard_reject():
    findings = confirm_unavailable(["status_final"], "post-outcome field")
    assert findings[0].severity is LeakageSeverity.HARD_REJECT
    assert findings[0].type is LeakageClass.L5


def test_target_history_requires_label_availability():
    assert check_target_history_safety(False) == []
    findings = check_target_history_safety(True, label_availability_known=False)
    assert findings and findings[0].type is LeakageClass.L3
    assert check_target_history_safety(True, label_availability_known=True) == []


# --- Stage D: fold safety ----------------------------------------------------
def test_fold_safety_flags_unsafe_scopes():
    ok_node = FeatureNode(
        feature_id="f1",
        operator="standardize",
        inputs=["x"],
        fit_scope=FitScope.TRAINING_FOLD,
    )
    dev_node = FeatureNode(
        feature_id="f2",
        operator="standardize",
        inputs=["x"],
        fit_scope=FitScope.DEVELOPMENT,
    )
    crossfit_node = FeatureNode(
        feature_id="f3",
        operator="target_mean_crossfit",
        inputs=["c", "y"],
        fit_scope=FitScope.NONE,
        target_permission=TargetPermission.P1_CROSSFIT,
    )
    assert fold_safety_report([ok_node]).ok
    assert not fold_safety_report([dev_node]).ok
    assert not fold_safety_report([crossfit_node]).ok


# --- Cross-fit target encoding ----------------------------------------------
def test_crossfit_excludes_own_target():
    encoder = CrossFitTargetEncoder(smoothing=0.0)
    cats = ["x"] * 10
    targets = [1.0] * 9 + [0.0]
    enc = encoder.fit_transform(cats, targets, n_folds=3, seed=0)
    assert abs(enc[9] - 1.0) < 1e-9  # row with target 0 encoded from target-1 rows
    assert abs(enc[9] - targets[9]) > 0.9  # own target excluded


def test_crossfit_single_occurrence_uses_prior():
    encoder = CrossFitTargetEncoder(smoothing=0.0)
    cats = ["u", "v", "w", "u", "v", "w"]
    targets = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    enc = encoder.fit_transform(cats, targets, n_folds=3, seed=1)
    assert all(0.0 <= v <= 1.0 for v in enc)
    assert set(enc) <= {0.0, 0.5, 1.0, 2 / 3, 1 / 3}


def test_crossfit_final_unknown_uses_prior():
    encoder = CrossFitTargetEncoder(smoothing=1.0)
    encoder.fit_final(["a", "a", "b"], [1.0, 1.0, 0.0])
    # known category: m-estimate smoothed mean = (2*1 + 1*(2/3)) / (2+1) = 8/9
    assert abs(encoder.transform(["a"])[0] - 8 / 9) < 1e-9
    # unknown category falls back to the global prior (2/3 here)
    assert abs(encoder.transform(["zz"])[0] - encoder.prior) < 1e-9


def test_crossfit_group_aware():
    encoder = CrossFitTargetEncoder(smoothing=0.0)
    cats = ["g1", "g2"] * 30
    targets = [0.0, 1.0] * 30
    groups = [i // 20 for i in range(60)]
    enc = encoder.fit_transform(cats, targets, n_folds=3, seed=0, group_ids=groups)
    assert all(0.0 <= v <= 1.0 for v in enc)
