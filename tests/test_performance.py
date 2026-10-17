import numpy as np
import pandas as pd
import pytest

from credit_scoring_observability.parameters import PerformanceParams
from credit_scoring_observability.performance import evaluate_performance

FAST = PerformanceParams(bootstrap_samples=0)


def frame(y=(0, 0, 1, 1), scores=(0.1, 0.2, 0.8, 0.9)):
    return pd.DataFrame({"target": y, "score": scores})


def test_known_perfect_and_inverted_ranking():
    control = frame()
    result = evaluate_performance(
        control, frame(scores=(0.9, 0.8, 0.2, 0.1)), 0.5, FAST
    )
    assert result.reference_roc_auc == 1
    assert result.roc_auc == 0
    assert result.auc_drop == 1
    assert result.performance_drop_detected
    assert result.labeled_rows == 4
    result.to_json()


def test_threshold_affects_decision_metrics_but_not_auc():
    control = frame()
    low = evaluate_performance(control, control, 0.15, FAST)
    high = evaluate_performance(control, control, 0.85, FAST)
    assert low.roc_auc == high.roc_auc == 1
    assert low.approval_rate == 0.25
    assert high.approval_rate == 0.75
    assert low.recall == 1
    assert high.recall == 0.5


def test_missing_target_preserves_operational_metrics_without_drop_decision():
    result = evaluate_performance(frame(), frame().drop(columns="target"), 0.5, FAST)
    assert result.roc_auc is None
    assert result.auc_drop is None
    assert result.performance_drop_detected is None
    assert result.approval_rate == 0.5
    assert result.unavailable_reason == "missing_target"
    assert result.labeled_rows == 0


def test_single_class_preserves_brier_but_auc_is_unavailable():
    result = evaluate_performance(frame(), frame(y=(0, 0, 0, 0)), 0.5, FAST)
    assert result.roc_auc is None
    assert result.pr_auc is None
    assert result.brier is not None
    assert result.unavailable_reason == "single_class_target"
    assert result.performance_drop_detected is None


def test_unusable_control_does_not_invent_a_comparison():
    result = evaluate_performance(frame().drop(columns="target"), frame(), 0.5, FAST)
    assert result.roc_auc == 1
    assert result.reference_roc_auc is None
    assert result.comparison_unavailable_reason == "control_missing_target"


def test_partial_labels_are_counted_and_empty_batch_is_safe():
    result = evaluate_performance(frame(), frame(y=(0, np.nan, 1, np.nan)), 0.5, FAST)
    assert result.labeled_rows == 2
    assert result.rows == 4
    assert result.roc_auc == 1
    empty = evaluate_performance(frame(), frame().iloc[:0], 0.5, FAST)
    assert empty.approval_rate is None
    assert empty.score_p50 is None
    empty.to_json()


def test_bootstrap_is_reproducible_and_does_not_mutate_inputs():
    rng = np.random.default_rng(1)
    y = np.tile([0, 1], 100)
    scores = np.clip(y * 0.3 + rng.uniform(0, 0.7, len(y)), 0, 1)
    current = frame(y, scores)
    before = current.copy()
    params = PerformanceParams(bootstrap_samples=30)
    first = evaluate_performance(current, current, 0.5, params)
    second = evaluate_performance(current, current, 0.5, params)
    assert first.auc_ci_low == second.auc_ci_low
    assert 0 <= first.auc_ci_low <= first.auc_ci_high <= 1
    pd.testing.assert_frame_equal(current, before)


@pytest.mark.parametrize(
    "scores",
    [(0.1, 0.2, 0.8, float("inf")), (0.1, 0.2, 0.8, np.nan), (-0.1, 0.2, 0.8, 0.9)],
)
def test_invalid_scores_are_rejected(scores):
    with pytest.raises(ValueError):
        evaluate_performance(frame(), frame(scores=scores), 0.5, FAST)


def test_invalid_labels_are_rejected():
    with pytest.raises(ValueError):
        evaluate_performance(frame(), frame(y=(0, 2, 1, 1)), 0.5, FAST)
