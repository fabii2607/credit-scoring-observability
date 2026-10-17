import pytest

from credit_scoring_observability.comparison import compare_intervention
from credit_scoring_observability.drift import detect_drift
from credit_scoring_observability.parameters import PerformanceParams
from credit_scoring_observability.performance import evaluate_performance
from credit_scoring_observability.synthetic import make_model_batch


def compare(original, current, scenario):
    params = PerformanceParams(bootstrap_samples=20)
    return compare_intervention(
        original,
        original,
        original,
        current,
        detect_drift(original, current, "month"),
        evaluate_performance(original, current, 0.2, params),
        scenario,
        0.2,
        performance_params=params,
    )


def test_stable_pairs_have_zero_effect(scoring_model):
    original = scoring_model.score(make_model_batch(500, 81))
    result = compare(original, original.copy(), "stable")
    assert result.paired_auc_drop == 0
    assert result.paired_auc_drop_ci_low == 0
    assert result.intervention_drift_share == 0


def test_relabel_preserves_scores_and_changes_auc(scoring_model):
    original = scoring_model.score(make_model_batch(500, 81))
    altered = original.copy()
    altered["target"] = 1 - altered["target"]
    result = compare(original, altered, "concept_drift")
    assert result.intervention_drift_share == 0
    assert result.intervention_score_psi == 0
    assert abs(result.paired_auc_drop) > 0
    assert result.paired_rows == 500


def test_pair_order_and_scenario_integrity(scoring_model):
    original = scoring_model.score(make_model_batch(500, 81))
    with pytest.raises(ValueError, match="ordem"):
        compare(original, original.iloc[::-1], "stable")
    altered = original.copy()
    altered["dti"] += 1
    with pytest.raises(ValueError, match="colunas"):
        compare(original, altered, "concept_drift")
