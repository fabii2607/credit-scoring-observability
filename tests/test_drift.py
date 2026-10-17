import json

import numpy as np
import pandas as pd
import pytest

from credit_scoring_observability.data_contract import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
)
from credit_scoring_observability.drift import categorical_js, detect_drift
from credit_scoring_observability.drift_report import (
    register_methods,
    write_drift_report,
)
from credit_scoring_observability.parameters import DriftParams
from credit_scoring_observability.simulation import apply_data_drift
from credit_scoring_observability.stats import compare_numeric
from credit_scoring_observability.synthetic import make_model_batch


@pytest.fixture
def scored():
    frame = make_model_batch(500, 30)
    frame["score"] = np.linspace(0.01, 0.9, len(frame))
    frame["prediction"] = (frame.score >= 0.2).astype("int8")
    return frame


def test_identical_data_has_no_drift_and_explicit_feature_denominator(scored):
    result = detect_drift(scored, scored, "2026-01")
    assert result.drift_share == 0
    assert result.analyzed_features == len(MODEL_FEATURES)
    assert result.score_psi == 0
    assert result.label_z == 0
    assert {f.feature for f in result.features} == set(MODEL_FEATURES)
    for feature in result.features:
        if feature.feature in CATEGORICAL_FEATURES:
            assert feature.ks_d is None
            assert feature.psi is None
            assert feature.categorical_distance == 0


def test_detects_numeric_and_categorical_changes_using_same_statistics(scored):
    changed = apply_data_drift(scored, 1)
    changed["home_ownership"] = "NEW_CATEGORY"
    result = detect_drift(scored, changed, "2026-03")
    income = next(f for f in result.features if f.feature == "annual_inc")
    assert income.psi == compare_numeric(scored.annual_inc, changed.annual_inc).psi
    assert len(result.drifted_features) >= 3
    home = next(f for f in result.features if f.feature == "home_ownership")
    assert home.drifted
    assert home.categorical_distance == pytest.approx(np.sqrt(np.log(2)))


def test_label_changes_are_not_feature_drift(scored):
    changed = scored.copy()
    changed["target"] = 1
    result = detect_drift(scored, changed, "2026-05")
    assert result.drift_share == 0
    assert result.score_psi == 0
    assert result.label_z > 3
    assert result.default_rate == 1


def test_all_missing_column_is_unavailable_not_zero_and_denominator_excludes_it(scored):
    changed = scored.copy()
    changed["dti"] = np.nan
    result = detect_drift(scored, changed, "2026-02")
    assert result.unavailable_features == ["dti"]
    assert result.analyzed_features == len(MODEL_FEATURES) - 1
    item = next(f for f in result.features if f.feature == "dti")
    assert item.psi is None
    assert item.missing_drift
    json.loads(result.to_json())


def test_absent_target_and_empty_data_do_not_invent_label_drift(scored):
    result = detect_drift(
        scored.drop(columns="target"), scored.drop(columns="target"), "2026-01"
    )
    assert result.label_z is None
    assert result.default_rate is None
    empty = detect_drift(scored, scored.iloc[:0], "empty")
    assert empty.drift_share is None
    assert empty.analyzed_features == 0


def test_js_aligns_new_categories_and_ignores_missingness():
    assert categorical_js(
        pd.Series(["A", "A"]), pd.Series(["B", "B"])
    ) == pytest.approx(np.sqrt(np.log(2)))
    assert categorical_js(pd.Series(["A", "B"]), pd.Series(["B", "A", None])) == 0
    assert categorical_js(pd.Series([None]), pd.Series(["B"])) is None


def test_evidently_custom_stattest_matches_numeric_gate(scored):
    from evidently.legacy.calculations.stattests.registry import get_registered_stattest
    from evidently.legacy.core import ColumnType

    params = DriftParams(bins=5, ks_d=0.9)
    numeric, _ = register_methods(params)
    changed = apply_data_drift(scored, 1)
    expected = compare_numeric(scored.annual_inc, changed.annual_inc, params)
    actual = get_registered_stattest(numeric, ColumnType.Numerical)(
        scored.annual_inc, changed.annual_inc, ColumnType.Numerical, params.psi_alert
    )
    assert actual.drift_score == expected.psi
    assert actual.drifted == expected.critical


def test_html_and_json_use_only_allowed_columns_and_strict_aggregate_data(
    scored, tmp_path
):
    changed = apply_data_drift(scored, 1)
    scored["customer_id"] = [f"PRIVATE_REFERENCE_{i}" for i in range(len(scored))]
    changed["customer_id"] = [f"PRIVATE_CURRENT_{i}" for i in range(len(changed))]
    scored["region"] = "PRIVATE_REGION"
    result = detect_drift(scored, changed, "2026-03")
    paths = write_drift_report(scored, changed, result, tmp_path)
    html = (tmp_path / "2026-03.html").read_text()
    summary = json.loads((tmp_path / "2026-03.json").read_text())
    assert "Evidently" in html
    assert "credit-summary" in html
    assert "PRIVATE_" not in html
    assert "PRIVATE_" not in json.dumps(summary)
    assert "annual_inc" in html
    assert summary["features"][0]["feature"] == MODEL_FEATURES[0]
    assert paths == result.report_paths


def test_report_with_no_observations_is_explicitly_unavailable(scored, tmp_path):
    result = detect_drift(scored, scored.iloc[:0], "empty")
    write_drift_report(scored, scored.iloc[:0], result, tmp_path)
    assert "Sem distribuições observadas" in (tmp_path / "empty.html").read_text()


def test_report_rejects_path_traversal(scored, tmp_path):
    result = detect_drift(scored, scored, "../private")
    with pytest.raises(ValueError, match="Identificador"):
        write_drift_report(scored, scored, result, tmp_path)
