import json

import pytest

from credit_scoring_observability.results import (
    DriftResult,
    FeatureDriftResult,
    PerformanceResult,
)


def test_nested_drift_contract_is_strict_json():
    result = DriftResult(
        batch_id="2026-01",
        features=[
            FeatureDriftResult(
                feature="annual_inc",
                psi=0.3,
                ks_d=0.2,
                status="ALERT",
                ks_confirmed=True,
            )
        ],
        analyzed_features=1,
        drift_share=1.0,
        drifted_features=["annual_inc"],
    )
    data = json.loads(result.to_json())
    assert data["features"][0]["psi"] == 0.3
    assert data["label_z"] is None


def test_performance_contract_can_express_absence_of_labels():
    result = PerformanceResult(approval_rate=0.6, unavailable_reason="missing_target")
    data = json.loads(result.to_json())
    assert data["roc_auc"] is None
    assert data["auc_drop"] is None
    assert data["labeled_rows"] == 0
    assert data["unavailable_reason"] == "missing_target"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_values_are_rejected_even_when_nested(value):
    result = DriftResult(
        batch_id="2026-01", features=[FeatureDriftResult(feature="dti", psi=value)]
    )
    with pytest.raises(ValueError):
        result.to_dict()
    with pytest.raises(ValueError):
        PerformanceResult(roc_auc=value).to_json()
