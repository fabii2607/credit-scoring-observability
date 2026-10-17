from dataclasses import replace

import numpy as np
import pytest

from credit_scoring_observability.gate import evaluate_gate
from credit_scoring_observability.parameters import PerformanceParams
from credit_scoring_observability.performance import auc_difference_interval
from credit_scoring_observability.results import DriftResult, PerformanceResult


def performance(**changes):
    base = PerformanceResult(
        rows=10000,
        labeled_rows=10000,
        control_rows=30000,
        control_labeled_rows=30000,
        roc_auc=0.70,
        reference_roc_auc=0.70,
        auc_drop=0,
        auc_drop_ci_low=-0.02,
        auc_drop_ci_high=0.02,
    )
    return replace(base, **changes)


@pytest.mark.parametrize(
    "changes,drifted,status,action",
    [
        ({}, [], "OK", "observe"),
        ({}, ["dti"], "WARN", "reinforced_monitoring"),
        (
            {"auc_drop": 0.10, "auc_drop_ci_low": 0.08, "auc_drop_ci_high": 0.12},
            [],
            "ALERT",
            "retrain",
        ),
        (
            {"auc_drop": 0.06, "auc_drop_ci_low": 0.03, "auc_drop_ci_high": 0.09},
            [],
            "ALERT",
            "investigate",
        ),
        ({"labeled_rows": 10}, [], "WARN", "investigate"),
        ({"control_labeled_rows": 10}, [], "WARN", "investigate"),
        ({"auc_drop_ci_high": 0.07}, [], "WARN", "investigate"),
        ({"roc_auc": None}, [], "WARN", "investigate"),
    ],
)
def test_gate_decisions(changes, drifted, status, action):
    drift = DriftResult("month", analyzed_features=23, drifted_features=drifted)
    result = evaluate_gate(drift, performance(**changes))
    assert (result.status, result.recommended_action) == (status, action)
    assert result.retrain_recommended == (action == "retrain")
    assert result.to_dict()["reasons"]


def test_disabled_bootstrap_never_recommends_retrain():
    perf = performance(auc_drop=0.1, auc_drop_ci_low=0.08, auc_drop_ci_high=0.12)
    result = evaluate_gate(
        DriftResult("month", analyzed_features=23),
        perf,
        performance_params=PerformanceParams(bootstrap_samples=0),
    )
    assert result.status == "ALERT"
    assert not result.retrain_recommended


def test_paired_interval_identity_and_reversal_are_deterministic():
    y = np.tile([0, 1], 100)
    scores = 0.1 + 0.8 * y
    params = PerformanceParams(bootstrap_samples=30)
    assert auc_difference_interval(y, scores, y, scores, params, paired=True) == (
        0.0,
        0.0,
    )
    first = auc_difference_interval(y, scores, y, 1 - scores, params, paired=True)
    assert first == (1.0, 1.0)
    assert first == auc_difference_interval(
        y, scores, y, 1 - scores, params, paired=True
    )
    assert auc_difference_interval(
        y, scores, y, scores, PerformanceParams(bootstrap_samples=0)
    ) == (None, None)
