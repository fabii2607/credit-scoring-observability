"""Oráculos simples e casos adversos para a estatística de drift."""

import json

import numpy as np
import pandas as pd
import pytest

from credit_scoring_observability.parameters import DriftParams
from credit_scoring_observability.stats import (
    compare_numeric,
    ks_two_sample,
    psi_quantile,
    psi_status,
)


def test_identical_distribution_and_no_input_mutation():
    reference = pd.Series([0, 1, 2, 3, 4, 5, 6, 7])
    before = reference.copy()
    result = compare_numeric(reference, reference)
    assert result.psi == pytest.approx(0)
    assert result.ks_d == 0
    assert result.ks_p_value == 1
    assert result.status == "OK"
    assert not result.critical
    pd.testing.assert_series_equal(reference, before)
    json.loads(result.to_json())


def test_psi_matches_hand_calculation():
    # p=(1/2,1/2), q=(3/4,1/4), PSI = (1/4)*ln(3).
    assert psi_quantile([0, 1, 2, 3], [0, 0, 0, 3], bins=2) == pytest.approx(
        np.log(3) / 4
    )


def test_ks_matches_known_ecdf_distance():
    result = ks_two_sample([0, 1, 2, 3], [2, 3, 4, 5])
    assert result.ks_d == 0.5
    assert 0 <= result.p_value <= 1


def test_shift_detected_without_changing_reference_bins():
    ref = np.arange(1000)
    cur = ref + 2000
    result = compare_numeric(ref, cur)
    assert result.psi > 0.25
    assert result.ks_d == 1
    assert result.critical
    assert result.effective_bins == 10
    # Se recalculássemos quantis na produção, o deslocamento seria escondido.
    assert psi_quantile(ref, cur) > 1


@pytest.mark.parametrize("current", [[-100, -99, -98, -97], [100, 101, 102, 103]])
def test_tails_outside_reference_are_counted(current):
    assert psi_quantile([0, 1, 2, 3], current, bins=2) > 1


def test_constant_reference_detects_both_tails_and_mixture():
    ref = [5] * 100
    assert psi_quantile(ref, ref) == 0
    for cur in ([4] * 100, [6] * 100, [4, 6] * 50):
        result = compare_numeric(ref, cur)
        assert result.psi > 0.25
        assert result.ks_d >= 0.5
        assert result.effective_bins == 3
        assert result.critical


def test_repeated_quantiles_do_not_collapse_binary_drift():
    ref = [0] * 90 + [1] * 10
    result = compare_numeric(ref, [1] * 100)
    assert result.psi > 0.25
    assert result.ks_d == 0.9
    assert result.effective_bins < 10


def test_missingness_changes_even_if_observed_distribution_is_identical():
    result = compare_numeric([1, 2, 3, 4], [1, 2, 3, 4, None, None, None, None])
    assert result.psi == 0
    assert result.ks_d == 0
    assert result.missing_rate_delta == 0.5
    assert result.missing_drift
    assert result.current_rows == 8
    assert result.current_valid_rows == 4


def test_reduction_of_missingness_is_detected_too():
    result = compare_numeric([1, 2, None, None], [1, 2])
    assert result.missing_rate_delta == -0.5
    assert result.missing_drift


@pytest.mark.parametrize(
    "ref,cur",
    [([], [1, 2]), ([1], []), ([None, None], [1, 2]), ([1, 2], [np.nan, pd.NA])],
)
def test_unavailable_is_not_zero(ref, cur):
    result = compare_numeric(ref, cur)
    assert result.psi is None
    assert result.ks_d is None
    assert result.status == "UNAVAILABLE"
    assert not result.critical
    assert result.unavailable_reason
    assert psi_quantile(ref, cur) is None
    assert "null" in result.to_json()


def test_all_missing_current_still_reports_missingness_drift():
    result = compare_numeric([1, 2], [None, None])
    assert result.missing_drift
    assert result.current_missing_rate == 1


@pytest.mark.parametrize(
    "bad", [[np.inf], [-np.inf], ["1", "2"], [True, False], [1 + 2j]]
)
def test_invalid_values_fail_explicitly(bad):
    with pytest.raises(ValueError):
        compare_numeric([1, 2], bad)
    with pytest.raises(ValueError):
        psi_quantile(bad, [1, 2])


def test_pandas_nullable_numbers_and_small_samples():
    values = pd.Series([1, pd.NA, 2], dtype="Float64")
    assert compare_numeric(values, values).psi == 0
    result = ks_two_sample([1], [2])
    assert result.ks_d == 1
    assert result.p_value == 1


@pytest.mark.parametrize(
    "value,status",
    [
        (None, "UNAVAILABLE"),
        (0, "OK"),
        (0.099, "OK"),
        (0.10, "WARN"),
        (0.25, "WARN"),
        (0.25001, "ALERT"),
    ],
)
def test_status_boundaries(value, status):
    assert psi_status(value) == status


def test_ks_threshold_is_strict_and_configurable():
    ref, cur = [0, 1, 2, 3], [2, 3, 4, 5]
    result = compare_numeric(ref, cur, DriftParams(ks_d=0.5))
    assert result.ks_d == 0.5
    assert not result.ks_confirmed
    assert not result.critical
    assert compare_numeric(ref, cur, DriftParams(ks_d=0.49)).ks_confirmed


@pytest.mark.parametrize("value", [-0.1, np.inf, np.nan])
def test_invalid_psi_is_not_classified(value):
    with pytest.raises(ValueError):
        psi_status(value)
