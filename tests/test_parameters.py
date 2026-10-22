import pytest
import yaml
from pydantic import ValidationError

from credit_scoring_observability.config import PARAMS_PATH
from credit_scoring_observability.parameters import (
    DriftParams,
    PerformanceParams,
    SimulationParams,
    get_params,
    load_params,
)


def test_default_params_load():
    params = get_params()
    assert params.split.test_size == 0.2
    assert params.train.min_precision == 0.30
    grid = params.train.threshold_grid.values()
    assert grid[0] == 0.10
    assert grid[-1] == 0.90
    assert 0.2 in grid


def test_unknown_key_fails(tmp_path):
    data = yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))
    data["split"]["test_sise"] = 0.3  # typo
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_params(path)


def test_stage_two_defaults_are_explicit_and_valid():
    params = load_params()
    assert params.simulation.batch_rows == 10_000
    assert params.simulation.months[0] == "2026-01"
    assert params.simulation.months[-1] == "2026-07"
    assert params.drift.psi_alert == 0.25
    assert params.performance.auc_drop == 0.05


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_rows": -1},
        {"batch_rows": 10.5},
        {"seed": -1},
        {"data_intensities": [0.6, 1.1]},
        {"concept_intensities": [0.5, 0.2, 0.9]},
        {"data_intensities": [float("nan"), 1]},
        {"months": ["2026-01"] * 7},
        {
            "months": [
                "2026-01",
                "2026-02",
                "2026-03",
                "2026-04",
                "2026-05",
                "2026-06",
                "2026-13",
            ]
        },
    ],
)
def test_invalid_simulation_configuration(kwargs):
    with pytest.raises(ValidationError):
        SimulationParams(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bins": 1},
        {"epsilon": 0},
        {"psi_warning": 0.3, "psi_alert": 0.2},
        {"ks_d": 1.1},
        {"psi_alert": float("inf")},
        {"psi_warnng": 0.1},
    ],
)
def test_invalid_drift_configuration(kwargs):
    with pytest.raises(ValidationError):
        DriftParams(**kwargs)


@pytest.mark.parametrize(
    "kwargs", [{"auc_drop": 0}, {"bootstrap_samples": -1}, {"auc_drop": float("nan")}]
)
def test_invalid_performance_configuration(kwargs):
    with pytest.raises(ValidationError):
        PerformanceParams(**kwargs)


def test_batch_volume_must_match_contract(tmp_path):
    data = yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))
    data["contract"]["min_rows"] = 20_000
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValidationError, match=r"contract\.min_rows"):
        load_params(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"demo_rows": 100},
        {"fico_drop_points": 12},
        {"concept_low_fico": 720, "concept_high_fico": 700},
        {"concept_default_probability": 1.1},
        {"concept_cure_probability": float("nan")},
        {"utilization_increase_pp": -1},
        {"dti_increase": float("inf")},
    ],
)
def test_invalid_intervention_configuration(kwargs):
    with pytest.raises(ValidationError):
        SimulationParams(**kwargs)
