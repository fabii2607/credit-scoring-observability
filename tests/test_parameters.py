import pytest
import yaml
from pydantic import ValidationError

from credit_scoring_observability.config import PARAMS_PATH
from credit_scoring_observability.parameters import get_params, load_params


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
