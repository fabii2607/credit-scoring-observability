import json
from unittest.mock import Mock

import pandas as pd
import pytest

from credit_scoring_observability.analyze import analyze_batch, run
from credit_scoring_observability.parameters import PerformanceParams, SimulationParams
from credit_scoring_observability.simulation import generate
from credit_scoring_observability.synthetic import make_model_batch


def test_invalid_row_blocks_whole_batch_before_inference_and_clears_stale_outputs(
    tmp_path,
):
    frame = make_model_batch(500, 10)
    frame.loc[0, "annual_inc"] = None
    path = tmp_path / "bad.parquet"
    frame.to_parquet(path, index=False)
    interim = tmp_path / "interim"
    work = interim / "bad"
    work.mkdir(parents=True)
    (work / "scored.parquet").write_text("stale")
    model = Mock()
    result = analyze_batch(
        path,
        model,
        pd.DataFrame(),
        pd.DataFrame(),
        interim,
        tmp_path / "reports",
        tmp_path / "quarantine",
    )
    assert result["status"] == "blocked"
    model.score.assert_not_called()
    assert not (work / "scored.parquet").exists()
    assert not path.exists()
    assert (tmp_path / "quarantine" / "bad.parquet").exists()
    report = json.loads((tmp_path / "reports" / "contracts" / "bad.json").read_text())
    assert not report["passed"]


def test_missing_batch_is_failed_not_success(tmp_path):
    model = Mock()
    result = analyze_batch(
        tmp_path / "absent.parquet",
        model,
        pd.DataFrame(),
        pd.DataFrame(),
        tmp_path / "interim",
        tmp_path / "reports",
        tmp_path / "quarantine",
    )
    assert result["status"] == "failed"
    model.score.assert_not_called()


@pytest.mark.integration
def test_months_are_scored_after_blocked_demo_and_reports_have_no_ids(
    tmp_path, scoring_model, monkeypatch
):
    pool_path, ref_path = tmp_path / "pool.parquet", tmp_path / "ref.parquet"
    pool = make_model_batch(6500, 55)
    reference = make_model_batch(500, 56)
    pool.to_parquet(pool_path, index=False)
    reference.assign(split="train").to_parquet(ref_path, index=False)
    production = tmp_path / "production"
    generate(
        pool_path,
        ref_path,
        production,
        SimulationParams(
            batch_rows=500, control_rows=500, calibration_rows=500, demo_rows=500
        ),
    )
    score = Mock(wraps=scoring_model.score)
    monkeypatch.setattr(scoring_model, "score", score)
    reports = tmp_path / "reports"
    outcomes = run(
        production,
        ref_path,
        tmp_path / "interim",
        reports,
        tmp_path / "quarantine",
        batch_ids=["corrupted", "2026-01", "2026-05"],
        model=scoring_model,
        performance_params=PerformanceParams(bootstrap_samples=0),
    )
    assert [o["status"] for o in outcomes] == ["blocked", "success", "success"]
    assert (
        score.call_count == 6
    )  # referência, controle, dois meses e seus originais; corrupted não pontuado.
    for month in ["2026-01", "2026-05"]:
        scored = pd.read_parquet(tmp_path / "interim" / month / "scored.parquet")
        assert {"score", "prediction"} <= set(scored)
        assert (reports / "drift" / f"{month}.html").exists()
        assert (reports / "performance" / f"{month}.json").exists()
        gate = json.loads((reports / "gate" / f"{month}.json").read_text())
        drift = json.loads((reports / "drift" / f"{month}.json").read_text())
        assert drift["gate"] == {k: v for k, v in gate.items() if k != "context"}
        html = (reports / "drift" / f"{month}.html").read_text()
        assert html.index('id="credit-gate"') < html.index('id="credit-performance"')
        assert gate["recommendation"] in html
        assert not gate["retrain_recommended"]  # bootstrap desabilitado
        assert (reports / "intervention" / f"{month}.json").exists()
    assert (reports / "experiment" / "auc_effects.png").exists()
    assert (reports / "experiment" / "features.csv").exists()
    assert (reports / "performance" / "summary.csv").exists()
    for path in [*(reports / "drift").glob("*"), *(reports / "performance").glob("*")]:
        text = path.read_text()
        assert not any(identifier in text for identifier in pool.customer_id)
        assert not any(identifier in text for identifier in reference.customer_id)
