import json

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from credit_scoring_observability.config import (
    ID_COLUMN,
    PREDICTION_COLUMN,
    SCORE_COLUMN,
    SPLIT_COLUMN,
    TARGET,
)
from credit_scoring_observability.data_contract import MODEL_FEATURES
from credit_scoring_observability.evaluate import (
    approval_rate,
    classification_metrics,
    ks_statistic,
    select_threshold,
)
from credit_scoring_observability.parameters import ThresholdGrid, TrainParams
from credit_scoring_observability.registry import load_model, save_model
from credit_scoring_observability.synthetic import make_model_batch
from credit_scoring_observability.train import train

PARAMS = TrainParams(
    train_sample_rows=2_500,
    validation_fraction=0.2,
    min_precision=0.30,
    threshold_grid=ThresholdGrid(start=0.10, stop=0.90, step=0.025),
    class_weights=["standard", "balanced"],
    max_iter=500,
    tol=1e-3,
)


@pytest.fixture(scope="module")
def trained():
    reference = make_model_batch(n=3_000, seed=1)
    reference[SPLIT_COLUMN] = "train"
    pool = make_model_batch(n=1_000, seed=2)
    pipeline, metadata = train(reference, pool, PARAMS)
    return pipeline, metadata, pool


@pytest.fixture
def saved_model(trained, tmp_path):
    pipeline, metadata, _ = trained
    paths = {
        "model_path": tmp_path / "model.joblib",
        "metadata_path": tmp_path / "model.json",
    }
    save_model(pipeline, metadata, **paths)
    return paths


def test_metrics_on_perfect_and_random_scores():
    y = np.array([0, 0, 1, 1])
    perfect = classification_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]), 0.5)
    assert perfect["roc_auc"] == 1.0
    assert perfect["ks"] == 1.0
    assert perfect["recall"] == 1.0
    assert perfect["approval_rate"] == 0.5
    assert ks_statistic(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.0
    assert approval_rate(np.array([0.1, 0.3]), 0.2) == 0.5


def test_select_threshold_respects_min_precision():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2_000)
    proba = np.clip(y * 0.3 + rng.uniform(0, 0.7, 2_000), 0, 1)
    selection = select_threshold(
        y, {"a": proba, "b": rng.uniform(0, 1, 2_000)}, np.arange(0.1, 0.9, 0.05), 0.6
    )
    assert selection.config == "a"
    assert selection.validation_metrics["precision"] >= 0.6


def test_select_threshold_fails_without_eligible_threshold():
    y = np.array([0, 1] * 50)
    with pytest.raises(ValueError, match="precision"):
        select_threshold(y, {"a": np.full(100, 0.5)}, np.array([0.4, 0.6]), 0.9)


def test_train_produces_metadata_like_notebook(trained):
    _, metadata, pool = trained
    for key in (
        "version",
        "decision_threshold",
        "feature_columns",
        "metrics_on_test",
        "selected_model",
    ):
        assert key in metadata
    assert sorted(metadata["feature_columns"]) == sorted(MODEL_FEATURES)
    assert metadata["test_rows"] == len(pool)
    assert metadata["metrics_on_test"]["roc_auc"] > 0.6  # sinal do gerador sintético
    assert 0 < metadata["decision_threshold"] < 1
    json.dumps(metadata)  # serializável


def test_load_model_scores_batch_with_versioned_threshold(saved_model, trained):
    _, metadata, pool = trained
    model = load_model(**saved_model)
    assert model.threshold == metadata["decision_threshold"]
    scored = model.score(pool)
    assert {SCORE_COLUMN, PREDICTION_COLUMN, ID_COLUMN, TARGET} <= set(scored.columns)
    assert scored[SCORE_COLUMN].between(0, 1).all()
    expected = (scored[SCORE_COLUMN] >= model.threshold).astype(np.int8)
    assert scored[PREDICTION_COLUMN].equals(expected)
    np.testing.assert_allclose(
        model.predict_proba(pool), scored[SCORE_COLUMN].to_numpy()
    )


def test_load_model_accepts_any_column_order(saved_model, trained):
    _, _, pool = trained
    model = load_model(**saved_model)
    shuffled = pool.loc[:, list(reversed(pool.columns))]
    np.testing.assert_allclose(model.predict_proba(shuffled), model.predict_proba(pool))


def test_invalid_batch_is_not_scored(saved_model, trained):
    _, _, pool = trained
    model = load_model(**saved_model)
    bad = pool.head(5).copy()
    bad.loc[bad.index[0], "term"] = 48
    with pytest.raises(SchemaErrors):
        model.score(bad)


def test_missing_model_has_actionable_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="make train"):
        load_model(
            model_path=tmp_path / "none.joblib", metadata_path=tmp_path / "none.json"
        )


def test_save_model_requires_threshold(tmp_path):
    with pytest.raises(ValueError, match="decision_threshold"):
        save_model(object(), {"version": "x", "feature_columns": []}, tmp_path / "m")


def test_pool_columns_are_untouched_by_scoring(saved_model, trained):
    _, _, pool = trained
    model = load_model(**saved_model)
    before = pool.copy()
    model.score(pool)
    pd.testing.assert_frame_equal(pool, before)
