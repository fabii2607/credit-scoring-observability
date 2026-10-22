"""Simulação com dados sintéticos: isolamento, invariantes e reprodução."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from credit_scoring_observability.config import ID_COLUMN, TARGET
from credit_scoring_observability.parameters import SimulationParams
from credit_scoring_observability.simulation import (
    DATA_FEATURES,
    apply_concept_drift,
    apply_data_drift,
    build_simulation,
    concept_segments,
    file_digest,
    frame_digest,
    generate,
)
from credit_scoring_observability.synthetic import make_model_batch
from credit_scoring_observability.validate import validate_batch

PARAMS = SimulationParams(
    batch_rows=500, control_rows=500, calibration_rows=500, demo_rows=500
)


@pytest.fixture(scope="module")
def pool():
    return make_model_batch(6500, seed=300)


@pytest.fixture(scope="module")
def reference():
    return make_model_batch(1000, seed=301)


@pytest.fixture(scope="module")
def simulation(pool, reference):
    return build_simulation(pool, reference[ID_COLUMN], PARAMS)


def test_months_reserves_and_reference_are_disjoint(simulation, reference):
    paths = [b["path"] for b in simulation.manifest["batches"]]
    paths += [v["path"] for v in simulation.manifest["reservations"].values()]
    ids = pd.concat([simulation.frames[p][ID_COLUMN] for p in paths])
    assert ids.is_unique
    assert not ids.isin(reference[ID_COLUMN]).any()
    assert len(ids) == 5000
    assert simulation.manifest["audit"]["unused_rows"] == 1500
    assert len(simulation.manifest["batches"]) == 7


def test_only_intended_columns_change_and_regular_batches_pass(simulation):
    for batch in simulation.manifest["batches"]:
        current = simulation.frames[batch["path"]]
        original = simulation.frames[batch["original_path"]]
        assert len(current) == PARAMS.batch_rows
        assert validate_batch(current, batch["batch_id"]).passed
        if batch["scenario"] == "stable":
            pd.testing.assert_frame_equal(current, original)
        elif batch["scenario"] == "data_drift":
            pd.testing.assert_frame_equal(
                current.drop(columns=DATA_FEATURES),
                original.drop(columns=DATA_FEATURES),
            )
            assert current["annual_inc"].sum() < original["annual_inc"].sum()
            assert current["dti"].sum() > original["dti"].sum()
            assert current["revol_util"].sum() > original["revol_util"].sum()
            assert current["fico_avg"].sum() < original["fico_avg"].sum()
        else:
            pd.testing.assert_frame_equal(
                current.drop(columns=TARGET), original.drop(columns=TARGET)
            )
            assert batch["labels_0_to_1"] + batch["labels_1_to_0"] > 0


def test_generation_is_deterministic_and_does_not_mutate_inputs(
    pool, reference, simulation
):
    before = frame_digest(pool)
    repeated = build_simulation(pool, reference[ID_COLUMN], PARAMS)
    assert repeated.manifest == simulation.manifest
    for path in simulation.frames:
        pd.testing.assert_frame_equal(repeated.frames[path], simulation.frames[path])
    assert frame_digest(pool) == before
    different = build_simulation(
        pool, reference[ID_COLUMN], PARAMS.model_copy(update={"seed": 43})
    )
    assert frame_digest(different.frames["2026-01.parquet"]) != frame_digest(
        simulation.frames["2026-01.parquet"]
    )


def test_demo_is_corrupted_and_keeps_its_own_source(simulation):
    bad = simulation.frames["corrupted.parquet"]
    demo = simulation.frames["reserves/demo.parquet"]
    assert validate_batch(demo, "demo").passed
    result = validate_batch(bad, "corrupted")
    assert not result.passed
    assert {"duplicate_customer_id", "unexpected_column", "not_nullable"} <= {
        v.rule for v in result.blockers
    }
    assert set(bad[ID_COLUMN]) == set(demo[ID_COLUMN])
    monthly_ids = set().union(
        *(
            set(simulation.frames[b["path"]][ID_COLUMN])
            for b in simulation.manifest["batches"]
        )
    )
    assert not monthly_ids & set(bad[ID_COLUMN])


def test_data_drift_preserves_nans_fico_granularity_and_allows_utilization_over_100(
    pool,
):
    batch = pool.head(20).copy()
    batch.loc[0, "dti"] = np.nan
    batch.loc[1, "revol_util"] = np.nan
    batch.loc[2, "revol_util"] = 120
    batch.loc[3, "fico_avg"] = 302
    before = batch.copy(deep=True)
    current = apply_data_drift(batch, 0.6)
    pd.testing.assert_frame_equal(batch, before)
    assert pd.isna(current.loc[0, "dti"])
    assert pd.isna(current.loc[1, "revol_util"])
    assert current.loc[2, "revol_util"] == 135
    assert current.loc[3, "fico_avg"] == 300
    assert current.loc[4, "fico_avg"] == batch.loc[4, "fico_avg"] - 20
    assert validate_batch(current, "test", min_rows=1).passed
    pd.testing.assert_frame_equal(apply_data_drift(batch, 0), batch)


def test_concept_changes_only_eligible_labels_with_nested_intensity(pool):
    before = pool.copy(deep=True)
    low = apply_concept_drift(pool, 0.35, np.random.default_rng(12), PARAMS)
    high = apply_concept_drift(pool, 0.75, np.random.default_rng(12), PARAMS)
    new_default, cure = concept_segments(pool, PARAMS)
    changed = high[TARGET].ne(pool[TARGET]).to_numpy()
    assert not (changed & ~(new_default | cure)).any()
    assert high.loc[changed & new_default, TARGET].eq(1).all()
    assert high.loc[changed & cure, TARGET].eq(0).all()
    assert (
        high.loc[low[TARGET].ne(pool[TARGET]), TARGET]
        == low.loc[low[TARGET].ne(pool[TARGET]), TARGET]
    ).all()
    pd.testing.assert_frame_equal(high.drop(columns=TARGET), pool.drop(columns=TARGET))
    pd.testing.assert_frame_equal(pool, before)
    pd.testing.assert_frame_equal(
        apply_concept_drift(pool, 0, np.random.default_rng(12)), pool
    )


def test_relabel_degrades_synthetic_model_without_using_its_scores_to_generate(
    pool, scoring_model
):
    current = apply_concept_drift(pool, 0.75, np.random.default_rng(22), PARAMS)
    scores = scoring_model.predict_proba(pool)
    assert (
        roc_auc_score(pool[TARGET], scores) - roc_auc_score(current[TARGET], scores)
        > 0.05
    )


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_intensity_is_rejected(pool, value):
    with pytest.raises(ValueError, match="Intensidade"):
        apply_data_drift(pool, value)
    with pytest.raises(ValueError, match="Intensidade"):
        apply_concept_drift(pool, value, np.random.default_rng(1))


@pytest.mark.parametrize(
    "problem",
    [
        "overlap",
        "duplicate",
        "missing_income",
        "missing_target",
        "small",
        "missing_column",
    ],
)
def test_invalid_pool_is_rejected_without_silent_exclusion(pool, reference, problem):
    bad = pool.copy()
    if problem == "overlap":
        bad.loc[0, ID_COLUMN] = reference.iloc[0][ID_COLUMN]
    elif problem == "duplicate":
        bad.loc[0, ID_COLUMN] = bad.iloc[1][ID_COLUMN]
    elif problem == "missing_income":
        bad.loc[0, "annual_inc"] = np.nan
    elif problem == "missing_target":
        bad[TARGET] = bad[TARGET].astype(float)
        bad.loc[0, TARGET] = np.nan
    elif problem == "small":
        bad = bad.head(1000)
    else:
        bad = bad.drop(columns="issue_date")
    with pytest.raises(ValueError):
        build_simulation(bad, reference[ID_COLUMN], PARAMS)


def test_generate_persists_reproducible_artifacts_and_manifest(
    pool, reference, tmp_path
):
    pool_path, reference_path = (
        tmp_path / "pool.parquet",
        tmp_path / "reference.parquet",
    )
    pool.to_parquet(pool_path, index=False)
    reference.to_parquet(reference_path, index=False)
    output = tmp_path / "production"
    manifest = generate(pool_path, reference_path, output, PARAMS)
    assert (
        json.loads((output / "manifest.json").read_text(encoding="utf-8")) == manifest
    )
    for path, info in manifest["artifacts"].items():
        assert file_digest(output / path) == info["sha256"]
        assert len(pd.read_parquet(output / path)) == info["rows"]
    again = generate(pool_path, reference_path, output, PARAMS)
    assert again == manifest
    assert manifest["sources"]["pool"]["sha256"] == file_digest(pool_path)
    manifest_text = (output / "manifest.json").read_text(encoding="utf-8")
    assert not any(identifier in manifest_text for identifier in pool[ID_COLUMN])
    assert not any(identifier in manifest_text for identifier in reference[ID_COLUMN])


def test_invalid_generation_does_not_publish_partial_results(pool, reference, tmp_path):
    pool_path, reference_path = tmp_path / "pool.parquet", tmp_path / "ref.parquet"
    pool.head(1000).to_parquet(pool_path, index=False)
    reference.to_parquet(reference_path, index=False)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="insuficiente"):
        generate(pool_path, reference_path, output, PARAMS)
    assert not output.exists()
