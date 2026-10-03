import json

import numpy as np
import pandas as pd
import pytest

from credit_scoring_observability.config import (
    ID_COLUMN,
    ISSUE_COLUMN,
    REGION_COLUMN,
    SPLIT_COLUMN,
    TARGET,
)
from credit_scoring_observability.data_contract import (
    MODEL_FEATURES,
    validate_model_input,
)
from credit_scoring_observability.parameters import SplitParams
from credit_scoring_observability.prepare import (
    DataLeakageError,
    build_frames,
    prepare,
    pseudonymize,
)
from credit_scoring_observability.preprocessing import (
    engineer_features,
    load_modeling_data,
    split_features_target,
    split_train_test,
)
from credit_scoring_observability.synthetic import make_model_batch

SALT = "sal-de-teste"
PARAMS = SplitParams(test_size=0.2, reference_sample_rows=300, chunksize=500)


@pytest.fixture
def raw_csv(tmp_path, raw_loans):
    path = tmp_path / "accepted.csv"
    raw_loans.to_csv(path, index=False)
    return path


@pytest.fixture
def loans(raw_csv):
    return load_modeling_data(
        raw_csv, chunksize=500, extra_columns=["id", "addr_state"]
    )


def test_pseudonymize_is_deterministic_and_salted():
    ids = pd.Series(["123", "456"])
    first = pseudonymize(ids, SALT)
    assert first.equals(pseudonymize(ids, SALT))
    assert not first.equals(pseudonymize(ids, "outro-sal"))
    assert first.str.len().eq(16).all()
    assert not first.isin(ids).any()


def test_extra_columns_do_not_change_rows_or_order(raw_csv):
    without = load_modeling_data(raw_csv, chunksize=500)
    with_meta = load_modeling_data(
        raw_csv, chunksize=500, extra_columns=["id", "addr_state"]
    )
    pd.testing.assert_frame_equal(without, with_meta.loc[:, without.columns])


def test_split_is_identical_to_notebooks(loans, raw_csv):
    prepared = build_frames(loans, SALT, PARAMS)
    # Fluxo dos notebooks 02–04, sem as colunas de metadado.
    nb = load_modeling_data(raw_csv, chunksize=500)
    X, y, _ = split_features_target(engineer_features(nb))
    X_train, X_test, _, _ = split_train_test(X, y, test_size=0.2, random_state=42)
    pd.testing.assert_frame_equal(
        prepared.reference.loc[:, X_train.columns],
        X_train.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        prepared.production_pool.loc[:, X_test.columns],
        X_test.reset_index(drop=True),
    )


def test_outputs_have_the_shared_schema(loans):
    prepared = build_frames(loans, SALT, PARAMS)
    expected = list(make_model_batch(n=10).columns)
    assert sorted(prepared.production_pool.columns) == sorted(expected)
    assert sorted(prepared.reference.columns) == sorted([*expected, SPLIT_COLUMN])
    assert (prepared.reference[SPLIT_COLUMN] == "train").all()
    assert len(prepared.reference_sample) == PARAMS.reference_sample_rows
    for frame in (prepared.reference, prepared.production_pool):
        assert "id" not in frame.columns
        assert "addr_state" not in frame.columns
        assert frame[ID_COLUMN].is_unique
        assert pd.api.types.is_datetime64_any_dtype(frame[ISSUE_COLUMN])
        assert frame[REGION_COLUMN].notna().all()
        assert set(frame[TARGET].unique()) <= {0, 1}
        validate_model_input(frame.loc[:, MODEL_FEATURES])


def test_reference_and_pool_do_not_overlap(loans):
    prepared = build_frames(loans, SALT, PARAMS)
    report = prepared.report
    assert report["id_overlap_reference_pool"] == 0
    assert (
        report["modeling_rows"]
        == report["reference_rows"] + (report["production_pool_rows"])
    )
    assert not set(prepared.reference[ID_COLUMN]) & set(
        prepared.production_pool[ID_COLUMN]
    )


def test_leakage_is_detected(loans):
    loans = loans.copy()
    loans["id"] = "mesmo-id"  # todo mundo vira o mesmo cliente
    with pytest.raises(DataLeakageError):
        build_frames(loans, SALT, PARAMS)


def test_prepare_writes_artifacts(raw_csv, tmp_path):
    out = tmp_path / "processed"
    prepared = prepare(raw_csv, output_dir=out, salt=SALT, params=PARAMS)
    assert {p.name for p in out.iterdir()} == {
        "reference.parquet",
        "reference_sample.parquet",
        "production_pool.parquet",
        "prepare_report.json",
    }
    pool = pd.read_parquet(out / "production_pool.parquet")
    assert len(pool) == prepared.report["production_pool_rows"]
    report = json.loads((out / "prepare_report.json").read_text(encoding="utf-8"))
    assert report["default_salt_used"] is False


def test_reference_sample_only_keeps_contract_valid_rows(loans):
    loans = loans.copy()
    loans.loc[loans.sample(frac=0.2, random_state=0).index, "annual_inc"] = np.nan
    prepared = build_frames(loans, SALT, PARAMS)
    sample = prepared.reference_sample
    dropped = prepared.report["reference_sample_dropped_by_contract"]
    assert dropped > 0
    assert len(sample) == PARAMS.reference_sample_rows - dropped
    assert sample["annual_inc"].notna().all()
    validate_model_input(sample.loc[:, MODEL_FEATURES])
    # O treino mantém as linhas com renda nula (o pipeline imputa).
    assert prepared.reference["annual_inc"].isna().any()
