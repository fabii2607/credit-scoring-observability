import pandas as pd

from credit_scoring_observability.config import (
    ID_COLUMN,
    REGION_COLUMN,
    REGIONS,
    TARGET,
)
from credit_scoring_observability.data_contract import (
    MODEL_FEATURES,
    validate_model_input,
)
from credit_scoring_observability.preprocessing import (
    BAD_STATUS,
    BASELINE_FEATURES,
    GOOD_STATUS,
    engineer_features,
)


def test_raw_loans_have_raw_schema_and_anomalies(raw_loans):
    assert set(BASELINE_FEATURES) <= set(raw_loans.columns)
    assert {"id", "addr_state", "loan_status", "issue_d"} <= set(raw_loans.columns)
    in_scope = raw_loans["loan_status"].isin(GOOD_STATUS + BAD_STATUS)
    assert 0 < (~in_scope).sum() < len(raw_loans)  # Current/Late para descartar
    assert raw_loans["emp_length"].isna().any()
    assert (raw_loans["dti"] < 0).any()
    assert (raw_loans["revol_util"] > 100).any()
    assert raw_loans["id"].is_unique


def test_raw_loans_go_through_engineer_features(raw_loans):
    engineered = engineer_features(raw_loans)
    assert engineered["term"].isin([36, 60]).all()
    assert engineered["credit_history_years"].dropna().gt(0).all()


def test_model_batch_passes_contract(model_batch):
    validated = validate_model_input(model_batch.loc[:, MODEL_FEATURES])
    assert len(validated) == len(model_batch)
    assert model_batch[ID_COLUMN].is_unique
    assert set(model_batch[REGION_COLUMN]) <= set(REGIONS)
    assert pd.api.types.is_datetime64_any_dtype(model_batch["issue_date"])


def test_model_batch_default_rate_is_realistic(model_batch):
    assert 0.10 < model_batch[TARGET].mean() < 0.30
