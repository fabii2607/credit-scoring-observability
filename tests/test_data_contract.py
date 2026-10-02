"""Data contract checks using synthetic rows, independent of Kaggle files."""
import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from credit_scoring_observability.data_contract import (
    MODEL_FEATURES,
    validate_model_input,
    predict_proba_validated,
    predict_classes_validated,
)

def valid_row():
    return {
        "loan_amnt": 12000.0,
        "term": 36,
        "home_ownership": "RENT",
        "annual_inc": 65000.0,
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "dti": 17.6,
        "delinq_2yrs": 0,
        "inq_last_6mths": 1,
        "open_acc": 11,
        "pub_rec": 0,
        "revol_bal": 11130.0,
        "revol_util": 52.2,
        "total_acc": 23,
        "collections_12_mths_ex_med": 0,
        "acc_now_delinq": 0,
        "pub_rec_bankruptcies": 0,
        "tax_liens": 0,
        "application_type": "Individual",
        "emp_length_years": 5,
        "emp_length_missing": 0,
        "fico_avg": 692.0,
        "credit_history_years": 14.7,
    }


def test_valid_batch_passes():
    batch = pd.DataFrame([valid_row()])
    validated = validate_model_input(batch)
    assert list(validated.columns) == MODEL_FEATURES
    assert len(validated) == 1


@pytest.mark.parametrize("field, value", [
    ("loan_amnt", -500.0),
    ("term", 48),
    ("annual_inc", np.nan),
    ("fico_avg", 910.0),
    ("dti", -1.0),
    ("emp_length_years", 15),
    ("application_type", "Invalid App"),
])
def test_invalid_values_are_rejected(field, value):
    row = valid_row()
    row[field] = value
    with pytest.raises(SchemaErrors):
        validate_model_input(pd.DataFrame([row]))


def test_missing_required_column_is_rejected():
    batch = pd.DataFrame([valid_row()]).drop(columns="fico_avg")
    with pytest.raises(SchemaErrors):
        validate_model_input(batch)


def test_extra_column_is_rejected():
    batch = pd.DataFrame([valid_row()])
    batch["loan_status"] = "Fully Paid"  # forbidden future/outcome column
    with pytest.raises(SchemaErrors):
        validate_model_input(batch)


def test_employment_flag_must_match_missing_code():
    row = valid_row()
    row["emp_length_missing"] = 1
    row["emp_length_years"] = 5
    with pytest.raises(SchemaErrors):
        validate_model_input(pd.DataFrame([row]))


def test_nullable_numeric_and_overlimit_revol_util_are_allowed():
    row = valid_row()
    row["dti"] = np.nan       # pipeline will impute it
    row["revol_util"] = 112.0  # can legitimately exceed 100%
    assert len(validate_model_input(pd.DataFrame([row]))) == 1


def test_bad_batch_never_reaches_the_model():
    class SpyModel:
        was_called = False
        def predict_proba(self, X):
            self.was_called = True
            return np.array([[0.8, 0.2]] * len(X))

    spy = SpyModel()
    row = valid_row()
    row["term"] = 48
    with pytest.raises(SchemaErrors):
        predict_proba_validated(spy, pd.DataFrame([row]))
    assert spy.was_called is False


def test_valid_batch_reaches_model_and_uses_threshold():
    class SpyModel:
        def predict_proba(self, X):
            return np.array([[0.725, 0.275]] * len(X))

    batch = pd.DataFrame([valid_row()])
    assert np.allclose(predict_proba_validated(SpyModel(), batch), [0.275])
    assert predict_classes_validated(SpyModel(), batch, threshold=0.2).tolist() == [1]
