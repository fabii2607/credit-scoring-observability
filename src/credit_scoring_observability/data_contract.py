"""Data contract for post-feature-engineering, pre-model Lending Club inputs.

The contract validates the 23 human-readable features consumed by the fitted
ColumnTransformer. It intentionally excludes `target`, `issue_date`, `loan_status`,
and the already encoded 44-column matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera import Check

# Keep aligned with split_features_target(engineer_features(...)).
MODEL_FEATURES = [
    "loan_amnt",
    "term",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "collections_12_mths_ex_med",
    "acc_now_delinq",
    "pub_rec_bankruptcies",
    "tax_liens",
    "application_type",
    "emp_length_years",
    "emp_length_missing",
    "fico_avg",
    "credit_history_years",
]

NUMERIC_FEATURES = [
    col
    for col in MODEL_FEATURES
    if col
    not in {"home_ownership", "verification_status", "purpose", "application_type"}
]
CATEGORICAL_FEATURES = [
    "home_ownership",
    "verification_status",
    "purpose",
    "application_type",
]


EMP_LENGTH_RULE = "emp_length_consistency"


def non_negative(*, nullable: bool = True) -> pa.Column:
    """Numeric values >= 0; NaN allowed only when the model imputes the field."""
    return pa.Column(float, Check.ge(0), nullable=nullable, coerce=True)


# These columns were observed in the EDA as available or mostly complete.
# Only annual_inc is strictly mandatory; other missing numeric values follow
# the training pipeline's median-imputation policy.
MODEL_INPUT_SCHEMA = pa.DataFrameSchema(
    columns={
        "loan_amnt": pa.Column(float, Check.gt(0), nullable=False, coerce=True),
        "term": pa.Column(float, Check.isin([36, 60]), nullable=False, coerce=True),
        "home_ownership": pa.Column(str, nullable=False, coerce=True),
        "annual_inc": non_negative(nullable=False),
        "verification_status": pa.Column(str, nullable=False, coerce=True),
        "purpose": pa.Column(str, nullable=False, coerce=True),
        "dti": non_negative(),
        "delinq_2yrs": non_negative(),
        "inq_last_6mths": non_negative(),
        "open_acc": non_negative(),
        "pub_rec": non_negative(),
        "revol_bal": non_negative(),
        # Utilização rotativa >100% pode representar uso acima do limite;
        # não imponha um teto artificial de 100%.
        "revol_util": non_negative(),
        "total_acc": non_negative(),
        "collections_12_mths_ex_med": non_negative(),
        "acc_now_delinq": non_negative(),
        "pub_rec_bankruptcies": non_negative(),
        "tax_liens": non_negative(),
        "application_type": pa.Column(
            str,
            Check.isin(["Individual", "Joint App"]),
            nullable=False,
            coerce=True,
        ),
        "emp_length_years": pa.Column(
            float,
            Check.isin(list(range(-1, 11))),
            nullable=False,
            coerce=True,
        ),
        "emp_length_missing": pa.Column(
            float,
            Check.isin([0, 1]),
            nullable=False,
            coerce=True,
        ),
        "fico_avg": pa.Column(
            float,
            Check.in_range(300, 850),
            nullable=False,
            coerce=True,
        ),
        "credit_history_years": non_negative(),
    },
    checks=[
        # Linha a linha (não `.all()`): o relatório aponta quais linhas falharam.
        Check(
            lambda frame: (
                (frame["emp_length_missing"] == 1) == (frame["emp_length_years"] == -1)
            ),
            name=EMP_LENGTH_RULE,
            error=f"{EMP_LENGTH_RULE}: emp_length_missing must match "
            "emp_length_years == -1",
        ),
    ],
    strict=True,
    coerce=True,
    name="credit_scoring_model_input_v1",
)


def validate_model_input(batch: pd.DataFrame) -> pd.DataFrame:
    """Validate a batch before the fitted preprocessor/model sees it.

    Raises pandera.errors.SchemaErrors and returns nothing downstream on failure.
    Missing/extra feature columns are rejected. Accepts both float32 and float64.
    """
    if not isinstance(batch, pd.DataFrame):
        raise TypeError("batch must be a pandas DataFrame")
    if batch.empty:
        raise ValueError("Empty batches cannot be scored")
    return MODEL_INPUT_SCHEMA.validate(batch.copy(deep=True), lazy=True)


def predict_proba_validated(model, batch: pd.DataFrame) -> np.ndarray:
    """Gate predictions: invalid data never reaches predict_proba."""
    validated = validate_model_input(batch)
    return model.predict_proba(validated.loc[:, MODEL_FEATURES].copy(deep=True))[:, 1]


def predict_classes_validated(
    model, batch: pd.DataFrame, threshold: float = 0.50
) -> np.ndarray:
    """Apply the versioned decision threshold outside sklearn.predict()."""
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    return (predict_proba_validated(model, batch) >= threshold).astype("int8")
