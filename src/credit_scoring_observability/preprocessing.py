from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

GOOD_STATUS = [
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
]

BAD_STATUS = [
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
]

BASELINE_FEATURES = [
    "loan_amnt",
    "term",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
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
]

EMP_LENGTH_MAP = {
    "< 1 year": 0,
    "1 year": 1,
    "2 years": 2,
    "3 years": 3,
    "4 years": 4,
    "5 years": 5,
    "6 years": 6,
    "7 years": 7,
    "8 years": 8,
    "9 years": 9,
    "10+ years": 10,
}


def load_modeling_data(
    data_path: str | Path,
    chunksize: int = 100_000,
    extra_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Load only baseline columns and loans with a known final outcome.

    `extra_columns` reads metadata that is not a feature (e.g. `id`, `addr_state`)
    as text, without changing which rows are kept or their order.
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path.resolve()}")

    cols_to_load = [*BASELINE_FEATURES, "loan_status", "issue_d", *extra_columns]
    chunks: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        data_path,
        usecols=cols_to_load,
        dtype=dict.fromkeys(extra_columns, "string"),
        chunksize=chunksize,
        low_memory=False,
    ):
        filtered = chunk[chunk["loan_status"].isin(GOOD_STATUS + BAD_STATUS)].copy()
        chunks.append(filtered)

    df = pd.concat(chunks, ignore_index=True)

    df["target"] = df["loan_status"].isin(BAD_STATUS).astype(int)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic transformations validated during the EDA."""
    df = df.copy()

    # Loan term: "36 months" -> 36
    df["term"] = df["term"].str.extract(r"(\d+)", expand=False).astype(int)

    # Employment length
    df["emp_length_years"] = df["emp_length"].map(EMP_LENGTH_MAP)
    df["emp_length_missing"] = df["emp_length"].isna().astype(int)
    df["emp_length_years"] = df["emp_length_years"].fillna(-1)

    # Consolidated FICO score
    df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

    # Temporal features
    df["issue_date"] = pd.to_datetime(
        df["issue_d"],
        format="%b-%Y",
        errors="coerce",
    )

    df["earliest_cr_date"] = pd.to_datetime(
        df["earliest_cr_line"],
        format="%b-%Y",
        errors="coerce",
    )

    df["credit_history_years"] = (
        df["issue_date"] - df["earliest_cr_date"]
    ).dt.days / 365.25

    # Clearly invalid values become missing and are imputed later
    df.loc[df["credit_history_years"] < 0, "credit_history_years"] = np.nan
    df.loc[df["dti"] < 0, "dti"] = np.nan

    return df


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return predictive features, target and temporal metadata."""
    metadata = df[["issue_date", "target"]].copy()

    X = df.drop(
        columns=[
            "loan_status",
            "target",
            "emp_length",
            "fico_range_low",
            "fico_range_high",
            "earliest_cr_line",
            "issue_d",
            "issue_date",
            "earliest_cr_date",
        ]
    )

    y = df["target"].copy()

    return X, y, metadata


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def get_feature_groups(
    X: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Identify numeric and categorical input columns."""
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()

    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    return numeric_features, categorical_features


def build_preprocessor(
    X_train: pd.DataFrame,
    sparse: bool = False,
) -> ColumnTransformer:
    """Build the preprocessing pipeline without fitting it.

    `sparse=True` reproduces the V2 pipeline (notebook 04): float32 one-hot and a
    sparse output matrix, which saves memory with the saga solver.
    """
    numeric_features, categorical_features = get_feature_groups(X_train)
    encoder_options = {"sparse_output": True, "dtype": np.float32} if sparse else {}

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                RobustScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Unknown",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    **encoder_options,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        sparse_threshold=1.0 if sparse else 0.3,
    )


def create_reference_dataset(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    metadata: pd.DataFrame,
    n_samples: int = 100_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a stratified reference dataset from the training population."""
    if n_samples > len(X_train):
        raise ValueError("n_samples cannot be greater than the training dataset size.")

    reference_idx, _ = train_test_split(
        X_train.index,
        train_size=n_samples,
        random_state=random_state,
        stratify=y_train,
    )

    reference_df = X_train.loc[reference_idx].copy()
    reference_df["target"] = y_train.loc[reference_idx].values
    reference_df["issue_date"] = metadata.loc[
        reference_idx,
        "issue_date",
    ].values

    return reference_df


def validate_processed_sample(
    preprocessor: ColumnTransformer,
    X_sample: pd.DataFrame,
) -> dict[str, int]:
    """Validate transformed data for NaNs and infinite values."""
    transformed = preprocessor.transform(X_sample)

    array = transformed.toarray() if hasattr(transformed, "toarray") else transformed

    return {
        "nan_count": int(np.isnan(array).sum()),
        "inf_count": int(np.isinf(array).sum()),
        "n_output_features": int(array.shape[1]),
    }
