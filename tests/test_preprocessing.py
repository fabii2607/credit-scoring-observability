import numpy as np
import pandas as pd

from credit_scoring_observability.preprocessing import (
    build_preprocessor,
    engineer_features,
)


def test_engineer_features():
    df = pd.DataFrame(
        {
            "term": ["36 months", "60 months"],
            "emp_length": ["10+ years", None],
            "fico_range_low": [680.0, 700.0],
            "fico_range_high": [684.0, 704.0],
            "issue_d": ["Dec-2015", "Jan-2016"],
            "earliest_cr_line": ["Jan-2005", "Jan-2006"],
            "dti": [18.0, -1.0],
        }
    )

    result = engineer_features(df)

    assert result["term"].tolist() == [36, 60]
    assert result["emp_length_years"].tolist() == [10.0, -1.0]
    assert result["emp_length_missing"].tolist() == [0, 1]
    assert result["fico_avg"].tolist() == [682.0, 702.0]
    assert result["credit_history_years"].min() > 0
    assert pd.isna(result.loc[1, "dti"])


def test_build_preprocessor():
    X = pd.DataFrame(
        {
            "loan_amnt": [10000.0, 15000.0, np.nan],
            "term": [36, 60, 36],
            "home_ownership": ["RENT", "MORTGAGE", None],
            "purpose": ["credit_card", "debt_consolidation", "car"],
        }
    )

    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)

    array = (
        transformed.toarray()
        if hasattr(transformed, "toarray")
        else transformed
    )

    assert np.isnan(array).sum() == 0
    assert np.isinf(array).sum() == 0
    assert array.shape[0] == len(X)
