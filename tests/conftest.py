"""Fixtures compartilhadas. Todos os testes usam dados sintéticos (sem Kaggle)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_scoring_observability.synthetic import make_model_batch, make_raw_loans


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def raw_loans() -> pd.DataFrame:
    """Empréstimos no formato do CSV bruto do Lending Club."""
    return make_raw_loans(n=2_000, seed=7)


@pytest.fixture
def model_batch() -> pd.DataFrame:
    """Lote pós-prepare (metadados + 23 features + target), válido no contrato."""
    return make_model_batch(n=2_000, seed=11)


@pytest.fixture(scope="session")
def scoring_model():
    """Modelo V2 treinado em dados sintéticos (mesmo código do `make train`)."""
    from credit_scoring_observability.config import SPLIT_COLUMN
    from credit_scoring_observability.parameters import ThresholdGrid, TrainParams
    from credit_scoring_observability.registry import ScoringModel
    from credit_scoring_observability.train import train

    params = TrainParams(
        train_sample_rows=2_500,
        validation_fraction=0.2,
        min_precision=0.30,
        threshold_grid=ThresholdGrid(start=0.10, stop=0.90, step=0.025),
        class_weights=["standard"],
        max_iter=500,
        tol=1e-3,
    )
    reference = make_model_batch(n=3_000, seed=1)
    reference[SPLIT_COLUMN] = "train"
    pipeline, metadata = train(reference, make_model_batch(n=1_500, seed=2), params)
    return ScoringModel(
        pipeline=pipeline,
        threshold=metadata["decision_threshold"],
        version=metadata["version"],
        feature_columns=metadata["feature_columns"],
        metadata=metadata,
    )
