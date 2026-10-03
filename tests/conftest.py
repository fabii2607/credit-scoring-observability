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
