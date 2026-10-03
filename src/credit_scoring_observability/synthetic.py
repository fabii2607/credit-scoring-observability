"""Dados sintéticos com o formato do Lending Club, para testes e para adiantar etapas.

Dois níveis, com os mesmos nomes de colunas dos artefatos reais:

- `make_raw_loans`: formato do CSV bruto do Kaggle (o que `prepare` lê), com as
  anomalias reais: desfechos fora do escopo (Current, Late), tempo de emprego
  ausente, DTI negativo, utilização do rotativo acima de 100%.
- `make_model_batch`: formato do que `prepare` grava e do que o contrato e o
  modelo consomem: `customer_id`, `issue_date`, `region`, as 23 features e o
  `target`. Passa no contrato. O default depende de FICO, DTI, renda, prazo e
  utilização, então um modelo treinado nele tem AUC bem acima de 0,5.

O CI não tem o CSV do Kaggle: todos os testes usam estas funções.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from credit_scoring_observability.config import (
    ID_COLUMN,
    ISSUE_COLUMN,
    REGION_COLUMN,
    STATE_TO_REGION,
    TARGET,
)
from credit_scoring_observability.data_contract import MODEL_FEATURES

HOME_OWNERSHIP = ["MORTGAGE", "RENT", "OWN"]
VERIFICATION = ["Not Verified", "Source Verified", "Verified"]
PURPOSES = [
    "debt_consolidation",
    "credit_card",
    "home_improvement",
    "other",
    "major_purchase",
    "small_business",
    "car",
    "medical",
]
EMP_LENGTH_LABELS = [
    "< 1 year",
    "1 year",
    *[f"{n} years" for n in range(2, 10)],
    "10+ years",
]
STATES = sorted(STATE_TO_REGION)


def default_probability(
    fico: np.ndarray,
    dti: np.ndarray,
    income: np.ndarray,
    term: np.ndarray,
    revol_util: np.ndarray,
) -> np.ndarray:
    """Probabilidade de default com sinal parecido com o real (~20% de positivos)."""
    logit = (
        -1.55
        - 0.020 * (fico - 700)
        + 0.030 * (np.nan_to_num(dti, nan=18.0) - 18)
        - 0.35 * np.log(income / 65_000)
        + 0.55 * (term == 60)
        + 0.006 * (np.nan_to_num(revol_util, nan=50.0) - 50)
    )
    return 1 / (1 + np.exp(-logit))


def _base_columns(n: int, rng: np.random.Generator) -> dict:
    """Colunas comuns aos dois formatos, em unidades do dado bruto."""
    fico_low = rng.choice(np.arange(660, 846, 5), size=n).astype(float)
    open_acc = rng.poisson(11, n) + 1
    return {
        "loan_amnt": rng.choice(np.arange(1_000, 40_001, 25), size=n).astype(float),
        "term": rng.choice([36, 60], size=n, p=[0.75, 0.25]),
        "home_ownership": rng.choice(HOME_OWNERSHIP, size=n, p=[0.5, 0.4, 0.1]),
        "annual_inc": np.round(rng.lognormal(np.log(65_000), 0.5, n), 2),
        "verification_status": rng.choice(VERIFICATION, size=n),
        "purpose": rng.choice(PURPOSES, size=n),
        "dti": np.round(rng.gamma(4.0, 4.5, n), 2),
        "delinq_2yrs": rng.poisson(0.3, n).astype(float),
        "inq_last_6mths": rng.poisson(0.8, n).astype(float),
        "open_acc": open_acc.astype(float),
        "pub_rec": rng.poisson(0.2, n).astype(float),
        "revol_bal": np.round(rng.lognormal(np.log(11_000), 0.9, n), 0),
        "revol_util": np.round(np.clip(rng.normal(52, 24, n), 0, 125), 1),
        "total_acc": (open_acc + rng.poisson(12, n)).astype(float),
        "collections_12_mths_ex_med": rng.poisson(0.02, n).astype(float),
        "acc_now_delinq": rng.poisson(0.005, n).astype(float),
        "pub_rec_bankruptcies": rng.poisson(0.12, n).astype(float),
        "tax_liens": rng.poisson(0.05, n).astype(float),
        "application_type": rng.choice(
            ["Individual", "Joint App"], size=n, p=[0.95, 0.05]
        ),
        "fico_range_low": fico_low,
        "fico_range_high": fico_low + 4,
        # Primeiro dia do mês, entre 2012-01 e 2018-12.
        "issue_month": pd.to_datetime(
            {
                "year": rng.integers(2012, 2019, n),
                "month": rng.integers(1, 13, n),
                "day": 1,
            }
        ),
        "history_years": rng.uniform(2, 35, n),
        "addr_state": rng.choice(STATES, size=n),
    }


def make_raw_loans(n: int = 2_000, seed: int = 42) -> pd.DataFrame:
    """Empréstimos no formato do CSV bruto (`accepted_2007_to_2018Q4.csv`).

    ~10% das linhas têm desfecho fora do escopo (Current, Late) e devem ser
    descartadas pelo `prepare`. Os ids são únicos.
    """
    rng = np.random.default_rng(seed)
    cols = _base_columns(n, rng)

    emp_idx = rng.integers(0, len(EMP_LENGTH_LABELS), n)
    emp_length = np.array(EMP_LENGTH_LABELS, dtype=object)[emp_idx]
    emp_length[rng.random(n) < 0.06] = np.nan

    dti = cols["dti"].copy()
    dti[rng.random(n) < 0.01] = -1.0  # inválido: vira ausente no engineer_features
    dti[rng.random(n) < 0.01] = np.nan

    p_default = default_probability(
        cols["fico_range_low"] + 2,
        dti,
        cols["annual_inc"],
        cols["term"],
        cols["revol_util"],
    )
    defaulted = rng.random(n) < p_default
    status = np.where(defaulted, "Charged Off", "Fully Paid").astype(object)
    out_of_scope = rng.random(n) < 0.10
    status[out_of_scope] = rng.choice(
        ["Current", "Late (31-120 days)"], out_of_scope.sum()
    )
    policy = (~out_of_scope) & (rng.random(n) < 0.01)
    status[policy & defaulted] = "Does not meet the credit policy. Status:Charged Off"
    status[policy & ~defaulted] = "Does not meet the credit policy. Status:Fully Paid"

    issue = pd.Series(cols["issue_month"])
    earliest = issue - pd.to_timedelta(cols["history_years"] * 365.25, unit="D")

    return pd.DataFrame(
        {
            "id": (np.arange(n) + 1_000_000).astype(str),
            "loan_amnt": cols["loan_amnt"],
            "term": [f" {t} months" for t in cols["term"]],
            "emp_length": emp_length,
            "home_ownership": cols["home_ownership"],
            "annual_inc": cols["annual_inc"],
            "verification_status": cols["verification_status"],
            "issue_d": issue.dt.strftime("%b-%Y"),
            "loan_status": status,
            "purpose": cols["purpose"],
            "addr_state": cols["addr_state"],
            "dti": dti,
            "delinq_2yrs": cols["delinq_2yrs"],
            "earliest_cr_line": earliest.dt.strftime("%b-%Y"),
            "fico_range_low": cols["fico_range_low"],
            "fico_range_high": cols["fico_range_high"],
            "inq_last_6mths": cols["inq_last_6mths"],
            "open_acc": cols["open_acc"],
            "pub_rec": cols["pub_rec"],
            "revol_bal": cols["revol_bal"],
            "revol_util": cols["revol_util"],
            "total_acc": cols["total_acc"],
            "collections_12_mths_ex_med": cols["collections_12_mths_ex_med"],
            "acc_now_delinq": cols["acc_now_delinq"],
            "pub_rec_bankruptcies": cols["pub_rec_bankruptcies"],
            "tax_liens": cols["tax_liens"],
            "application_type": cols["application_type"],
        }
    )


def make_model_batch(n: int = 2_000, seed: int = 42) -> pd.DataFrame:
    """Lote no formato pós-`prepare`: metadados + 23 features + target.

    Passa no contrato (`data_contract`). Ausências permitidas pelo contrato
    aparecem em `dti` e `revol_util` (o pipeline do modelo imputa).
    """
    rng = np.random.default_rng(seed)
    cols = _base_columns(n, rng)

    emp_years = rng.integers(0, 11, n).astype(float)
    emp_missing = rng.random(n) < 0.06
    emp_years[emp_missing] = -1.0

    dti = cols["dti"].copy()
    dti[rng.random(n) < 0.01] = np.nan
    revol_util = cols["revol_util"].copy()
    revol_util[rng.random(n) < 0.005] = np.nan

    fico_avg = cols["fico_range_low"] + 2
    p_default = default_probability(
        fico_avg, dti, cols["annual_inc"], cols["term"], revol_util
    )

    features = {
        **{k: cols[k] for k in MODEL_FEATURES if k in cols},
        "term": cols["term"].astype(float),
        "dti": dti,
        "revol_util": revol_util,
        "emp_length_years": emp_years,
        "emp_length_missing": emp_missing.astype(float),
        "fico_avg": fico_avg,
        "credit_history_years": np.round(cols["history_years"], 4),
    }
    ids = [
        hashlib.sha256(f"synthetic-{seed}:{i}".encode()).hexdigest()[:16]
        for i in range(n)
    ]
    return pd.DataFrame(
        {
            ID_COLUMN: ids,
            ISSUE_COLUMN: cols["issue_month"].to_numpy(),
            REGION_COLUMN: pd.Series(cols["addr_state"])
            .map(STATE_TO_REGION)
            .to_numpy(),
            **{col: features[col] for col in MODEL_FEATURES},
            TARGET: (rng.random(n) < p_default).astype("int8"),
        }
    )
