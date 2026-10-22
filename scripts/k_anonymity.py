"""Risco de reidentificação (k-anonimato) nos dados preparados.

Reproduz a tabela do item 2.2 de docs/lgpd.md: para cada combinação de
quase-identificadores, a fração de registros únicos (k = 1) e com k < 5 na
população de modelagem (reference.parquet + production_pool.parquet).

Uso (precisa de `make prepare`):
    uv run python scripts/k_anonymity.py
"""

from __future__ import annotations

import pandas as pd

from credit_scoring_observability.config import (
    ISSUE_COLUMN,
    PRODUCTION_POOL_FILE,
    REFERENCE_FILE,
    REGION_COLUMN,
)

LOAN_BAND_USD = 5_000
COMBINATIONS = {
    "região + mês + valor exato": [REGION_COLUMN, ISSUE_COLUMN, "loan_amnt"],
    "região + mês + faixa de US$ 5 mil": [REGION_COLUMN, ISSUE_COLUMN, "loan_band"],
    "região + mês + valor + renda": [
        REGION_COLUMN,
        ISSUE_COLUMN,
        "loan_amnt",
        "annual_inc",
    ],
    "região + mês + valor + renda + FICO": [
        REGION_COLUMN,
        ISSUE_COLUMN,
        "loan_amnt",
        "annual_inc",
        "fico_avg",
    ],
}


def k_anonymity_table(data: pd.DataFrame) -> pd.DataFrame:
    """Fração de registros com k = 1 e k < 5 por combinação de quase-identificadores."""
    data = data.assign(loan_band=(data["loan_amnt"] // LOAN_BAND_USD) * LOAN_BAND_USD)
    rows = []
    for label, columns in COMBINATIONS.items():
        k = data.groupby(columns, dropna=False, observed=True)[columns[0]].transform(
            "size"
        )
        rows.append(
            {
                "quase-identificadores": label,
                "k = 1": (k == 1).mean(),
                "k < 5": (k < 5).mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    columns = [REGION_COLUMN, ISSUE_COLUMN, "loan_amnt", "annual_inc", "fico_avg"]
    data = pd.concat(
        [
            pd.read_parquet(REFERENCE_FILE, columns=columns),
            pd.read_parquet(PRODUCTION_POOL_FILE, columns=columns),
        ],
        ignore_index=True,
    )
    table = k_anonymity_table(data)
    print(f"População de modelagem: {len(data):,} registros\n".replace(",", "."))
    print("| Quase-identificadores | Registros únicos (k = 1) | k < 5 |")
    print("|---|---|---|")
    for row in table.itertuples(index=False):
        k1 = f"{row[1]:.1%}".replace(".", ",")
        k5 = f"{row[2]:.1%}".replace(".", ",")
        print(f"| {row[0]} | {k1} | {k5} |")


if __name__ == "__main__":
    main()
