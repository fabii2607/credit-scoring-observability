"""Caminhos, nomes de colunas e constantes compartilhadas por todas as etapas.

As listas de features do modelo continuam em `data_contract.MODEL_FEATURES`
(fonte única). Aqui ficam só os nomes de colunas de metadados e os caminhos dos
artefatos que uma etapa entrega para a outra.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/credit_scoring_observability/config.py -> raiz do repositório.
# PROJECT_ROOT no ambiente permite rodar de outro diretório (ex.: Airflow).
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))

RANDOM_STATE = 42

# ── Parâmetros ───────────────────────────────────────────────────────────────
PARAMS_PATH = PROJECT_ROOT / "params.yaml"

# ── Dados ────────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_FILE_NAMES = ("accepted_2007_to_2018Q4.csv", "accepted_2007_to_2018Q4.csv.gz")

PROCESSED_DIR = DATA_DIR / "processed"
# Treino do modelo (Referência). Sem imputação: o pipeline do modelo imputa.
REFERENCE_FILE = PROCESSED_DIR / "reference.parquet"
# Amostra estratificada de 100 mil linhas do treino: base do drift (Etapa 2).
REFERENCE_SAMPLE_FILE = PROCESSED_DIR / "reference_sample.parquet"
# Teste reservado da Etapa 1, nunca visto no treino: de onde saem os lotes M1–M7.
PRODUCTION_POOL_FILE = PROCESSED_DIR / "production_pool.parquet"
PREPARE_REPORT_FILE = PROCESSED_DIR / "prepare_report.json"

# Lotes mensais simulados (Etapa 2) e lotes bloqueados pelo contrato.
PRODUCTION_DIR = DATA_DIR / "production"
QUARANTINE_DIR = DATA_DIR / "quarantine"

# ── Modelos e relatórios ─────────────────────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODELS_DIR / "credit_scoring_optimized.joblib"
MODEL_METADATA_FILE = MODELS_DIR / "credit_scoring_optimized_metadata.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
CONTRACT_REPORTS_DIR = REPORTS_DIR / "contracts"
LOGS_DIR = PROJECT_ROOT / "logs"

# ── Colunas ──────────────────────────────────────────────────────────────────
# Colunas do CSV bruto lidas só como metadado (nunca viram feature).
RAW_ID_COLUMN = "id"
RAW_STATE_COLUMN = "addr_state"

ID_COLUMN = "customer_id"  # SHA-256 salgado do id original (pseudonimização)
ISSUE_COLUMN = "issue_date"  # mês de originação, para ordenar e simular o tempo
REGION_COLUMN = "region"  # atributo sensível, usado só na análise de fairness
TARGET = "target"  # 1 = inadimplente
SPLIT_COLUMN = "split"  # "train" na Referência
ROW_HASH_COLUMN = "row_hash"  # hash do conteúdo das features (duplicatas/vazamento)

# Colunas que o modelo pontuado acrescenta ao lote (convenção para as Etapas 2–4).
SCORE_COLUMN = "score"  # probabilidade de inadimplência
PREDICTION_COLUMN = "prediction"  # 1 se score >= decision_threshold

METADATA_COLUMNS = [ID_COLUMN, ISSUE_COLUMN, REGION_COLUMN]

# Regiões do Census dos EUA a partir do estado (fairness por região).
_REGIONS = {
    "Northeast": "CT ME MA NH RI VT NJ NY PA",
    "Midwest": "IL IN MI OH WI IA KS MN MO NE ND SD",
    "South": "DE FL GA MD NC SC VA DC WV AL KY MS TN AR LA OK TX",
    "West": "AZ CO ID MT NV NM UT WY AK CA HI OR WA",
}
STATE_TO_REGION = {
    state: region for region, states in _REGIONS.items() for state in states.split()
}
REGIONS = list(_REGIONS)


def raw_file() -> Path:
    """Caminho do CSV bruto do Kaggle (aceita o .csv ou o .csv.gz baixado).

    Procura em `data/raw/` e nas subpastas: o zip do Kaggle cria a pasta
    `accepted_2007_to_2018q4.csv/` com o `.csv.gz` dentro. Só arquivos contam
    (no Windows, a pasta tem o mesmo nome do CSV, sem diferenciar maiúsculas).
    """
    wanted = [name.lower() for name in RAW_FILE_NAMES]
    found = sorted(
        (wanted.index(p.name.lower()), len(p.parts), p)
        for p in RAW_DIR.rglob("*")
        if p.is_file() and p.name.lower() in wanted
    )
    if found:
        return found[0][2]
    raise FileNotFoundError(
        f"Dataset não encontrado em {RAW_DIR}. Baixe o dataset do Kaggle "
        "(wordsforthewise/lending-club), descompacte e deixe o arquivo "
        "accepted_2007_to_2018Q4.csv (ou .csv.gz) em data/raw/ ou numa subpasta."
    )
