"""Separação Referência × Produção a partir do CSV bruto do Lending Club.

Reproduz exatamente a população e o split dos notebooks 02–04 (mesmas linhas,
mesma ordem, `random_state=42`) e acrescenta o que as outras etapas precisam:

1. Lê as features do baseline e, como metadado, `id` e `addr_state`.
2. Pseudonimiza o `id` (SHA-256 com sal secreto do `.env`, 16 caracteres) e
   descarta o original. Pseudonimizado continua sendo dado pessoal (LGPD Art. 13).
3. Mapeia `addr_state` para a região (só para fairness; nunca vira feature).
4. Aplica `engineer_features` e o split 80/20 estratificado da Etapa 1.
5. Grava, sem imputação (o pipeline do modelo imputa):
   - `reference.parquet`: treino do modelo (Referência), `split = "train"`;
   - `reference_sample.parquet`: as 100 mil linhas do Reference Dataset do
     notebook 02, base de comparação do drift;
   - `production_pool.parquet`: o teste (269.620 linhas), nunca visto no treino,
     de onde a Etapa 2 tira os lotes mensais;
   - `prepare_report.json`: contagens e checagens.
6. Falha (`DataLeakageError`) se um mesmo `customer_id` cair nos dois lados.

Uso:
    uv run python -m credit_scoring_observability.prepare [--raw caminho.csv]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from credit_scoring_observability.config import (
    ID_COLUMN,
    ISSUE_COLUMN,
    PREPARE_REPORT_FILE,
    PROCESSED_DIR,
    PRODUCTION_POOL_FILE,
    RANDOM_STATE,
    RAW_ID_COLUMN,
    RAW_STATE_COLUMN,
    REFERENCE_FILE,
    REFERENCE_SAMPLE_FILE,
    REGION_COLUMN,
    SPLIT_COLUMN,
    STATE_TO_REGION,
    TARGET,
    raw_file,
)
from credit_scoring_observability.logger import get_logger, setup_logging
from credit_scoring_observability.parameters import SplitParams, get_params
from credit_scoring_observability.preprocessing import (
    engineer_features,
    load_modeling_data,
    split_features_target,
    split_train_test,
)

logger = get_logger(__name__)

DEFAULT_SALT = "troque-este-sal"
UNKNOWN_REGION = "Unknown"


class DataLeakageError(RuntimeError):
    """O mesmo cliente aparece na Referência e no pool de produção."""


@dataclass
class PreparedData:
    reference: pd.DataFrame
    reference_sample: pd.DataFrame
    production_pool: pd.DataFrame
    report: dict


@dataclass
class PrepareReport:
    modeling_rows: int
    reference_rows: int
    reference_sample_rows: int
    production_pool_rows: int
    default_rate_reference: float
    default_rate_pool: float
    issue_date_min: str
    issue_date_max: str
    rows_by_region: dict[str, int]
    unmapped_states: int
    id_overlap_reference_pool: int
    content_duplicates_across_splits: int
    default_salt_used: bool


def get_salt() -> str:
    """Lê o sal de pseudonimização do ambiente (.env)."""
    load_dotenv()
    salt = os.environ.get("PSEUDONYMIZATION_SALT", "")
    if not salt or salt == DEFAULT_SALT:
        logger.warning(
            "sal de pseudonimização padrão em uso: defina PSEUDONYMIZATION_SALT no .env"
        )
        return DEFAULT_SALT
    return salt


def pseudonymize(ids: pd.Series, salt: str) -> pd.Series:
    """SHA-256 salgado do id original, truncado em 16 caracteres hexadecimais.

    Determinístico para o mesmo sal (o controlador consegue refazer a associação
    para atender o titular, LGPD Art. 18); sem o sal, não é revertido por dicionário.
    """
    return ids.astype("string").map(
        lambda value: hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()[:16]
    )


def to_region(states: pd.Series) -> pd.Series:
    """Estado (UF americana) → região do Census; desconhecido vira "Unknown"."""
    return states.map(STATE_TO_REGION).fillna(UNKNOWN_REGION).astype(str)


def content_hash(features: pd.DataFrame) -> pd.Series:
    """Hash vetorizado do conteúdo das features (perfis idênticos entre splits).

    Trata NaN de forma consistente e independe do índice.
    """
    return pd.util.hash_pandas_object(features, index=False)


def reference_sample_index(
    X_train: pd.DataFrame, y_train: pd.Series, n_samples: int
) -> pd.Index:
    """Mesmo sorteio do `create_reference_dataset` (notebook 02)."""
    if n_samples > len(X_train):
        raise ValueError("n_samples cannot be greater than the training dataset size.")
    index, _ = train_test_split(
        X_train.index,
        train_size=n_samples,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    return index


def build_frames(loans: pd.DataFrame, salt: str, params: SplitParams) -> PreparedData:
    """Do bruto filtrado (`load_modeling_data`) aos três conjuntos da Etapa 1."""
    meta = pd.DataFrame(
        {
            ID_COLUMN: pseudonymize(loans[RAW_ID_COLUMN], salt),
            REGION_COLUMN: to_region(loans[RAW_STATE_COLUMN]),
        },
        index=loans.index,
    )
    unmapped = int(loans[RAW_STATE_COLUMN].map(STATE_TO_REGION).isna().sum())
    # id e UF saem antes do feature engineering: nunca chegam às features.
    loans = loans.drop(columns=[RAW_ID_COLUMN, RAW_STATE_COLUMN])

    X, y, temporal = split_features_target(engineer_features(loans))
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=params.test_size, random_state=RANDOM_STATE
    )

    def assemble(X_part: pd.DataFrame, y_part: pd.Series) -> pd.DataFrame:
        frame = pd.concat(
            [
                meta.loc[X_part.index, [ID_COLUMN]],
                temporal.loc[X_part.index, [ISSUE_COLUMN]],
                meta.loc[X_part.index, [REGION_COLUMN]],
                X_part,
            ],
            axis=1,
        )
        frame[TARGET] = y_part.astype("int8")
        return frame

    reference = assemble(X_train, y_train)
    reference[SPLIT_COLUMN] = "train"
    pool = assemble(X_test, y_test)
    sample_idx = reference_sample_index(X_train, y_train, params.reference_sample_rows)
    sample = reference.loc[sample_idx]

    overlap = len(set(reference[ID_COLUMN]) & set(pool[ID_COLUMN]))
    if overlap:
        raise DataLeakageError(
            f"{overlap} customer_id aparecem na Referência e no pool de produção"
        )
    dup_across = len(set(content_hash(X_train)) & set(content_hash(X_test)))

    report = PrepareReport(
        modeling_rows=len(X),
        reference_rows=len(reference),
        reference_sample_rows=len(sample),
        production_pool_rows=len(pool),
        default_rate_reference=round(float(y_train.mean()), 6),
        default_rate_pool=round(float(y_test.mean()), 6),
        issue_date_min=str(temporal[ISSUE_COLUMN].min().date()),
        issue_date_max=str(temporal[ISSUE_COLUMN].max().date()),
        rows_by_region=meta[REGION_COLUMN].value_counts().sort_index().to_dict(),
        unmapped_states=unmapped,
        id_overlap_reference_pool=overlap,
        content_duplicates_across_splits=dup_across,
        default_salt_used=salt == DEFAULT_SALT,
    )
    return PreparedData(
        reference=reference.reset_index(drop=True),
        reference_sample=sample.reset_index(drop=True),
        production_pool=pool.reset_index(drop=True),
        report=asdict(report),
    )


def save(prepared: PreparedData, output_dir: Path = PROCESSED_DIR) -> dict[str, Path]:
    """Grava os parquet e o relatório; devolve os caminhos."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "reference": output_dir / REFERENCE_FILE.name,
        "reference_sample": output_dir / REFERENCE_SAMPLE_FILE.name,
        "production_pool": output_dir / PRODUCTION_POOL_FILE.name,
        "report": output_dir / PREPARE_REPORT_FILE.name,
    }
    prepared.reference.to_parquet(paths["reference"], index=False)
    prepared.reference_sample.to_parquet(paths["reference_sample"], index=False)
    prepared.production_pool.to_parquet(paths["production_pool"], index=False)
    paths["report"].write_text(
        json.dumps(prepared.report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return paths


def prepare(
    raw_path: Path | None = None,
    output_dir: Path = PROCESSED_DIR,
    salt: str | None = None,
    params: SplitParams | None = None,
) -> PreparedData:
    """Executa o prepare de ponta a ponta e grava os artefatos."""
    params = params or get_params().split
    raw_path = raw_path or raw_file()
    logger.info("lendo dataset bruto", caminho=str(raw_path))
    loans = load_modeling_data(
        raw_path,
        chunksize=params.chunksize,
        extra_columns=[RAW_ID_COLUMN, RAW_STATE_COLUMN],
    )
    prepared = build_frames(loans, salt or get_salt(), params)
    paths = save(prepared, output_dir)
    logger.info(
        "prepare concluído",
        referencia=prepared.report["reference_rows"],
        pool=prepared.report["production_pool_rows"],
        amostra_referencia=prepared.report["reference_sample_rows"],
        saida=str(paths["reference"].parent),
    )
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=None, help="CSV bruto do Kaggle")
    args = parser.parse_args()
    setup_logging()
    prepare(args.raw)


if __name__ == "__main__":
    main()
