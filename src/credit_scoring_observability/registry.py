"""Registro do modelo: salvar e carregar o pipeline junto com o threshold versionado.

`load_model()` é a interface que as outras etapas usam para pontuar lotes. Hoje
o modelo fica em `models/` (joblib + JSON de metadados). A Etapa 3 pode trocar
o armazenamento por MLflow (alias `@production`) por trás desta mesma interface,
sem mudar quem chama.

    from credit_scoring_observability.registry import load_model

    model = load_model()
    scored = model.score(batch)  # + colunas "score" e "prediction"

Carregue arquivos joblib apenas de origem confiável.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from credit_scoring_observability.config import (
    MODEL_FILE,
    MODEL_METADATA_FILE,
    PREDICTION_COLUMN,
    SCORE_COLUMN,
)
from credit_scoring_observability.data_contract import validate_model_input


@dataclass
class ScoringModel:
    """Pipeline treinado + threshold de decisão + metadados da versão."""

    pipeline: object
    threshold: float
    version: str
    feature_columns: list[str]
    metadata: dict = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilidade de inadimplência (classe 1), só depois do contrato.

        Aceita o lote com colunas extras (customer_id, region...): só as features
        vão para o contrato e para o modelo. Lote inválido levanta SchemaErrors e
        não é pontuado.
        """
        validated = validate_model_input(X.loc[:, self.feature_columns])
        return self.pipeline.predict_proba(validated.loc[:, self.feature_columns])[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Classe com o threshold versionado (não use pipeline.predict, que usa 0,5)."""
        return (self.predict_proba(X) >= self.threshold).astype(np.int8)

    def score(self, batch: pd.DataFrame) -> pd.DataFrame:
        """Cópia do lote com `score` (probabilidade) e `prediction` (0/1)."""
        scored = batch.copy()
        scored[SCORE_COLUMN] = self.predict_proba(batch)
        scored[PREDICTION_COLUMN] = (scored[SCORE_COLUMN] >= self.threshold).astype(
            np.int8
        )
        return scored


def save_model(
    pipeline: object,
    metadata: dict,
    model_path: Path = MODEL_FILE,
    metadata_path: Path = MODEL_METADATA_FILE,
) -> None:
    """Grava o pipeline (joblib) e os metadados (JSON com o decision_threshold)."""
    for key in ("decision_threshold", "feature_columns", "version"):
        if key not in metadata:
            raise ValueError(f"metadata precisa de '{key}'")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",  # LF também no Windows: sem diff falso no git
    )


def load_model(
    alias: str = "production",
    model_path: Path = MODEL_FILE,
    metadata_path: Path = MODEL_METADATA_FILE,
) -> ScoringModel:
    """Carrega o modelo em produção com o threshold dos metadados."""
    if alias != "production":
        raise NotImplementedError(
            "Só o alias 'production' existe; versões challenger/previous entram "
            "com o retreino (Etapa 2) e o registro no MLflow (Etapa 3)."
        )
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {model_path.parent}. Rode `make prepare` e "
            "`make train` (ou receba os dois arquivos do grupo)."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return ScoringModel(
        pipeline=joblib.load(model_path),
        threshold=float(metadata["decision_threshold"]),
        version=str(metadata["version"]),
        feature_columns=list(metadata["feature_columns"]),
        metadata=metadata,
    )
