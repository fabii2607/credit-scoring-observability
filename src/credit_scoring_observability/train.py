"""Treino do modelo V2 (notebook 04) em script, a partir da Referência do `prepare`.

Fluxo (o mesmo do notebook 04):
1. Amostra estratificada de 300 mil linhas da Referência (treino).
2. Validação interna de 20%: ajusta Regressão Logística padrão e com
   `class_weight="balanced"`.
3. Escolhe configuração + threshold na validação: maior F2 com Precision >= 0,30.
4. Reajusta a configuração escolhida nas 300 mil linhas.
5. Avalia UMA vez no pool de produção (o teste da Etapa 1, nunca visto).
6. Salva `models/credit_scoring_optimized.joblib` + `_metadata.json` (com o
   `decision_threshold`), no mesmo formato do notebook.

Uso:
    uv run python -m credit_scoring_observability.train
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from credit_scoring_observability.config import (
    MODEL_FILE,
    MODEL_METADATA_FILE,
    PRODUCTION_POOL_FILE,
    RANDOM_STATE,
    REFERENCE_FILE,
    SPLIT_COLUMN,
    TARGET,
)
from credit_scoring_observability.data_contract import MODEL_FEATURES
from credit_scoring_observability.evaluate import (
    classification_metrics,
    select_threshold,
)
from credit_scoring_observability.logger import get_logger, setup_logging
from credit_scoring_observability.parameters import TrainParams, get_params
from credit_scoring_observability.preprocessing import (
    build_preprocessor,
    get_feature_groups,
)
from credit_scoring_observability.registry import save_model

logger = get_logger(__name__)

MODEL_VERSION = "V2_optimized"
CLASS_WEIGHTS = {"standard": None, "balanced": "balanced"}


def feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Só as 23 features, na ordem do parquet, com os tipos usados no notebook 04.

    Numéricas em float32 e categóricas como object (o SimpleImputer constante do
    pipeline espera texto puro, não o tipo `str` do pandas 3).
    """
    X = data.loc[:, [c for c in data.columns if c in MODEL_FEATURES]].copy()
    numeric, categorical = get_feature_groups(X)
    X[numeric] = X[numeric].astype("float32")
    X[categorical] = X[categorical].astype("object")
    return X


def build_model(X: pd.DataFrame, class_weight: str | None, params: TrainParams):
    """Pipeline completo V2: pré-processamento esparso + Regressão Logística (saga)."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(X, sparse=True)),
            (
                "model",
                LogisticRegression(
                    solver="saga",
                    max_iter=params.max_iter,
                    tol=params.tol,
                    class_weight=class_weight,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _fit(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> bool:
    """Ajusta o modelo; devolve True se houve ConvergenceWarning."""
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
    return any(issubclass(w.category, ConvergenceWarning) for w in seen)


def train(
    reference: pd.DataFrame,
    pool: pd.DataFrame,
    params: TrainParams,
) -> tuple[Pipeline, dict]:
    """Treina, seleciona o threshold, avalia no pool e devolve (pipeline, metadados)."""
    if SPLIT_COLUMN in reference:
        reference = reference.loc[reference[SPLIT_COLUMN] == "train"]
    X_full, y_full = feature_frame(reference), reference[TARGET]
    X_pool, y_pool = feature_frame(pool), pool[TARGET]

    sample_rows = min(params.train_sample_rows, len(X_full))
    if sample_rows < len(X_full):
        X_train, _, y_train, _ = train_test_split(
            X_full,
            y_full,
            train_size=sample_rows,
            random_state=RANDOM_STATE,
            stratify=y_full,
        )
    else:
        X_train, y_train = X_full, y_full

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=params.validation_fraction,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    validation_proba = {}
    for config in params.class_weights:
        model = build_model(X_fit, CLASS_WEIGHTS[config], params)
        start = time.perf_counter()
        _fit(model, X_fit, y_fit)
        validation_proba[config] = model.predict_proba(X_val)[:, 1]
        logger.info(
            "configuração avaliada na validação",
            config=config,
            minutos=round((time.perf_counter() - start) / 60, 2),
        )

    selection = select_threshold(
        y_val,
        validation_proba,
        params.threshold_grid.values(),
        params.min_precision,
    )
    logger.info(
        "threshold selecionado", config=selection.config, threshold=selection.threshold
    )

    final = build_model(X_train, CLASS_WEIGHTS[selection.config], params)
    converged_with_warning = _fit(final, X_train, y_train)
    proba_pool = final.predict_proba(X_pool)[:, 1]
    metrics = classification_metrics(y_pool, proba_pool, selection.threshold)

    metadata = {
        "version": MODEL_VERSION,
        "dataset": "wordsforthewise/lending-club",
        "positive_class": 1,
        "random_state": RANDOM_STATE,
        "train_sample_rows": len(X_train),
        "test_rows": len(X_pool),
        "validation_fraction": params.validation_fraction,
        "selection_criterion": (
            f"max F2 subject to precision >= {params.min_precision} on validation"
        ),
        "selected_model": selection.config,
        "class_weight": CLASS_WEIGHTS[selection.config],
        "decision_threshold": selection.threshold,
        "feature_columns": list(X_train.columns),
        "validation_metrics": selection.validation_metrics,
        "metrics_on_test": metrics,
        "test_set": "production_pool (teste reservado da Etapa 1)",
        "convergence_warning": converged_with_warning,
        "usage_note": (
            "Use registry.load_model(): aplica o contrato e o decision_threshold. "
            "pipeline.predict usa threshold 0.5."
        ),
    }
    return final, metadata


def main(
    reference_path: Path = REFERENCE_FILE,
    pool_path: Path = PRODUCTION_POOL_FILE,
) -> dict:
    setup_logging()
    if not reference_path.exists() or not pool_path.exists():
        raise FileNotFoundError("Rode `make prepare` antes do treino.")
    reference = pd.read_parquet(reference_path)
    pool = pd.read_parquet(pool_path)
    pipeline, metadata = train(reference, pool, get_params().train)
    save_model(pipeline, metadata, MODEL_FILE, MODEL_METADATA_FILE)
    test = metadata["metrics_on_test"]
    logger.info(
        "modelo salvo",
        caminho=str(MODEL_FILE),
        threshold=metadata["decision_threshold"],
        roc_auc=round(test["roc_auc"], 4),
        recall=round(test["recall"], 4),
        precision=round(test["precision"], 4),
    )
    return metadata


if __name__ == "__main__":
    main()
