"""Métricas de classificação e seleção de threshold (critério do notebook 04).

Usado no treino (Etapa 1), na performance por lote e no retreino (Etapa 2) e
na fairness (Etapa 4). Convenção: classe positiva = inadimplente (1); score =
probabilidade de inadimplência; aprovado = score abaixo do threshold.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def ks_statistic(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Estatística KS: maior distância entre as curvas de bons e maus (max TPR - FPR)."""
    fpr, tpr, _ = roc_curve(y_true, proba)
    return float(np.max(tpr - fpr))


def approval_rate(proba: np.ndarray, threshold: float) -> float:
    """Fração aprovada: clientes com score abaixo do threshold."""
    return float(np.mean(np.asarray(proba) < threshold))


def classification_metrics(
    y_true: np.ndarray, proba: np.ndarray, threshold: float
) -> dict[str, float]:
    """Métricas de ranking (AUC, PR-AUC, KS, Brier) e de decisão no threshold."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(np.int8)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "ks": ks_statistic(y_true, proba),
        "brier": float(brier_score_loss(y_true, proba)),
        "approval_rate": approval_rate(proba, threshold),
        "threshold": float(threshold),
    }


def threshold_table(
    y_true: np.ndarray, proba: np.ndarray, thresholds: np.ndarray
) -> pd.DataFrame:
    """Precision, recall, F1, F2 e taxa de positivos para cada threshold."""
    rows = []
    for thr in thresholds:
        pred = (proba >= thr).astype(np.int8)
        rows.append(
            {
                "threshold": float(thr),
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0),
                "f2": fbeta_score(y_true, pred, beta=2, zero_division=0),
                "predicted_positive_rate": float(pred.mean()),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class ThresholdSelection:
    config: str
    threshold: float
    validation_metrics: dict[str, float]


def select_threshold(
    y_val: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    thresholds: np.ndarray,
    min_precision: float,
) -> ThresholdSelection:
    """Maior F2 com Precision >= `min_precision` na validação (notebook 04).

    `probabilities` mapeia o nome da configuração (ex.: "standard", "balanced")
    para os scores dela na validação; a escolha é conjunta (configuração +
    threshold). Desempate: recall e depois precision.
    """
    tables = []
    for config, proba in probabilities.items():
        table = threshold_table(y_val, proba, thresholds)
        table.insert(0, "config", config)
        tables.append(table)
    results = pd.concat(tables, ignore_index=True)

    eligible = results.loc[results["precision"] >= min_precision]
    if eligible.empty:
        raise ValueError(
            f"Nenhum threshold atinge precision >= {min_precision} na validação."
        )
    best = eligible.sort_values(
        ["f2", "recall", "precision"], ascending=[False, False, False]
    ).iloc[0]
    return ThresholdSelection(
        config=str(best["config"]),
        threshold=float(best["threshold"]),
        validation_metrics={
            k: float(best[k])
            for k in ("precision", "recall", "f1", "f2", "predicted_positive_rate")
        },
    )
