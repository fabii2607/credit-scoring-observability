"""Viés por região: aprovação, TPR/FPR e razão de impacto desigual.

O Lending Club não tem idade, sexo nem raça (nenhum dado sensível do Art. 11 da
LGPD). O atributo auditado é a REGIÃO do cliente (Census dos EUA, derivada do
estado no `prepare`): ela não entra no modelo, mas é proxy conhecido de raça e
renda (*redlining*), então o modelo pode discriminar por ela indiretamente.

Convenção: decisão 1 = negar (score >= threshold); aprovação = 1 - taxa de
negação. Razão de impacto desigual = menor aprovação entre as regiões / maior;
a regra dos 4/5 trata < 0,8 como sinal de alerta.

Usado em:
    - relatório de governança (`make fairness`, README, model card);
    - monitoramento por lote, sem rótulo (`approval_disparity`, Etapa 3);
    - guardrail do retreino (`fairness_by_region`, Etapa 2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    count,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)

FOUR_FIFTHS = 0.8


def _groups(groups: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(np.asarray(groups)).astype(str)


def approval_disparity(prediction: np.ndarray, groups: pd.Series) -> dict:
    """Aprovação por região e razão de impacto desigual, sem rótulo.

    Serve ao monitoramento por lote: a decisão (1 = negar) existe no momento da
    pontuação, meses antes do desfecho do empréstimo.
    """
    approved = pd.Series(1 - np.asarray(prediction))
    rates = approved.groupby(_groups(groups)).mean()
    return {
        "approval_rate": rates.round(4).to_dict(),
        "disparate_impact": round(float(rates.min() / rates.max()), 4),
    }


def fairness_by_region(
    y_true: np.ndarray, score: np.ndarray, threshold: float, region: pd.Series
) -> dict:
    """Métricas por região para a decisão `score >= threshold` (negar).

    - `fpr`: bons pagadores negados (custo para quem pagaria);
    - `tpr`: inadimplentes barrados;
    - `default_rate`: inadimplência observada (parte da diferença de aprovação é
      risco real).
    """
    y_true = np.asarray(y_true).astype(int)
    denied = (np.asarray(score) >= threshold).astype(int)
    frame = MetricFrame(
        metrics={
            "denial_rate": selection_rate,
            "tpr": true_positive_rate,
            "fpr": false_positive_rate,
            "n": count,
        },
        y_true=y_true,
        y_pred=denied,
        sensitive_features=_groups(region),
    )
    by_group = frame.by_group
    approval = 1 - by_group["denial_rate"]
    default_rate = pd.Series(y_true).groupby(_groups(region)).mean()
    return {
        "threshold": float(threshold),
        "n": by_group["n"].astype(int).to_dict(),
        "approval_rate": approval.round(4).to_dict(),
        "tpr": by_group["tpr"].round(4).to_dict(),
        "fpr": by_group["fpr"].round(4).to_dict(),
        "default_rate": default_rate.round(4).to_dict(),
        "disparate_impact": round(float(approval.min() / approval.max()), 4),
        "tpr_gap": round(float(by_group["tpr"].max() - by_group["tpr"].min()), 4),
        "fpr_gap": round(float(by_group["fpr"].max() - by_group["fpr"].min()), 4),
        "passes_four_fifths": bool(approval.min() / approval.max() >= FOUR_FIFTHS),
    }
