"""Viés por região: aprovação, TPR/FPR e razão de impacto desigual.

O Lending Club não tem idade, sexo nem raça (nenhum dado sensível do Art. 11 da
LGPD). O atributo auditado é a REGIÃO do cliente (Census dos EUA, derivada do
estado no `prepare`): ela não entra no modelo, mas é proxy conhecido de raça e
renda (*redlining*), então o modelo pode discriminar por ela indiretamente.

Convenção: decisão 1 = negar (score >= threshold); aprovação = 1 - taxa de
negação. Razão de impacto desigual = menor aprovação entre as regiões / maior;
a regra dos 4/5 trata < 0,8 como sinal de alerta.

Usado em:
    - relatório de governança no pool (`make fairness`, README, model card);
    - medição por lote simulado, com rótulo (`make fairness-batches`, abaixo);
    - `approval_disparity` mede a razão sem rótulo, mas não está integrada ao
      pipeline mensal; o guardrail de retreino é só uma política proposta.

Uso (`make fairness-batches`, precisa dos lotes de `make simulate`):
    uv run python -m credit_scoring_observability.fairness
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    count,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)

from credit_scoring_observability.config import (
    PRODUCTION_DIR,
    REGION_COLUMN,
    REPORTS_DIR,
    SCORE_COLUMN,
    TARGET,
)
from credit_scoring_observability.logger import get_logger, setup_logging

logger = get_logger(__name__)

FOUR_FIFTHS = 0.8

FAIRNESS_BY_BATCH_REPORT = REPORTS_DIR / "fairness" / "fairness_by_batch.json"
# Lote da demonstração de contrato: bloqueado antes da inferência, sem fairness.
CORRUPTED_BATCH_ID = "corrupted"


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


def fairness_by_batches(model, production_dir: Path = PRODUCTION_DIR) -> dict:
    """Fairness por região em cada lote mensal simulado (`make simulate`).

    Lê `manifest.json`, pontua cada lote de `batches` com o modelo vigente e
    aplica `fairness_by_region` ao rótulo do lote. O lote corrompido da demo
    não entra: o contrato o bloqueia antes da inferência. Os meses são cenários
    sintéticos; a medição não roda dentro do pipeline mensal.
    """
    manifest_path = Path(production_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    batches = {}
    for batch in manifest["batches"]:
        batch_id = str(batch["batch_id"])
        if batch_id == CORRUPTED_BATCH_ID:
            continue
        scored = model.score(pd.read_parquet(Path(production_dir) / batch["path"]))
        batches[batch_id] = {
            "scenario": batch.get("scenario"),
            "intensity": batch.get("intensity"),
            **fairness_by_region(
                scored[TARGET],
                scored[SCORE_COLUMN],
                model.threshold,
                scored[REGION_COLUMN],
            ),
        }
    return {
        "model_version": model.version,
        "threshold": float(model.threshold),
        "source": "data/production (make simulate)",
        "batches": batches,
    }


def main(
    production_dir: Path = PRODUCTION_DIR, output: Path = FAIRNESS_BY_BATCH_REPORT
) -> dict:
    from credit_scoring_observability.registry import load_model

    setup_logging()
    report = fairness_by_batches(load_model(), production_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    impacts = {k: v["disparate_impact"] for k, v in report["batches"].items()}
    logger.info(
        "fairness por região nos lotes simulados",
        lotes=len(impacts),
        impacto_desigual_min=min(impacts.values(), default=None),
        impacto_desigual_max=max(impacts.values(), default=None),
        abaixo_de_4_5=[k for k, v in impacts.items() if v < FOUR_FIFTHS],
        relatorio=str(output),
    )
    return report


if __name__ == "__main__":
    main()
