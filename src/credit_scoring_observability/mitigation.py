"""Mitigação de viés por região: thresholds por grupo com paridade de FPR.

Avaliada e registrada, **não adotada** (`adopted_in_production: false`). Usar a
região explicitamente na decisão é tratamento diferenciado por um proxy de raça,
mais difícil de justificar (LGPD Art. 6º, IX, não discriminação) do que um
modelo que não a usa; a adoção dependeria de avaliação jurídica. O relatório
mostra o que a mitigação faria e quanto custaria.

Método (determinístico, ao contrário do `ThresholdOptimizer` do fairlearn, que
sorteia a decisão dentro do grupo e seria indefensável diante do Art. 20):
    1. meta = FPR global do threshold atual (bons pagadores negados);
    2. threshold do grupo = quantil (1 - meta) do score dos bons pagadores do grupo;
    3. ajuste numa metade do pool, avaliação na outra (estratificado por região e
       target).

Uso (`make fairness`):
    uv run python -m credit_scoring_observability.mitigation
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from credit_scoring_observability.config import (
    PRODUCTION_POOL_FILE,
    RANDOM_STATE,
    REGION_COLUMN,
    REPORTS_DIR,
    SCORE_COLUMN,
    TARGET,
)
from credit_scoring_observability.fairness import fairness_by_region
from credit_scoring_observability.logger import get_logger, setup_logging

logger = get_logger(__name__)

FAIRNESS_REPORT = REPORTS_DIR / "fairness" / "fairness_report.json"


def fpr_parity_thresholds(
    y_true: np.ndarray, score: np.ndarray, groups: pd.Series, target_fpr: float
) -> dict[str, float]:
    """Threshold por grupo que nega `target_fpr` dos bons pagadores do grupo."""
    frame = pd.DataFrame(
        {"y": np.asarray(y_true), "score": np.asarray(score), "g": np.asarray(groups)}
    )
    good = frame.loc[frame["y"] == 0]
    return {
        str(group): float(np.quantile(part["score"], 1 - target_fpr))
        for group, part in good.groupby("g")
    }


def decide(score: np.ndarray, groups: pd.Series, thresholds: dict[str, float]):
    """Decisão (1 = negar) com o threshold do grupo de cada linha."""
    limits = pd.Series(np.asarray(groups)).astype(str).map(thresholds).to_numpy()
    return (np.asarray(score) >= limits).astype(int)


def _summary(y_true, denied, groups) -> dict:
    """Aprovação, FPR/TPR por grupo e default entre os aprovados."""
    frame = pd.DataFrame(
        {"y": np.asarray(y_true), "denied": denied, "g": np.asarray(groups)}
    )
    approved = frame.loc[frame["denied"] == 0]
    by_group = frame.groupby("g")
    approval = 1 - by_group["denied"].mean()
    fpr = frame.loc[frame["y"] == 0].groupby("g")["denied"].mean()
    tpr = frame.loc[frame["y"] == 1].groupby("g")["denied"].mean()
    return {
        "approval_rate": round(float(1 - frame["denied"].mean()), 4),
        "default_rate_among_approved": round(float(approved["y"].mean()), 4),
        "disparate_impact": round(float(approval.min() / approval.max()), 4),
        "approval_by_group": approval.round(4).to_dict(),
        "fpr_by_group": fpr.round(4).to_dict(),
        "tpr_by_group": tpr.round(4).to_dict(),
        "fpr_gap": round(float(fpr.max() - fpr.min()), 4),
    }


def evaluate_mitigation(
    scored: pd.DataFrame,
    threshold: float,
    group_column: str = REGION_COLUMN,
    seed: int = RANDOM_STATE,
) -> dict:
    """Ajusta os thresholds numa metade do lote pontuado e avalia na outra."""
    strata = scored[group_column].astype(str) + "|" + scored[TARGET].astype(str)
    fit, holdout = train_test_split(
        scored, test_size=0.5, random_state=seed, stratify=strata
    )
    y_fit, s_fit, g_fit = fit[TARGET], fit[SCORE_COLUMN], fit[group_column]
    target_fpr = float(np.mean(s_fit[y_fit == 0] >= threshold))
    thresholds = fpr_parity_thresholds(y_fit, s_fit, g_fit, target_fpr)

    y, s, g = holdout[TARGET], holdout[SCORE_COLUMN], holdout[group_column]
    before = _summary(y, (s >= threshold).astype(int).to_numpy(), g)
    after = _summary(y, decide(s, g, thresholds), g)
    return {
        "method": "thresholds por grupo com paridade de FPR (determinísticos)",
        "group_column": group_column,
        "global_threshold": float(threshold),
        "target_fpr": round(target_fpr, 4),
        "group_thresholds": {k: round(v, 4) for k, v in thresholds.items()},
        "fit_rows": len(fit),
        "holdout_rows": len(holdout),
        "before": before,
        "after": after,
        "adopted_in_production": False,
        "reason_not_adopted": (
            "Usar a região na decisão é tratamento diferenciado por proxy de raça "
            "(LGPD Art. 6º, IX); a adoção depende de avaliação jurídica."
        ),
    }


def fairness_report(scored: pd.DataFrame, threshold: float, source: str) -> dict:
    """Fairness do modelo vigente + avaliação da mitigação, num lote pontuado."""
    return {
        "source": source,
        "rows": len(scored),
        "baseline": fairness_by_region(
            scored[TARGET], scored[SCORE_COLUMN], threshold, scored[REGION_COLUMN]
        ),
        "mitigation": evaluate_mitigation(scored, threshold),
    }


def main(pool_path: Path = PRODUCTION_POOL_FILE, output: Path = FAIRNESS_REPORT):
    from credit_scoring_observability.registry import load_model

    setup_logging()
    model = load_model()
    scored = model.score(pd.read_parquet(pool_path))
    report = {
        "model_version": model.version,
        **fairness_report(scored, model.threshold, source=pool_path.name),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
    base, mit = report["baseline"], report["mitigation"]
    logger.info(
        "fairness por região",
        impacto_desigual=base["disparate_impact"],
        regra_4_5=base["passes_four_fifths"],
        fpr_gap_antes=mit["before"]["fpr_gap"],
        fpr_gap_depois=mit["after"]["fpr_gap"],
        relatorio=str(output),
    )
    return report


if __name__ == "__main__":
    main()
