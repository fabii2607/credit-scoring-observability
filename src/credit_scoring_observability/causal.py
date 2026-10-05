"""Análise causal do drift: DAG de hipóteses e atribuição por intervenção.

Hipóteses (DAG, também em docs/lgpd.md e no README):

    choque macro (inflação) ──► renda real ↓ ──► dti ↑ ───────────┐
                           └──► utilização do rotativo ↑ ────────┼──► default
    novos perfis de cliente ──► FICO ↓ ────────────────────────────┘
    renegociação / aperto de renda ──────────────► P(default | X) muda (concept)

Atribuição por intervenção do(X) sobre o MODELO: para cada grupo de variáveis
causalmente ligadas, a distribuição do lote é substituída pela da Referência
(mapeamento de quantis que PRESERVA A ORDEM dentro do lote, então as correlações
entre as colunas restauradas e o resto do perfil continuam) e mede-se quanto o
efeito no modelo volta ao normal.

    renda+dívida   annual_inc, dti   (dti = dívida / renda: restaurar só um quebra a relação)
    rotativo       revol_util, revol_bal
    score          fico_avg

O que isto mede: a SENSIBILIDADE do modelo a cada grupo, sob as hipóteses do DAG.
Não identifica a causa no mundo real — para isso seriam necessários dados de
intervenção ou um desenho quase-experimental. O método vem da Aula 4 de Drift
(restaurar uma feature isolada quebra as correlações; por isso, grupos).

Uso (`make causal`, depois do `make simulate` da Etapa 2):
    uv run python -m credit_scoring_observability.causal [--batches 2026-04 2026-06]
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from credit_scoring_observability.config import (
    PRODUCTION_DIR,
    REFERENCE_SAMPLE_FILE,
    REPORTS_DIR,
    TARGET,
)
from credit_scoring_observability.logger import get_logger, setup_logging

logger = get_logger(__name__)

CAUSAL_GROUPS: dict[str, list[str]] = {
    "renda+dívida": ["annual_inc", "dti"],
    "rotativo": ["revol_util", "revol_bal"],
    "score": ["fico_avg"],
}
CAUSAL_REPORT = REPORTS_DIR / "causal" / "causal_analysis.json"
EPSILON = 1e-6
PSI_ALERT = 0.10
AUC_DROP_ALERT = 0.05


def restore_group(
    batch: pd.DataFrame, reference: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """do(grupo = distribuição da Referência), preservando a ordem dentro do lote.

    Cada valor do lote vira o quantil da Referência na mesma posição relativa
    (rank) que ele ocupa no lote. Nulos continuam nulos (o pipeline imputa).
    """
    out = batch.copy()
    for col in columns:
        values = out[col]
        present = values.notna()
        ref_values = np.sort(reference[col].dropna().to_numpy(dtype=float))
        ranks = values[present].rank(method="first", pct=True).to_numpy()
        idx = np.clip(np.ceil(ranks * len(ref_values)).astype(int) - 1, 0, None)
        restored = values.astype(float).copy()
        restored[present] = ref_values[idx]
        out[col] = restored
    return out


def score_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI do score com bins pelos quantis da Referência.

    Cópia mínima para a análise causal; a Etapa 2 tem o PSI oficial das features.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    inner = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))[1:-1]
    ref_pct = np.bincount(
        np.searchsorted(inner, ref, side="right"), minlength=len(inner) + 1
    )
    cur_pct = np.bincount(
        np.searchsorted(inner, cur, side="right"), minlength=len(inner) + 1
    )
    ref_pct = np.clip(ref_pct / ref_pct.sum(), EPSILON, None)
    cur_pct = np.clip(cur_pct / cur_pct.sum(), EPSILON, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def model_effects(model, frame: pd.DataFrame, reference_score: np.ndarray) -> dict:
    """Efeitos de um lote no modelo: PSI do score, aprovação, score médio e AUC."""
    score = model.predict_proba(frame)
    effects = {
        "score_psi": score_psi(reference_score, score),
        "approval_rate": float(np.mean(score < model.threshold)),
        "mean_score": float(score.mean()),
        "roc_auc": None,
    }
    if TARGET in frame and frame[TARGET].nunique() == 2:
        effects["roc_auc"] = float(roc_auc_score(frame[TARGET], score))
    return effects


def recovery(drifted, intervened, reference) -> float | None:
    """Fração do desvio (lote - Referência) removida pela intervenção."""
    if drifted is None or intervened is None or reference is None:
        return None
    gap = drifted - reference
    if abs(gap) < 1e-9:
        return None
    return float((drifted - intervened) / gap)


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def diagnose(drifted: dict, reference: dict, all_restored: dict) -> str:
    """Covariate shift × concept drift a partir dos efeitos e da recuperação."""
    auc_drop = (
        reference["roc_auc"] - drifted["roc_auc"]
        if drifted["roc_auc"] is not None and reference["roc_auc"] is not None
        else 0.0
    )
    if drifted["score_psi"] > PSI_ALERT and auc_drop <= AUC_DROP_ALERT:
        return (
            "covariate shift: o score mudou e restaurar X o devolve à Referência; "
            "a AUC se mantém -> monitorar/recalibrar, sem retreino"
        )
    if auc_drop > AUC_DROP_ALERT and (all_restored["recovery_roc_auc"] or 0) < 0.5:
        return (
            "concept drift: X está estável (restaurar X não recupera a AUC) e a "
            "performance caiu -> P(Y|X) mudou, retreinar com dados recentes"
        )
    if drifted["score_psi"] > PSI_ALERT / 2 or auc_drop > AUC_DROP_ALERT / 2:
        return (
            "drift misto abaixo dos limiares: X e P(Y|X) podem ter mudado juntos; "
            "investigar o lote"
        )
    return "sem drift relevante atribuível aos grupos analisados"


def analyze_batch(
    batch: pd.DataFrame,
    reference: pd.DataFrame,
    model,
    batch_id: str,
    groups: dict[str, list[str]] = CAUSAL_GROUPS,
    reference_roc_auc: float | None = None,
) -> dict:
    """Intervenções em todas as combinações de grupos para um lote.

    `reference_roc_auc`: AUC da Referência fora do treino (padrão: a do pool nos
    metadados do modelo); a Referência de distribuição pode ter saído do treino.
    """
    reference_score = model.predict_proba(reference)
    ref_effects = model_effects(model, reference, reference_score)
    ref_effects["score_psi"] = 0.0
    if reference_roc_auc is None:
        reference_roc_auc = (
            getattr(model, "metadata", {}).get("metrics_on_test", {}).get("roc_auc")
        )
    if reference_roc_auc is not None:
        ref_effects["roc_auc"] = float(reference_roc_auc)
    drifted = model_effects(model, batch, reference_score)

    names = list(groups)
    combos = [c for k in range(1, len(names) + 1) for c in combinations(names, k)]
    interventions = {}
    for combo in combos:
        columns = [col for name in combo for col in groups[name]]
        effects = model_effects(
            model, restore_group(batch, reference, columns), reference_score
        )
        interventions[" + ".join(combo)] = {
            **{k: _round(v) for k, v in effects.items()},
            "recovery_score_psi": _round(
                recovery(drifted["score_psi"], effects["score_psi"], 0.0)
            ),
            "recovery_approval_rate": _round(
                recovery(
                    drifted["approval_rate"],
                    effects["approval_rate"],
                    ref_effects["approval_rate"],
                )
            ),
            "recovery_roc_auc": _round(
                recovery(drifted["roc_auc"], effects["roc_auc"], ref_effects["roc_auc"])
            ),
        }

    all_restored = interventions[" + ".join(names)]
    singles = [interventions[name]["recovery_score_psi"] or 0.0 for name in names]
    return {
        "batch_id": batch_id,
        "rows": len(batch),
        "reference": {k: _round(v) for k, v in ref_effects.items()},
        "drifted": {k: _round(v) for k, v in drifted.items()},
        "interventions": interventions,
        # Soma das recuperações isoladas ≠ conjunta: os efeitos se sobrepõem.
        "interaction_score_psi": _round(
            (all_restored["recovery_score_psi"] or 0.0) - sum(singles)
        ),
        "diagnosis": diagnose(drifted, ref_effects, all_restored),
    }


def available_batches(production_dir: Path = PRODUCTION_DIR) -> list[str]:
    """Lotes simulados em data/production (sem os de demonstração do contrato)."""
    if not production_dir.exists():
        return []
    return sorted(
        p.stem for p in production_dir.glob("*.parquet") if p.stem != "corrupted"
    )


def run(batch_ids: list[str] | None = None, output: Path = CAUSAL_REPORT) -> dict:
    from credit_scoring_observability.registry import load_model

    batch_ids = batch_ids or available_batches()
    if not batch_ids:
        raise SystemExit(
            "Nenhum lote em data/production/. Rode `make simulate` (Etapa 2) antes."
        )
    model = load_model()
    reference = pd.read_parquet(REFERENCE_SAMPLE_FILE)
    report = {
        "model_version": model.version,
        "groups": CAUSAL_GROUPS,
        "method_limit": (
            "Mede a sensibilidade do modelo a cada grupo sob as hipóteses do DAG; "
            "não prova a causa no mundo real."
        ),
        "batches": [
            analyze_batch(
                pd.read_parquet(PRODUCTION_DIR / f"{batch_id}.parquet"),
                reference,
                model,
                batch_id,
            )
            for batch_id in batch_ids
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    for batch in report["batches"]:
        logger.info(
            "atribuição causal",
            batch_id=batch["batch_id"],
            diagnostico=batch["diagnosis"],
        )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Atribuição causal do drift por lote")
    parser.add_argument(
        "--batches", nargs="*", help="ids em data/production/ (padrão: todos)"
    )
    args = parser.parse_args(argv)
    setup_logging()
    run(args.batches)


if __name__ == "__main__":
    main()
