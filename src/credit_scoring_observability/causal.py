"""Análise causal do drift: DAG de hipóteses e atribuição por intervenção.

Hipóteses (DAG, também em docs/lgpd.md e no README):

    choque macro (inflação) ──► renda real ↓ ──► dti ↑ ───────────┐
                           └──► utilização do rotativo ↑ ────────┼──► default
    novos perfis de cliente ──► FICO ↓ ────────────────────────────┘
    renegociação / aperto de renda ──────────────► P(default | X) muda (concept)

Perturbação do MODELO por restauração de marginais, em grupos relacionados:
renda+dívida (annual_inc, dti), rotativo (revol_util, revol_bal), score (fico_avg).
O mapeamento por quantis é monotônico por coluna. Não impõe as equações do DAG,
não preserva necessariamente correlações lineares/distribuição conjunta e não
equivale à identificação de do(X) no mundo real. Mede sensibilidade do modelo.
O DAG é uma hipótese ilustrativa, não estimada com os dados.

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
    MODEL_FILE,
    MODEL_METADATA_FILE,
    PRODUCTION_DIR,
    REFERENCE_SAMPLE_FILE,
    REPORTS_DIR,
    SPLIT_COLUMN,
    TARGET,
)
from credit_scoring_observability.drift_report import safe_batch_id, write_json
from credit_scoring_observability.logger import get_logger, setup_logging
from credit_scoring_observability.parameters import get_params
from credit_scoring_observability.simulation import file_digest
from credit_scoring_observability.stats import psi_quantile
from credit_scoring_observability.validate import validate_batch

logger = get_logger(__name__)

CAUSAL_GROUPS: dict[str, list[str]] = {
    "renda+dívida": ["annual_inc", "dti"],
    "rotativo": ["revol_util", "revol_bal"],
    "score": ["fico_avg"],
}
CAUSAL_REPORT = REPORTS_DIR / "causal" / "causal_analysis.json"
PSI_ALERT = 0.10
AUC_DROP_ALERT = 0.05


def restore_group(
    batch: pd.DataFrame, reference: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Restaura marginais por quantis sem alegar intervenção causal identificada.

    Cada valor do lote vira o quantil da Referência na mesma posição relativa
    (rank) que ele ocupa no lote. Nulos continuam nulos (o pipeline imputa).
    """
    out = batch.copy()
    for col in columns:
        values = out[col]
        present = values.notna()
        ref_values = np.sort(reference[col].dropna().to_numpy(dtype=float))
        if not len(ref_values):
            raise ValueError(f"Referência sem valores observados para {col}")
        ranks = values[present].rank(method="average", pct=True).to_numpy()
        idx = np.clip(np.ceil(ranks * len(ref_values)).astype(int) - 1, 0, None)
        restored = values.astype(float).copy()
        restored[present] = ref_values[idx]
        out[col] = restored
    return out


def score_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Mesmo PSI por quantis do monitoramento, sem implementação paralela."""
    value = psi_quantile(reference, current, bins=bins)
    if value is None:
        raise ValueError("PSI do score indisponível")
    return value


def model_effects(model, frame: pd.DataFrame, reference_score: np.ndarray) -> dict:
    """Efeitos de um lote no modelo: PSI do score, aprovação, score médio e AUC."""
    score = model.predict_proba(frame)
    effects = {
        "score_psi": score_psi(reference_score, score),
        "approval_rate": float(np.mean(score < model.threshold)),
        "mean_score": float(score.mean()),
        "roc_auc": None,
    }
    if TARGET in frame:
        known = frame[TARGET].notna().to_numpy()
        if frame.loc[known, TARGET].nunique() == 2:
            effects["roc_auc"] = float(
                roc_auc_score(frame.loc[known, TARGET], score[known])
            )
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


def diagnose(
    drifted: dict,
    reference: dict,
    all_restored: dict,
    auc_drop_limit: float = AUC_DROP_ALERT,
    psi_limit: float = PSI_ALERT,
) -> str:
    """Hipóteses de sensibilidade, sem substituir a decisão do gate."""
    if drifted["roc_auc"] is None or reference["roc_auc"] is None:
        return "performance indisponível: não concluir ausência de concept drift"
    auc_drop = reference["roc_auc"] - drifted["roc_auc"]
    score_recovered = (all_restored["recovery_score_psi"] or 0) >= 0.5
    if (
        drifted["score_psi"] > psi_limit
        and auc_drop <= auc_drop_limit
        and score_recovered
    ):
        return (
            "covariate shift compatível: score deslocado e parcialmente recuperado "
            "pela restauração de marginais; investigar entradas e calibração"
        )
    if auc_drop > auc_drop_limit and (all_restored["recovery_roc_auc"] or 0) < 0.5:
        return (
            "concept drift suspeito: queda de AUC sem recuperação suficiente pela "
            "restauração dos grupos; verificar X e rótulos e consultar o gate; "
            "não prova mudança de P(Y|X) ou causa econômica"
        )
    if drifted["score_psi"] > psi_limit / 2 or auc_drop > auc_drop_limit / 2:
        return "sinais para investigação: restauração de marginais não identifica o mecanismo do drift"
    return "sem sinais relevantes pelos critérios de sensibilidade avaliados"


def analyze_batch(
    batch: pd.DataFrame,
    reference: pd.DataFrame,
    model,
    batch_id: str,
    groups: dict[str, list[str]] = CAUSAL_GROUPS,
    reference_roc_auc: float | None = None,
    auc_drop_limit: float = AUC_DROP_ALERT,
    psi_limit: float = PSI_ALERT,
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
    ref_effects["roc_auc"] = (
        float(reference_roc_auc) if reference_roc_auc is not None else None
    )
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
        "diagnosis": diagnose(
            drifted, ref_effects, all_restored, auc_drop_limit, psi_limit
        ),
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
    if len(set(batch_ids)) != len(batch_ids):
        raise ValueError("Seleção duplicada")
    model = load_model()
    params = get_params()
    manifest_path = PRODUCTION_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference = pd.read_parquet(REFERENCE_SAMPLE_FILE).drop(
        columns=SPLIT_COLUMN, errors="ignore"
    )
    control_path = PRODUCTION_DIR / "reserves/control.parquet"
    if (
        file_digest(control_path)
        != manifest["artifacts"]["reserves/control.parquet"]["sha256"]
    ):
        raise ValueError("Controle difere do manifesto")
    control = pd.read_parquet(control_path)
    for name, frame in [("reference_sample", reference), ("control", control)]:
        if not validate_batch(frame, name).passed:
            raise ValueError("Referência/controle inválido")
    control_auc = model_effects(model, control, model.predict_proba(reference))[
        "roc_auc"
    ]
    batches = []
    for batch_id in batch_ids:
        safe_batch_id(batch_id)
        path = PRODUCTION_DIR / f"{batch_id}.parquet"
        if batch_id == "corrupted" or path.name not in manifest["artifacts"]:
            raise ValueError("Lote não é um mês regular do manifesto")
        if file_digest(path) != manifest["artifacts"][path.name]["sha256"]:
            raise ValueError("Lote difere do manifesto")
        frame = pd.read_parquet(path)
        if not validate_batch(frame, batch_id).passed:
            raise ValueError("Lote inválido; análise de sensibilidade não executada")
        result = analyze_batch(
            frame,
            reference,
            model,
            batch_id,
            reference_roc_auc=control_auc,
            auc_drop_limit=params.performance.auc_drop,
            psi_limit=params.drift.psi_warning,
        )
        result["batch_sha256"] = file_digest(path)
        batches.append(result)
    report = {
        "model_version": model.version,
        "model_sha256": file_digest(MODEL_FILE),
        "metadata_sha256": file_digest(MODEL_METADATA_FILE),
        "threshold": model.threshold,
        "groups": CAUSAL_GROUPS,
        "baseline_source": "reserves/control.parquet (fora do treino)",
        "control_sha256": file_digest(control_path),
        "reference_sha256": file_digest(REFERENCE_SAMPLE_FILE),
        "simulation_manifest_sha256": file_digest(manifest_path),
        "implementation_sha256": file_digest(Path(__file__)),
        "auc_drop_limit": params.performance.auc_drop,
        "score_psi_limit": params.drift.psi_warning,
        "method_limit": (
            "Sensibilidade do modelo à restauração de marginais. Não identifica "
            "causa econômica, não preserva necessariamente relações estruturais "
            "ou correlações lineares e não substitui a recomendação do gate."
        ),
        "batches": batches,
    }
    write_json(output, report)
    for batch in report["batches"]:
        logger.info(
            "análise de sensibilidade",
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
