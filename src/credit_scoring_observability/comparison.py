"""Comparação pareada entre registros originais e suas versões simuladas."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_scoring_observability.config import ID_COLUMN, SCORE_COLUMN, TARGET
from credit_scoring_observability.drift import detect_drift
from credit_scoring_observability.parameters import DriftParams, PerformanceParams
from credit_scoring_observability.performance import (
    auc_difference_interval,
    evaluate_performance,
)
from credit_scoring_observability.results import (
    DriftResult,
    InterventionResult,
    PerformanceResult,
)
from credit_scoring_observability.simulation import DATA_FEATURES


def compare_intervention(
    reference_scored: pd.DataFrame,
    control_scored: pd.DataFrame,
    original_scored: pd.DataFrame,
    current_scored: pd.DataFrame,
    current_drift: DriftResult,
    current_perf: PerformanceResult,
    scenario: str,
    threshold: float,
    drift_params: DriftParams | None = None,
    performance_params: PerformanceParams | None = None,
) -> InterventionResult:
    """Mantém identidade e ordem; PSI não é decomposto por subtração.

    Compara original×referência, alterado×referência e original×alterado.
    Bootstrap pareado é estratificado por target ORIGINAL. Diagnóstico refere-se
    ao efeito da intervenção sintética naquela amostra, não à economia real.
    """
    dp, pp = drift_params or DriftParams(), performance_params or PerformanceParams()
    if scenario not in {"stable", "data_drift", "concept_drift"}:
        raise ValueError("Cenário de intervenção desconhecido")
    if (
        not original_scored[ID_COLUMN].is_unique
        or not current_scored[ID_COLUMN].is_unique
    ):
        raise ValueError("Comparação pareada exige ids únicos")
    if len(original_scored) != len(current_scored) or not original_scored[
        ID_COLUMN
    ].reset_index(drop=True).equals(current_scored[ID_COLUMN].reset_index(drop=True)):
        raise ValueError(
            "Registros originais e alterados precisam da mesma identidade e ordem"
        )
    allowed = (
        {*DATA_FEATURES, SCORE_COLUMN, "prediction"}
        if scenario == "data_drift"
        else {TARGET}
        if scenario == "concept_drift"
        else set()
    )
    fixed = [c for c in original_scored if c not in allowed]
    if set(original_scored) != set(current_scored) or not original_scored[
        fixed
    ].reset_index(drop=True).equals(current_scored[fixed].reset_index(drop=True)):
        raise ValueError("Intervenção modificou colunas fora do cenário declarado")
    original_drift = detect_drift(
        reference_scored, original_scored, current_drift.batch_id + "_original", dp
    )
    paired_drift = detect_drift(
        original_scored, current_scored, current_drift.batch_id + "_intervention", dp
    )
    original_perf = evaluate_performance(
        control_scored,
        original_scored,
        threshold,
        pp.model_copy(update={"bootstrap_samples": 0}),
        current_perf.baseline_source or "provided_control",
    )
    features = []
    for before, after, paired in zip(
        original_drift.features,
        current_drift.features,
        paired_drift.features,
        strict=True,
    ):
        features.append(
            {
                "feature": before.feature,
                "original_psi": before.psi,
                "current_psi": after.psi,
                "original_ks_d": before.ks_d,
                "current_ks_d": after.ks_d,
                "original_js": before.categorical_distance,
                "current_js": after.categorical_distance,
                "intervention_psi": paired.psi,
                "intervention_ks_d": paired.ks_d,
                "intervention_js": paired.categorical_distance,
                "original_drifted": before.drifted,
                "current_drifted": after.drifted,
            }
        )
    result = InterventionResult(
        batch_id=current_drift.batch_id,
        scenario=scenario,
        original_performance=original_perf.to_dict(),
        original_drifted_features=original_drift.drifted_features,
        current_drifted_features=current_drift.drifted_features,
        original_drift_share=original_drift.drift_share,
        current_drift_share=current_drift.drift_share,
        intervention_drift_share=paired_drift.drift_share,
        intervention_score_psi=paired_drift.score_psi,
        features=features,
    )
    if TARGET not in original_scored or TARGET not in current_scored:
        result.unavailable_reason = "missing_target"
        return result
    known = (
        original_scored[TARGET].notna().to_numpy()
        & current_scored[TARGET].notna().to_numpy()
    )
    result.paired_rows = int(known.sum())
    before_y = original_scored[TARGET].to_numpy(dtype=float, na_value=np.nan)[known]
    after_y = current_scored[TARGET].to_numpy(dtype=float, na_value=np.nan)[known]
    if len(np.unique(before_y)) != 2 or len(np.unique(after_y)) != 2:
        result.unavailable_reason = "single_class_or_missing_paired_target"
        return result
    from sklearn.metrics import roc_auc_score

    before_score = original_scored[SCORE_COLUMN].to_numpy(dtype=float)[known]
    after_score = current_scored[SCORE_COLUMN].to_numpy(dtype=float)[known]
    result.paired_auc_drop = float(
        roc_auc_score(before_y, before_score) - roc_auc_score(after_y, after_score)
    )
    result.paired_auc_drop_ci_low, result.paired_auc_drop_ci_high = (
        auc_difference_interval(
            before_y, before_score, after_y, after_score, pp, paired=True
        )
    )
    return result
