"""Performance realizada por lote, comparada a controle pontuado fora do treino."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from credit_scoring_observability.config import SCORE_COLUMN, TARGET
from credit_scoring_observability.parameters import PerformanceParams
from credit_scoring_observability.results import PerformanceResult


def _scores(frame: pd.DataFrame) -> np.ndarray:
    scores = frame[SCORE_COLUMN].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores devem ser probabilidades finitas em [0, 1]")
    return scores


def _observed(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str | None]:
    scores = _scores(frame)
    if TARGET not in frame:
        return np.array([]), np.array([]), "missing_target"
    observed = frame[TARGET].notna().to_numpy()
    y = frame.loc[observed, TARGET].to_numpy(dtype=float)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Target observado deve ser binário")
    reason = (
        "missing_target"
        if not len(y)
        else "single_class_target"
        if len(np.unique(y)) < 2
        else None
    )
    return y, scores[observed], reason


def _auc_interval(
    y: np.ndarray, scores: np.ndarray, params: PerformanceParams
) -> tuple[float | None, float | None]:
    if not params.bootstrap_samples:
        return None, None
    rng = np.random.default_rng(params.seed)
    groups = [np.flatnonzero(y == value) for value in (0, 1)]
    values = []
    for _ in range(params.bootstrap_samples):
        idx = np.concatenate(
            [rng.choice(group, len(group), replace=True) for group in groups]
        )
        values.append(roc_auc_score(y[idx], scores[idx]))
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def evaluate_performance(
    reference_scored: pd.DataFrame,
    batch_scored: pd.DataFrame,
    threshold: float,
    params: PerformanceParams | None = None,
    baseline_source: str = "provided_control",
) -> PerformanceResult:
    """reference_scored deve ser CONTROLE fora do treino, não referência de drift.

    O chamador garante origem e disjunção do controle. Threshold é o limiar de
    decisão do modelo; auc_drop é outro parâmetro. CI 95% via bootstrap
    estratificado da AUC atual, não intervalo da diferença entre AUCs.
    Sem target ou sem duas classes, nenhuma queda é inferida. Rótulos parciais
    são avaliados apenas onde conhecidos; labeled_rows explicita a cobertura.
    """
    params = params or PerformanceParams()
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Threshold deve estar entre 0 e 1")
    scores = _scores(batch_scored)
    y, labeled_scores, reason = _observed(batch_scored)
    ref_y, ref_scores, ref_reason = _observed(reference_scored)
    result = PerformanceResult(
        rows=len(batch_scored),
        control_rows=len(reference_scored),
        control_labeled_rows=len(ref_y),
        threshold=float(threshold),
        labeled_rows=len(y),
        baseline_source=baseline_source,
        unavailable_reason=reason,
        approval_rate=float(np.mean(scores < threshold)) if len(scores) else None,
        default_rate=float(y.mean()) if len(y) else None,
    )
    if len(scores):
        result.score_p50, result.score_p90, result.score_p99 = map(
            float, np.quantile(scores, [0.5, 0.9, 0.99])
        )
    if len(y):
        predictions = labeled_scores >= threshold
        result.brier = float(brier_score_loss(y, labeled_scores))
        result.precision = float(precision_score(y, predictions, zero_division=0))
        result.recall = float(recall_score(y, predictions, zero_division=0))
    if ref_reason is None:
        result.reference_roc_auc = float(roc_auc_score(ref_y, ref_scores))
    if reason is None:
        result.roc_auc = float(roc_auc_score(y, labeled_scores))
        result.pr_auc = float(average_precision_score(y, labeled_scores))
        result.auc_ci_low, result.auc_ci_high = _auc_interval(y, labeled_scores, params)
    if result.roc_auc is not None and result.reference_roc_auc is not None:
        result.auc_drop = result.reference_roc_auc - result.roc_auc
        result.performance_drop_detected = result.auc_drop > params.auc_drop
        result.auc_drop_ci_low, result.auc_drop_ci_high = auc_difference_interval(
            ref_y, ref_scores, y, labeled_scores, params
        )
    else:
        result.comparison_unavailable_reason = reason or f"control_{ref_reason}"
    return result


def auc_difference_interval(
    before_y: np.ndarray,
    before_scores: np.ndarray,
    after_y: np.ndarray,
    after_scores: np.ndarray,
    params: PerformanceParams,
    paired: bool = False,
) -> tuple[float | None, float | None]:
    """IC percentil 95% da queda de AUC; pares mantêm o mesmo índice sorteado.

    Estratifica por before_y. No caso independente, estratifica cada amostra
    separadamente. O chamador filtra rótulos conhecidos e garante pareamento.
    O intervalo é condicional às amostras observadas e ao procedimento simulado.
    """
    if (
        not params.bootstrap_samples
        or len(np.unique(before_y)) != 2
        or len(np.unique(after_y)) != 2
    ):
        return None, None
    if paired and len(before_y) != len(after_y):
        raise ValueError("Amostras pareadas devem ter o mesmo tamanho")
    rng = np.random.default_rng(params.seed)
    before_groups = [np.flatnonzero(before_y == v) for v in (0, 1)]
    after_groups = [np.flatnonzero(after_y == v) for v in (0, 1)]
    values = []
    for _ in range(params.bootstrap_samples):
        idx_before = np.concatenate(
            [rng.choice(g, len(g), replace=True) for g in before_groups]
        )
        idx_after = (
            idx_before
            if paired
            else np.concatenate(
                [rng.choice(g, len(g), replace=True) for g in after_groups]
            )
        )
        if len(np.unique(after_y[idx_after])) != 2:
            continue
        values.append(
            float(
                roc_auc_score(before_y[idx_before], before_scores[idx_before])
                - roc_auc_score(after_y[idx_after], after_scores[idx_after])
            )
        )
    if len(values) < 0.9 * params.bootstrap_samples:
        return None, None
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)
