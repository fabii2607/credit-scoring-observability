"""Detecção de mudanças de distribuição; não comprova concept drift.

Funções puras: relatórios Evidently e persistência ficam em drift_report.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from credit_scoring_observability.config import SCORE_COLUMN, TARGET
from credit_scoring_observability.data_contract import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
)
from credit_scoring_observability.parameters import DriftParams
from credit_scoring_observability.results import DriftResult, FeatureDriftResult
from credit_scoring_observability.stats import compare_numeric, psi_quantile


def categorical_js(reference: pd.Series, current: pd.Series) -> float | None:
    """Distância Jensen–Shannon (log natural), alinhada pela união de categorias.

    Ausências ficam fora da distribuição e são monitoradas separadamente.
    Não há epsilon: a definição de JS trata probabilidades zero.
    """
    ref, cur = reference.dropna(), current.dropna()
    if ref.empty or cur.empty:
        return None
    p, q = ref.value_counts(normalize=True), cur.value_counts(normalize=True)
    categories = p.index.union(q.index, sort=False)
    return float(
        jensenshannon(
            p.reindex(categories, fill_value=0), q.reindex(categories, fill_value=0)
        )
    )


def _categorical(
    feature: str, ref: pd.Series, cur: pd.Series, params: DriftParams
) -> FeatureDriftResult:
    distance = categorical_js(ref, cur)
    ref_missing = float(ref.isna().mean()) if len(ref) else None
    cur_missing = float(cur.isna().mean()) if len(cur) else None
    delta = (
        cur_missing - ref_missing
        if ref_missing is not None and cur_missing is not None
        else None
    )
    drifted = distance is not None and distance >= params.categorical_js
    return FeatureDriftResult(
        feature=feature,
        method="jensenshannon_natural_log",
        categorical_distance=distance,
        drifted=drifted,
        status="UNAVAILABLE" if distance is None else "ALERT" if drifted else "OK",
        reference_missing_rate=ref_missing,
        current_missing_rate=cur_missing,
        missing_rate_delta=delta,
        missing_drift=abs(delta) >= params.missing_rate_delta
        if delta is not None
        else None,
        unavailable_reason="empty_or_all_missing_sample" if distance is None else None,
    )


def _labels(frame: pd.DataFrame) -> np.ndarray:
    if TARGET not in frame:
        return np.array([], dtype=float)
    y = frame[TARGET].dropna()
    if not y.isin([0, 1]).all():
        raise ValueError("Target deve ser binário quando observado")
    return y.to_numpy(dtype=float)


def detect_drift(
    reference_scored: pd.DataFrame,
    batch_scored: pd.DataFrame,
    batch_id: str,
    params: DriftParams | None = None,
) -> DriftResult:
    """Compara somente as features; score e prevalência têm campos separados.

    drift_share: numéricas com PSI ALERT e KS confirmado + categóricas com JS
    acima do limite, dividido pelas features analisáveis. Ausências não entram
    nessa fração. PSI WARN e alertas de ausência permanecem nos detalhes.
    """
    params = params or DriftParams()
    required = {*MODEL_FEATURES, SCORE_COLUMN}
    if required - set(reference_scored) or required - set(batch_scored):
        raise ValueError("Referência e lote precisam das features e do score")
    features = []
    for feature in MODEL_FEATURES:
        ref, cur = reference_scored[feature], batch_scored[feature]
        if feature in CATEGORICAL_FEATURES:
            item = _categorical(feature, ref, cur, params)
        else:
            stats = compare_numeric(ref, cur, params)
            item = FeatureDriftResult(
                feature=feature,
                psi=stats.psi,
                ks_d=stats.ks_d,
                ks_p_value=stats.ks_p_value,
                status=stats.status,
                ks_confirmed=stats.ks_confirmed,
                drifted=stats.critical,
                reference_missing_rate=stats.reference_missing_rate,
                current_missing_rate=stats.current_missing_rate,
                missing_rate_delta=stats.missing_rate_delta,
                missing_drift=stats.missing_drift,
                unavailable_reason=stats.unavailable_reason,
            )
        features.append(item)
    unavailable = [f.feature for f in features if f.status == "UNAVAILABLE"]
    analyzed = len(features) - len(unavailable)
    drifted = [f.feature for f in features if f.drifted]
    ref_y, cur_y = _labels(reference_scored), _labels(batch_scored)
    label_z = None
    if len(ref_y) and len(cur_y):
        pooled = float((ref_y.sum() + cur_y.sum()) / (len(ref_y) + len(cur_y)))
        se = np.sqrt(pooled * (1 - pooled) * (1 / len(ref_y) + 1 / len(cur_y)))
        label_z = float((cur_y.mean() - ref_y.mean()) / se) if se else 0.0
    return DriftResult(
        batch_id=batch_id,
        features=features,
        analyzed_features=analyzed,
        unavailable_features=unavailable,
        drift_share=len(drifted) / analyzed if analyzed else None,
        drifted_features=drifted,
        score_psi=psi_quantile(
            reference_scored[SCORE_COLUMN],
            batch_scored[SCORE_COLUMN],
            params.bins,
            params.epsilon,
        ),
        default_rate=float(cur_y.mean()) if len(cur_y) else None,
        label_z=label_z,
    )
