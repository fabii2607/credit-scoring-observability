"""Contratos de saída da Etapa 2, sem I/O ou dependência de dashboards.

Métricas não calculáveis usam None (null no JSON). NaN/Infinity são rejeitados
na serialização, em vez de gerar JSON inválido ou esconder erros de cálculo.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Literal

DriftStatus = Literal["OK", "WARN", "ALERT", "UNAVAILABLE"]


@dataclass
class JsonResult:
    """Conversão estrita para artefatos JSON agregados."""

    def to_dict(self) -> dict:
        result = asdict(self)
        json.dumps(result, allow_nan=False)
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)


@dataclass
class FeatureDriftResult(JsonResult):
    feature: str
    status: DriftStatus = "UNAVAILABLE"
    psi: float | None = None
    ks_d: float | None = None
    ks_p_value: float | None = None
    ks_confirmed: bool = False
    drifted: bool = False
    reference_missing_rate: float | None = None
    current_missing_rate: float | None = None
    missing_rate_delta: float | None = None
    missing_drift: bool | None = None
    method: str = "psi_quantile_ks"
    categorical_distance: float | None = None
    unavailable_reason: str | None = None


@dataclass
class DriftResult(JsonResult):
    batch_id: str
    features: list[FeatureDriftResult] = field(default_factory=list)
    # Denominador explícito: features analisáveis, sem score/target/metadados.
    analyzed_features: int = 0
    unavailable_features: list[str] = field(default_factory=list)
    drift_share: float | None = None
    drifted_features: list[str] = field(default_factory=list)
    score_psi: float | None = None
    default_rate: float | None = None
    label_z: float | None = None
    report_paths: dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceResult(JsonResult):
    """AUC-base deve vir de controle fora do treino, não da referência de drift.

    Sem rótulo ou sem duas classes, roc_auc fica None e unavailable_reason explica
    o motivo. O avaliador está em performance.evaluate_performance.
    """

    roc_auc: float | None = None
    reference_roc_auc: float | None = None
    auc_drop: float | None = None
    pr_auc: float | None = None
    brier: float | None = None
    approval_rate: float | None = None
    precision: float | None = None
    recall: float | None = None
    score_p50: float | None = None
    score_p90: float | None = None
    score_p99: float | None = None
    auc_ci_low: float | None = None
    auc_ci_high: float | None = None
    labeled_rows: int = 0
    baseline_source: str | None = None
    unavailable_reason: str | None = None
    comparison_unavailable_reason: str | None = None
    rows: int = 0
    threshold: float | None = None
    default_rate: float | None = None
    performance_drop_detected: bool | None = None
    auc_drop_ci_low: float | None = None
    auc_drop_ci_high: float | None = None
    control_rows: int = 0
    control_labeled_rows: int = 0


@dataclass
class GateResult(JsonResult):
    batch_id: str
    status: Literal["OK", "WARN", "ALERT"]
    diagnosis: str
    recommended_action: str
    recommendation: str
    reasons: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    retrain_recommended: bool = False


@dataclass
class InterventionResult(JsonResult):
    batch_id: str
    scenario: str
    original_performance: dict = field(default_factory=dict)
    original_drifted_features: list[str] = field(default_factory=list)
    current_drifted_features: list[str] = field(default_factory=list)
    original_drift_share: float | None = None
    current_drift_share: float | None = None
    intervention_drift_share: float | None = None
    intervention_score_psi: float | None = None
    paired_auc_drop: float | None = None
    paired_auc_drop_ci_low: float | None = None
    paired_auc_drop_ci_high: float | None = None
    paired_rows: int = 0
    unavailable_reason: str | None = None
    features: list[dict] = field(default_factory=list)
