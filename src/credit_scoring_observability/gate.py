"""Recomendação operacional baseada em evidências, sem treinar/promover modelos."""

from __future__ import annotations

from credit_scoring_observability.parameters import (
    DriftParams,
    GateParams,
    PerformanceParams,
)
from credit_scoring_observability.results import (
    DriftResult,
    GateResult,
    PerformanceResult,
)


def evaluate_gate(
    drift: DriftResult,
    perf: PerformanceResult,
    params: GateParams | None = None,
    performance_params: PerformanceParams | None = None,
    drift_params: DriftParams | None = None,
) -> GateResult:
    """Retreino recomendado somente com queda realizada e IC inteiro acima do limite.

    Diagnósticos são suspeitas operacionais. Ausência de drift marginal não
    exclui mudanças conjuntas de X, problemas de rótulos ou outras explicações.
    O gate não recebe o nome do cenário simulado para decidir a recomendação.
    """
    params, pp, dp = (
        params or GateParams(),
        performance_params or PerformanceParams(),
        drift_params or DriftParams(),
    )
    drift.to_dict()
    perf.to_dict()
    limitations = [
        "A recomendação exige análise humana; nenhum modelo é retreinado ou promovido por este gate."
    ]
    reasons = [
        f"{len(drift.drifted_features)} de {drift.analyzed_features} features analisáveis com drift confirmado."
    ]
    coverage = perf.labeled_rows / perf.rows if perf.rows else 0.0
    observed = (
        perf.roc_auc is not None
        and perf.reference_roc_auc is not None
        and perf.auc_drop is not None
    )
    control_coverage = (
        perf.control_labeled_rows / perf.control_rows if perf.control_rows else 0.0
    )
    adequate = (
        observed
        and min(perf.labeled_rows, perf.control_labeled_rows) >= params.min_labeled_rows
        and min(coverage, control_coverage) >= params.min_label_coverage
    )
    if not adequate:
        reasons.append(
            f"Evidência supervisionada insuficiente: lote com {perf.labeled_rows} rótulos ({coverage:.1%}); controle com {perf.control_labeled_rows} ({control_coverage:.1%})."
        )
        if perf.unavailable_reason or perf.comparison_unavailable_reason:
            limitations.append(
                perf.unavailable_reason or perf.comparison_unavailable_reason
            )
        return GateResult(
            drift.batch_id,
            "WARN",
            "insufficient_evidence",
            "investigate",
            "Investigar qualidade e disponibilidade dos rótulos",
            reasons,
            limitations,
        )
    reasons.append(
        f"AUC controle {perf.reference_roc_auc:.4f}; lote {perf.roc_auc:.4f}; queda {perf.auc_drop:+.4f} (limite {pp.auc_drop:.4f})."
    )
    if perf.auc_drop > pp.auc_drop:
        confirmed = (
            perf.auc_drop_ci_low is not None
            and perf.auc_drop_ci_high is not None
            and pp.bootstrap_samples >= params.min_bootstrap_samples
            and pp.auc_drop < perf.auc_drop_ci_low <= perf.auc_drop_ci_high
        )
        if confirmed:
            reasons.append(
                f"IC 95% da queda [{perf.auc_drop_ci_low:.4f}, {perf.auc_drop_ci_high:.4f}] acima do limite."
            )
        else:
            limitations.append(
                "Queda observada acima do limite, mas falta confirmação pelo intervalo ou pelo mínimo de reamostragens configurado."
            )
        diagnosis = "mixed_suspected" if drift.drifted_features else "concept_suspected"
        if drift.unavailable_features or not drift.analyzed_features:
            diagnosis = "undetermined"
        limitations.append(
            "Queda de AUC e estabilidade marginal de X não provam concept drift ou causa econômica."
        )
        return GateResult(
            drift.batch_id,
            "ALERT",
            diagnosis,
            "retrain" if confirmed else "investigate",
            "Investigar degradação e avaliar retreinamento"
            if confirmed
            else "Investigar degradação antes de decidir sobre retreinamento",
            reasons,
            limitations,
            confirmed,
        )
    if drift.unavailable_features or not drift.analyzed_features:
        reasons.append("Há distribuições de features indisponíveis.")
        return GateResult(
            drift.batch_id,
            "WARN",
            "insufficient_evidence",
            "investigate",
            "Investigar a cobertura das features",
            reasons,
            limitations,
        )
    if drift.drifted_features:
        reasons.append(
            "Mudanças de entrada sem queda de AUC acima do limite; acompanhar Brier e aprovação."
        )
        return GateResult(
            drift.batch_id,
            "WARN",
            "covariate_suspected",
            "reinforced_monitoring",
            "Reforçar monitoramento e investigar mudanças nas entradas",
            reasons,
            limitations,
        )
    if perf.auc_drop_ci_high is not None and perf.auc_drop_ci_high > pp.auc_drop:
        reasons.append("O intervalo da diferença ainda permite queda acima do limite.")
        return GateResult(
            drift.batch_id,
            "WARN",
            "uncertain_performance",
            "investigate",
            "Investigar a incerteza da performance",
            reasons,
            limitations,
        )
    if drift.label_z is not None and abs(drift.label_z) > dp.label_z:
        reasons.append(f"Mudança de prevalência: z = {drift.label_z:.2f}.")
        return GateResult(
            drift.batch_id,
            "WARN",
            "prior_suspected",
            "investigate",
            "Investigar mudança na prevalência dos rótulos",
            reasons,
            limitations,
        )
    if any(f.status != "OK" or f.missing_drift for f in drift.features) or (
        drift.score_psi is not None and drift.score_psi >= dp.psi_warning
    ):
        reasons.append("Há sinais de atenção em distribuições, ausências ou score.")
        return GateResult(
            drift.batch_id,
            "WARN",
            "distribution_warning",
            "reinforced_monitoring",
            "Reforçar monitoramento dos sinais de atenção",
            reasons,
            limitations,
        )
    reasons.append("Nenhum sinal excedeu os limites operacionais avaliados.")
    if perf.auc_drop_ci_low is None:
        limitations.append(
            "Intervalo da queda indisponível; nenhuma recomendação de retreino é emitida."
        )
    return GateResult(
        drift.batch_id,
        "OK",
        "none",
        "observe",
        "Continuar monitoramento regular",
        reasons,
        limitations,
    )
