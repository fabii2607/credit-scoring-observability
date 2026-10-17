"""HTML Evidently com PSI/KS do projeto e apenas colunas de distribuição.

Integração testada com Evidently 0.7.23. Não recebe identificadores no relatório
Evidently; não inclui linhas individuais na tabela de resumo do projeto.
"""

from __future__ import annotations

import hashlib
import json
import re
from html import escape
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.legacy.calculations.stattests.registry import StatTest, register_stattest
from evidently.legacy.core import ColumnType
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset

from credit_scoring_observability.config import SCORE_COLUMN
from credit_scoring_observability.data_contract import CATEGORICAL_FEATURES
from credit_scoring_observability.drift import categorical_js
from credit_scoring_observability.parameters import DriftParams
from credit_scoring_observability.results import (
    DriftResult,
    GateResult,
    InterventionResult,
    PerformanceResult,
)
from credit_scoring_observability.stats import compare_numeric


def safe_batch_id(batch_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", batch_id):
        raise ValueError("Identificador de lote inválido")
    return batch_id


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def register_methods(params: DriftParams) -> tuple[str, str]:
    """Registra métodos por configuração para evitar colisões entre execuções."""
    suffix = hashlib.sha256(params.model_dump_json().encode()).hexdigest()[:12]
    numeric, categorical = f"credit_quantile_psi_ks_{suffix}", f"credit_js_{suffix}"

    def numeric_test(ref, cur, feature_type, threshold):
        result = compare_numeric(ref, cur, params)
        if result.psi is None:
            raise ValueError(
                "Feature indisponível não deve entrar no relatório Evidently"
            )
        return result.psi, result.psi > threshold and result.ks_confirmed

    def categorical_test(ref, cur, feature_type, threshold):
        value = categorical_js(ref, cur)
        if value is None:
            raise ValueError("Feature categórica indisponível")
        return value, value >= threshold

    register_stattest(
        StatTest(
            numeric,
            "PSI por quantis + confirmação KS D",
            [ColumnType.Numerical],
            params.psi_alert,
        ),
        numeric_test,
    )
    register_stattest(
        StatTest(
            categorical,
            "Jensen-Shannon (log natural)",
            [ColumnType.Categorical],
            params.categorical_js,
        ),
        categorical_test,
    )
    return numeric, categorical


def _format(value: float | None) -> str:
    return "Indisponível" if value is None else f"{value:.6g}"


def _recommendation(gate: GateResult | None) -> str:
    if gate is None:
        return ""
    color, background, label = {
        "OK": ("#166534", "#f0fdf4", "OK"),
        "WARN": ("#92400e", "#fffbeb", "ATENÇÃO"),
        "ALERT": ("#991b1b", "#fef2f2", "ALERTA"),
    }[gate.status]
    reasons = "".join(f"<li>{escape(reason)}</li>" for reason in gate.reasons)
    limitations = "".join(f"<li>{escape(reason)}</li>" for reason in gate.limitations)
    return f"""<section id="credit-gate" data-gate-status="{gate.status}" style="font-family:system-ui,sans-serif;margin:20px auto;padding:28px;max-width:1400px;background:{background};color:#172b42;border:2px solid {color};border-left:12px solid {color};border-radius:12px">
    <h1 style="margin-top:0">Monitoramento de crédito — {escape(gate.batch_id)}</h1>
    <strong style="display:inline-block;padding:6px 14px;border-radius:6px;color:white;background:{color}">{label}</strong>
    <h2 style="font-size:28px;color:{color}">{escape(gate.recommendation)}</h2>
    <h3>Por que esta recomendação?</h3><ul>{reasons}</ul>
    <details><summary>Condições e limites da recomendação</summary><ul>{limitations}</ul></details>
    <p>A seguir: evidências de performance, comparação antes/depois, estatísticas e gráficos de distribuição.</p></section>"""


def _performance_summary(
    perf: PerformanceResult | None, comparison: InterventionResult | None
) -> str:
    if perf is None:
        return ""
    values = [
        ("AUC do controle", perf.reference_roc_auc),
        ("AUC do mês", perf.roc_auc),
        ("Queda AUC vs. controle", perf.auc_drop),
        ("Taxa de aprovação", perf.approval_rate),
    ]
    cards = "".join(
        f'<div style="flex:1;min-width:190px;padding:16px;background:white;border:1px solid #cbd5e1;border-radius:8px"><span>{label}</span><br><strong style="font-size:26px">{_format(value)}</strong></div>'
        for label, value in values
    )
    paired = ""
    if comparison is not None:
        paired = f"""<h3>Os mesmos registros, antes e depois da intervenção</h3>
        <p>AUC original: {_format(comparison.original_performance.get("roc_auc"))}.
        Queda pareada: {_format(comparison.paired_auc_drop)};
        IC 95% [{_format(comparison.paired_auc_drop_ci_low)}, {_format(comparison.paired_auc_drop_ci_high)}].
        Features com drift antes: {len(comparison.original_drifted_features)}; depois: {len(comparison.current_drifted_features)}.</p>
        <p>PSI original/referência e PSI alterado/referência são comparações separadas, não efeitos aditivos.
        A comparação pareada mede a intervenção sintética nesta amostra e não prova causa econômica.</p>"""
    return f"""<section id="credit-performance" style="font-family:system-ui,sans-serif;max-width:1400px;padding:24px;margin:16px auto;background:#f4f7fb;color:#172b42;border-radius:12px">
    <h2>Evidências de performance</h2><div style="display:flex;gap:12px;flex-wrap:wrap">{cards}</div>
    <p>IC 95% da queda contra controle: [{_format(perf.auc_drop_ci_low)}, {_format(perf.auc_drop_ci_high)}].
    Rótulos conhecidos: {perf.labeled_rows} / {perf.rows}. Threshold de decisão: {_format(perf.threshold)}.
    Brier: {_format(perf.brier)}. AUC usa scores contínuos, independentemente do threshold.</p>{paired}</section>"""


def _summary(result: DriftResult, params: DriftParams, context: dict) -> str:
    rows = "".join(
        f"<tr><td>{escape(item.feature)}</td><td>{escape(item.method)}</td>"
        f"<td>{_format(item.psi)}</td><td>{_format(item.ks_d)}</td><td>{_format(item.ks_p_value)}</td>"
        f"<td>{_format(item.categorical_distance)}</td><td>{item.status}</td>"
        f"<td>{'Sim' if item.drifted else 'Não'}</td><td>{_format(item.missing_rate_delta)}</td>"
        f"<td>{'Indisponível' if item.missing_drift is None else 'Sim' if item.missing_drift else 'Não'}</td></tr>"
        for item in result.features
    )
    return f"""<section id="credit-summary" style="font-family:system-ui,sans-serif;padding:24px;margin:16px auto;max-width:1400px;color:#172b42;background:#f4f7fb;border:1px solid #cbd5e1;border-radius:12px">
    <h2>Distribuições de entrada — {escape(result.batch_id)}</h2>
    <p>Comparação da referência de distribuição com o lote pontuado. Modelo: {escape(str(context.get("model_version", "não informado")))}.</p>
    <p><strong>Features com drift confirmado: {len(result.drifted_features)} / {result.analyzed_features}</strong>.
    Proporção: {_format(result.drift_share)}. Score PSI: {_format(result.score_psi)}.</p>
    <p>Numéricas: PSI &gt; {params.psi_alert} e KS D &gt; {params.ks_d}. WARN: PSI de {params.psi_warning} até {params.psi_alert}.
    Categóricas: Jensen–Shannon ≥ {params.categorical_js}. Ausências são avaliadas separadamente (|delta| ≥ {params.missing_rate_delta}).</p>
    <p>O resumo Dataset Drift do Evidently sinaliza pelo menos uma feature confirmada. Score é analisado à parte e não entra no denominador.
    PSI alto sem confirmação KS aparece em ALERT na faixa PSI, mas não como drift confirmado.
    Mudança de distribuição ou de prevalência não comprova concept drift; consulte a avaliação de performance contra o controle fora do treino.</p>
    <div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%;text-align:left;font-size:13px" border="1" cellpadding="7">
    <thead><tr><th>Feature</th><th>Método</th><th>PSI</th><th>KS D</th><th>KS p-valor</th><th>JS</th><th>Faixa</th><th>Drift confirmado</th><th>Delta nulos</th><th>Alerta nulos</th></tr></thead><tbody>{rows}</tbody></table></div>
    <p>Distribuições e histogramas abaixo são produzidos pelo Evidently {escape(version("evidently"))}; estatísticas numéricas usam os mesmos cálculos do JSON.
    Indisponibilidade é explícita. Relatório agregado, sem identificadores de clientes. Gráficos de distribuição não estabelecem causalidade.</p></section>"""


def write_drift_report(
    reference_scored: pd.DataFrame,
    batch_scored: pd.DataFrame,
    result: DriftResult,
    output_dir: Path,
    params: DriftParams | None = None,
    context: dict | None = None,
    gate: GateResult | None = None,
    performance: PerformanceResult | None = None,
    comparison: InterventionResult | None = None,
) -> dict[str, str]:
    """Filtra metadados ANTES de criar Dataset; categoriza features explicitamente.

    Exclui colunas não analisáveis dos gráficos, preservando-as no resumo JSON/HTML.
    O snapshot não é versionado automaticamente e não contém target/prediction.
    """
    params, context = params or DriftParams(), context or {}
    intro = (
        _recommendation(gate)
        + _performance_summary(performance, comparison)
        + _summary(result, params, context)
    )
    batch_id = safe_batch_id(result.batch_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path, json_path = (
        output_dir / f"{batch_id}.html",
        output_dir / f"{batch_id}.json",
    )
    names = [f.feature for f in result.features if f.status != "UNAVAILABLE"]
    cats = [c for c in names if c in CATEGORICAL_FEATURES]
    nums = [c for c in names if c not in cats]
    score_available = result.score_psi is not None
    columns = [*names, *([SCORE_COLUMN] if score_available else [])]
    numeric, categorical = register_methods(params)
    metrics = []
    if names:
        metrics.append(
            DataDriftPreset(
                columns=names,
                num_method=numeric,
                cat_method=categorical,
                num_threshold=params.psi_alert,
                cat_threshold=params.categorical_js,
                drift_share=1 / len(names),
            )
        )
    if score_available:
        metrics.append(
            ValueDrift(column=SCORE_COLUMN, method=numeric, threshold=params.psi_alert)
        )
    if metrics:
        definition = DataDefinition(
            numerical_columns=[*nums, *([SCORE_COLUMN] if score_available else [])],
            categorical_columns=cats,
        )
        ref = reference_scored.loc[:, columns].copy().reset_index(drop=True)
        cur = batch_scored.loc[:, columns].copy().reset_index(drop=True)
        for frame in (ref, cur):
            frame[definition.numerical_columns] = frame[
                definition.numerical_columns
            ].astype(float)
            for col in cats:
                frame[col] = frame[col].astype("object")
        snapshot = Report(metrics).run(
            current_data=Dataset.from_pandas(cur, data_definition=definition),
            reference_data=Dataset.from_pandas(ref, data_definition=definition),
            name=f"Credit drift {batch_id}",
        )
        snapshot.save_html(str(html_path))
        html = html_path.read_text(encoding="utf-8")
        # Evidently pode emitir um fragmento, não um documento HTML completo.
        if re.search(r"<body[^>]*>", html, flags=re.IGNORECASE):
            html = re.sub(
                r"(<body[^>]*>)",
                lambda m: m.group(1) + intro,
                html,
                count=1,
                flags=re.IGNORECASE,
            )
        else:
            html = (
                "<!doctype html><html lang='pt-BR'><meta charset='utf-8'><title>Drift "
                + escape(batch_id)
                + "</title><body>"
                + intro
                + html
                + "</body></html>"
            )
    else:
        html = (
            "<!doctype html><html lang='pt-BR'><meta charset='utf-8'><body>"
            + intro
            + "<p>Sem distribuições observadas para gerar gráficos Evidently.</p></body></html>"
        )
    html_path.write_text(html, encoding="utf-8")
    # Caminhos relativos ao diretório do relatório; sem nomes locais de usuário.
    paths = {"html": html_path.name, "json": json_path.name}
    result.report_paths = paths
    write_json(
        json_path,
        {
            **result.to_dict(),
            "parameters": params.model_dump(mode="json"),
            "reference_rows": len(reference_scored),
            "current_rows": len(batch_scored),
            "evidently_version": version("evidently"),
            "context": context,
            "gate": gate.to_dict() if gate is not None else None,
            "performance": performance.to_dict() if performance is not None else None,
            "intervention": comparison.to_dict() if comparison is not None else None,
        },
    )
    return paths
