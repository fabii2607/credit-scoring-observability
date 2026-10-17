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
from credit_scoring_observability.results import DriftResult
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
    <h1>Monitoramento de crédito — {escape(result.batch_id)}</h1>
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
) -> dict[str, str]:
    """Filtra metadados ANTES de criar Dataset; categoriza features explicitamente.

    Exclui colunas não analisáveis dos gráficos, preservando-as no resumo JSON/HTML.
    O snapshot não é versionado automaticamente e não contém target/prediction.
    """
    params, context = params or DriftParams(), context or {}
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
                lambda m: m.group(1) + _summary(result, params, context),
                html,
                count=1,
                flags=re.IGNORECASE,
            )
        else:
            html = (
                "<!doctype html><html lang='pt-BR'><meta charset='utf-8'><title>Drift "
                + escape(batch_id)
                + "</title><body>"
                + _summary(result, params, context)
                + html
                + "</body></html>"
            )
    else:
        html = (
            "<!doctype html><html lang='pt-BR'><meta charset='utf-8'><body>"
            + _summary(result, params, context)
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
        },
    )
    return paths
