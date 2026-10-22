"""Consolidação do experimento: tabelas, gráficos estáticos e diagnóstico legível."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from credit_scoring_observability.drift_report import write_json


def _value(value) -> str:
    if value is None:
        return "—"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _plots(rows: list[dict], output: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    if not rows:
        return []
    x = np.arange(len(rows))
    labels = [r["batch_id"] for r in rows]

    def values(key):
        return np.array(
            [r.get(key) if r.get(key) is not None else np.nan for r in rows],
            dtype=float,
        )

    with plt.rc_context(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    ):
        fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
        ax.plot(
            x,
            values("original_roc_auc"),
            "o--",
            color="#64748b",
            label="Original do mesmo mês",
        )
        ax.plot(x, values("roc_auc"), "o-", color="#1565c0", label="Após intervenção")
        ax.vlines(
            x,
            values("auc_ci_low"),
            values("auc_ci_high"),
            color="#1565c0",
            alpha=0.5,
            linewidth=3,
            label="IC 95% AUC atual",
        )
        baseline = next(
            (
                r["reference_roc_auc"]
                for r in rows
                if r.get("reference_roc_auc") is not None
            ),
            None,
        )
        if baseline is not None:
            ax.axhline(
                baseline,
                color="#374151",
                linestyle=":",
                label="Controle fora do treino",
            )
        ax.set(
            title="Capacidade de ranking ao longo da simulação",
            ylabel="ROC-AUC",
            xlabel="Mês simulado",
            xticks=x,
            xticklabels=labels,
            ylim=(0, 1),
        )
        ax.legend(loc="lower left")
        ax.grid(axis="y", alpha=0.2)
        fig.savefig(output / "auc_timeline.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
        ax.plot(
            x - 0.06,
            values("auc_drop"),
            "o-",
            color="#b45309",
            label="Queda vs. controle independente",
        )
        ax.vlines(
            x - 0.06,
            values("auc_drop_ci_low"),
            values("auc_drop_ci_high"),
            color="#b45309",
            alpha=0.5,
            linewidth=3,
        )
        ax.plot(
            x + 0.06,
            values("paired_auc_drop"),
            "s-",
            color="#1d4ed8",
            label="Queda pareada: original - alterado",
        )
        ax.vlines(
            x + 0.06,
            values("paired_auc_drop_ci_low"),
            values("paired_auc_drop_ci_high"),
            color="#1d4ed8",
            alpha=0.5,
            linewidth=3,
        )
        ax.axhline(0, color="#64748b", linewidth=1)
        ax.set(
            title="Diferença amostral e efeito da intervenção — IC 95%",
            ylabel="Queda de AUC",
            xlabel="Mês simulado",
            xticks=x,
            xticklabels=labels,
        )
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
        fig.savefig(output / "auc_effects.png", dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(
            2, 1, figsize=(10, 7), layout="constrained", sharex=True
        )
        nfeatures = values("analyzed_features")
        axes[0].bar(
            x - 0.18,
            values("original_drift_share") * nfeatures,
            width=0.36,
            color="#94a3b8",
            label="Antes",
        )
        axes[0].bar(
            x + 0.18,
            values("drift_share") * nfeatures,
            width=0.36,
            color="#0f766e",
            label="Depois",
        )
        axes[0].set(
            title="Distribuições de entrada antes/depois",
            ylabel="Features com drift confirmado",
        )
        axes[0].legend()
        axes[1].plot(x, values("score_psi"), "o-", color="#7c3aed")
        axes[1].set(
            ylabel="PSI do score vs. referência",
            xlabel="Mês simulado",
            xticks=x,
            xticklabels=labels,
        )
        for ax in axes:
            ax.grid(axis="y", alpha=0.2)
        fig.savefig(output / "drift_timeline.png", dpi=160)
        plt.close(fig)
    return ["auc_timeline.png", "auc_effects.png", "drift_timeline.png"]


def write_experiment(outcomes: list[dict], reports_dir: Path, context: dict) -> None:
    """Não recalcula inferência nem altera decisões: apresenta os resultados reais."""
    output = reports_dir / "experiment"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("auc_timeline.png", "auc_effects.png", "drift_timeline.png"):
        (output / name).unlink(missing_ok=True)
    successful = [r for r in outcomes if r["status"] == "success"]
    features = []
    for row in successful:
        path = reports_dir / "intervention" / f"{row['batch_id']}.json"
        if path.exists():
            comparison = json.loads(path.read_text(encoding="utf-8"))
            features.extend(
                {"batch_id": row["batch_id"], **f} for f in comparison["features"]
            )
    pd.DataFrame(outcomes).to_csv(output / "monthly.csv", index=False)
    pd.DataFrame(
        features,
        columns=list(features[0])
        if features
        else ["batch_id", "feature", "original_psi", "current_psi"],
    ).to_csv(output / "features.csv", index=False)
    groups = {
        name: [r for r in successful if r.get("scenario") == name]
        for name in ("stable", "data_drift", "concept_drift")
    }
    checks = []
    for scenario, group in groups.items():
        if scenario == "stable":
            passed = all(
                r.get("drift_share") == 0
                and r.get("performance_drop_detected") is False
                for r in group
            )
        elif scenario == "data_drift":
            passed = all(
                len(r.get("drifted_features", "").split(",")) >= 2
                and bool(r.get("drifted_features"))
                for r in group
            )
        else:
            passed = all(
                r.get("drift_share") == 0 and r.get("performance_drop_detected") is True
                for r in group
            )
        checks.append(
            {
                "scenario": scenario,
                "observed_batches": len(group),
                "expected_pattern_observed": passed if group else None,
            }
        )
    plots = _plots(successful, output)
    write_json(
        output / "summary.json",
        {
            "context": context,
            "batches": outcomes,
            "scenario_checks": checks,
            "plots": plots,
        },
    )
    header = "| Mês | Cenário | AUC original | AUC atual | Queda vs. controle | Queda pareada | Drift share | Gate | Recomendação |\n|---|---|---:|---:|---:|---:|---:|---|---|"
    table = "\n".join(
        "| "
        + " | ".join(
            _value(r.get(k))
            for k in (
                "batch_id",
                "scenario",
                "original_roc_auc",
                "roc_auc",
                "auc_drop",
                "paired_auc_drop",
                "drift_share",
                "gate_status",
                "recommendation",
            )
        )
        + " |"
        for r in successful
    )
    scenario_text = "\n".join(
        f"- **{c['scenario']}**: {c['observed_batches']} lotes; padrão esperado observado: {_value(c['expected_pattern_observed'])}."
        for c in checks
    )
    failed = (
        "\n".join(
            f"- {r['batch_id']}: {r['status']}."
            for r in outcomes
            if r["status"] != "success"
        )
        or "Nenhum lote bloqueado/falho nesta seleção."
    )
    images = "\n\n".join(
        f"![{name.replace('_', ' ').removesuffix('.png')}]({name})" for name in plots
    )
    text = f"""# Experimento consolidado de monitoramento

Esta é a seleção da última execução de `make analyze`. Modelo congelado e
controle fora do treino. Os meses representam uma simulação, não produção real.

{header}
{table}

## Verificação dos cenários

{scenario_text}

Meses estáveis verificam alertas inesperados nesta amostra; dois meses não
estimam uma taxa geral de falsos positivos. Alterações nas features podem mudar
aprovação e Brier mesmo sem queda relevante de AUC. Relabel pode degradar a AUC
sem mudar X ou os scores dos mesmos registros.

## Antes e depois

As versões originais e alteradas têm os mesmos ids e ordem. A comparação contra
referência identifica diferenças já existentes; a comparação pareada isola a
intervenção nessa amostra. PSI original e alterado não são efeitos aditivos.
Intervalos da queda vs. controle usam bootstrap independente; os da queda
pareada mantêm os pares e estratificam pelo rótulo original. ICs são percentis
de 95% condicionais aos dados e ao procedimento, não prova de causa econômica.

{images}

## Decisões e limites

O gate usa métricas, nunca o nome do cenário simulado. Recomenda avaliar
retreinamento somente com rótulos suficientes e IC da queda acima do limite
operacional. Diagnósticos são suspeitas; nenhum modelo é treinado ou promovido.
Os detalhes e a justificativa estão nos HTMLs de cada mês e JSONs em `reports/gate/`.
Tabelas detalhadas: [meses](monthly.csv), [features antes/depois](features.csv) e
[proveniência e resultados](summary.json).

## Lotes não analisados

{failed}
"""
    (output / "summary.md").write_text(text, encoding="utf-8")
