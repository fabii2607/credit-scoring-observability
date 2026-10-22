"""Atualiza a seleção pública e seus hashes a partir de uma execução completa."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import date
from importlib.metadata import distributions
from pathlib import Path

import pandas as pd
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
SELECTED = ["2026-01", "2026-04", "2026-06"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def freeze(execution_date: str):
    date.fromisoformat(execution_date)
    reports, destination = ROOT / "reports", ROOT / "docs/evidencias"
    summary = load(reports / "performance/summary.json")
    causal = load(reports / "causal/causal_analysis.json")
    stack = load(reports / "monitoring/stack-check.json")
    alerts = load(reports / "monitoring/alertmanager-check.json")
    simulation = load(ROOT / "data/production/manifest.json")
    expected = {b["batch_id"] for b in simulation["batches"]}
    regular = [r for r in summary["batches"] if r["batch_id"] in expected]
    if len(regular) != len(expected) or {r["batch_id"] for r in regular} != expected:
        raise ValueError("Consolidado precisa conter todos os meses, sem duplicações")
    if any(r["status"] != "success" for r in regular):
        raise ValueError("Todos os meses precisam ter sucesso")
    blocked = [r for r in summary["batches"] if r["batch_id"] == "corrupted"]
    if len(blocked) != 1 or blocked[0]["status"] != "blocked":
        raise ValueError("Falta evidência do bloqueio da demo")
    required_retrain = {
        r["batch_id"] for r in regular if r.get("recommended_action") == "retrain"
    }
    received_retrain = {
        a["batch_id"]
        for a in alerts
        if a["name"] == "RetrainRecommended" and a["state"] == "active"
    }
    if not required_retrain <= received_retrain:
        raise ValueError(
            "Aguardar recebimento dos alertas de recomendação no Alertmanager"
        )
    if not any(
        a["name"] == "ContractViolation" and a["batch_id"] == "corrupted"
        for a in alerts
    ):
        raise ValueError("Falta confirmação do alerta de contrato")
    context = summary["context"]
    for source in simulation["sources"].values():
        if Path(source["name"]).name != source["name"]:
            raise ValueError("Nome de origem inválido no manifesto")
        if digest(ROOT / "data/processed" / source["name"]) != source["sha256"]:
            raise ValueError("Origem difere do manifesto da simulação")
    if any(
        causal[key] != context[key]
        for key in (
            "control_sha256",
            "reference_sha256",
            "simulation_manifest_sha256",
            "model_sha256",
            "metadata_sha256",
        )
    ):
        raise ValueError(
            "Análise de sensibilidade e monitoramento usam origens diferentes"
        )
    for row in causal["batches"]:
        actual = next(r for r in regular if r["batch_id"] == row["batch_id"])
        if abs(row["drifted"]["roc_auc"] - actual["roc_auc"]) > 0.000051:
            raise ValueError("AUC da sensibilidade diverge do monitoramento")
    if len(causal["batches"]) != len(expected):
        raise ValueError("Sensibilidade incompleta")
    executed = {
        r["batch_id"]: r for r in load(reports / "monitoring/summary.json")["batches"]
    }
    for row in stack["batches"]:
        if row.get("mlflow_run_id") != executed[row["batch_id"]]["mlflow_run_id"]:
            raise ValueError("Verificação da stack não corresponde à execução atual")
    verified = {r["batch_id"] for r in stack["batches"]}
    if verified != expected | {"corrupted"}:
        raise ValueError("Verificação da stack incompleta")
    for metric in (
        "metrics_match",
        "mlflow_verified",
        "trace_delivered",
        "logs_correlated",
    ):
        if not all(r[metric] for r in stack["batches"]):
            raise ValueError("Falha na verificação da stack")

    artifacts = {}

    def copy(relative, source=None, transform=None):
        source = source or reports / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if transform:
            target.write_text(transform(source.read_text()))
        else:
            target.write_bytes(source.read_bytes())
        artifacts[relative] = {
            "source": source.relative_to(ROOT).as_posix(),
            "source_sha256": digest(source),
            "sha256": digest(target),
            "bytes": target.stat().st_size,
        }
        if transform:
            artifacts[relative]["transformation"] = (
                "Remoção de espaços em linhas vazias do HTML."
            )

    for month in SELECTED:
        for folder, extension in [
            ("drift", "html"),
            ("drift", "json"),
            ("performance", "json"),
            ("gate", "json"),
            ("intervention", "json"),
        ]:
            relative = f"{folder}/{month}.{extension}"
            transform = (
                (
                    lambda text: "\n".join(
                        line if line.strip() else "" for line in text.split("\n")
                    )
                )
                if extension == "html"
                else None
            )
            copy(relative, transform=transform)
    for name in ["summary.csv", "summary.json"]:
        copy(f"performance/{name}")
    for source in sorted((reports / "experiment").iterdir()):
        if source.is_file():
            copy(f"experiment/{source.name}", source)
    copy("causal/causal_analysis.json")
    for name in ["stack-check.json", "alertmanager-check.json"]:
        copy(f"observability/{name}", reports / "monitoring" / name)
    contract = load(reports / "contracts/corrupted.json")
    public = {
        key: contract[key]
        for key in ("batch_id", "n_rows", "passed", "n_blockers", "n_warnings")
    }
    public["violations"] = [
        {
            k: v
            for k, v in row.items()
            if k in {"rule", "column", "severity", "count", "layer"}
        }
        for row in contract["violations"]
    ]
    public["scope"] = "Contagens agregadas, sem exemplos individuais."
    target = destination / "contracts/corrupted-summary.json"
    write(target, public)
    artifacts["contracts/corrupted-summary.json"] = {
        "source": "reports/contracts/corrupted.json",
        "source_sha256": digest(reports / "contracts/corrupted.json"),
        "sha256": digest(target),
        "bytes": target.stat().st_size,
        "transformation": "Campos agregados, sem exemplos individuais.",
    }
    run_source = reports / "monitoring/summary.json"
    source_rows = load(run_source)["batches"]
    fields = [
        "batch_id",
        "status",
        "rows",
        "roc_auc",
        "reference_roc_auc",
        "auc_drop",
        "gate_status",
        "recommended_action",
        "telemetry_status",
    ]
    target = destination / "observability/run-summary.json"
    write(
        target,
        {
            "execution_date": execution_date,
            "batches": [{k: r[k] for k in fields if k in r} for r in source_rows],
        },
    )
    artifacts["observability/run-summary.json"] = {
        "source": "reports/monitoring/summary.json",
        "source_sha256": digest(run_source),
        "sha256": digest(target),
        "bytes": target.stat().st_size,
        "transformation": "Somente campos agregados da execução.",
    }
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    sources = [
        p
        for folder in ("src", "scripts", "monitoring")
        for p in (ROOT / folder).rglob("*")
        if p.is_file()
        and p.suffix in {".py", ".yml", ".yaml", ".alloy", ".json"}
        and "__pycache__" not in p.parts
    ]
    sources += [
        ROOT / p
        for p in (
            "params.yaml",
            "pyproject.toml",
            "uv.lock",
            "Makefile",
            "docker-compose.yml",
        )
    ]
    manifest = {
        "execution_date": execution_date,
        "packaged_date": execution_date,
        "timezone": "America/Sao_Paulo",
        "scope": "Simulação acadêmica; não produção real ou validação externa.",
        "selected_months": SELECTED,
        "source_revision": revision,
        "source_state": "Snapshot do working tree; HEAD é a base, fontes efetivas identificadas por SHA-256.",
        "source_sha256": {
            p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(sources)
        },
        "installed_versions": {d.metadata["Name"]: d.version for d in distributions()},
        "context": context,
        "artifacts": artifacts,
        "validation": {
            "regular_batches": len(regular),
            "scored_rows": sum(r["rows"] for r in regular),
            "blocked_demo": True,
            "causal_baseline_matches": True,
            "stack_verified": True,
        },
    }
    write(destination / "manifest.json", manifest)
    # Manifestos antigos subordinados são substituídos por referências consistentes.
    write(
        destination / "observability/manifest.json",
        {
            "execution_date": execution_date,
            "parent_manifest": "../manifest.json",
            "artifacts": {
                k.removeprefix("observability/"): v
                for k, v in artifacts.items()
                if k.startswith("observability/")
            },
        },
    )
    table = "\n".join(
        f"| {r['batch_id']} | {r['roc_auc']:.4f} | {r['auc_drop']:+.4f} | {r['paired_auc_drop']:+.4f} | {r['drifted_features'] or 'Nenhuma'} | {r['gate_status']} |"
        for r in regular
    )
    (destination / "README.md").write_text(f"""# Evidências da execução de referência

Execução de **{execution_date}** (America/Sao_Paulo), modelo V2 congelado,
threshold 0,20. Sete lotes de 10 mil registros e uma demo de 2.100 linhas bloqueada.

| Mês | AUC | Queda vs. controle | Queda pareada | Features confirmadas | Gate |
|---|---:|---:|---:|---|---|
{table}

AUC do controle de 30 mil registros: **{regular[0]["reference_roc_auc"]:.4f}**.
A referência de distribuição vem do treino e não é baseline de performance.
Os meses são sintéticos; o pool já participou da avaliação histórica do modelo.
Queda pareada compara os mesmos registros antes/depois; PSI não é aditivo.

## Navegação

- HTMLs de [janeiro](drift/2026-01.html), [abril](drift/2026-04.html) e [junho](drift/2026-06.html).
- [Experimento consolidado e gráficos](experiment/summary.md).
- [CSV mensal](performance/summary.csv) e [JSON completo](performance/summary.json).
- [Sensibilidade à restauração de marginais](causal/causal_analysis.json).
- [Verificação da observabilidade](observability/README.md).
- [Bloqueio do contrato](contracts/corrupted-summary.json).
- [Manifesto de dados, modelo, fontes, parâmetros e dependências](manifest.json).
- [Linha do tempo e interpretação](../incident-timeline.md).

Baixe os HTMLs e abra no navegador; o GitHub normalmente mostra o código.
Cada HTML contém Evidently e o resumo com recomendação, métricas e comparação
pareada. Nenhuma API é necessária para abrir os arquivos.

## Proveniência

Todos os resultados desta seleção foram reempacotados para esta execução.
Os JSONs são cópias agregadas; no contrato foram removidos exemplos individuais.
Nos HTMLs foram removidos somente espaços em linhas vazias.
Parquets, sal, modelos binários e logs brutos não são publicados.

HEAD identifica a base do código; arquivos efetivos do working tree são
identificados pelos hashes do manifesto, inclusive ajustes ainda não commitados.
Depois de alterar código, modelo, dados ou parâmetros, repita os comandos do
[guia de entrega](../delivery.md) e atualize as evidências.
""")
    (
        destination / "observability/README.md"
    ).write_text(f"""# Evidências de observabilidade

Execução de {execution_date}. Os oito lotes foram verificados pelas APIs locais.

- [Resumo dos lotes](run-summary.json).
- [Métricas, MLflow, dashboard, logs e traces](stack-check.json).
- [Recebimento de alertas pelo Alertmanager](alertmanager-check.json).
- [Manifesto](manifest.json), subordinado ao [manifesto principal](../manifest.json).

Foram verificados {stack["targets_up"]} alvos disponíveis e o provisionamento do
Grafana. Os valores de AUC/gate coincidem com os relatórios. Bloqueio não gerou
predições e não interrompeu os demais meses. Alertas têm atraso de avaliação;
os snapshots podem registrar estados em instantes diferentes.

Verificação pelas APIs, sem revisão visual final do dashboard.
MLflow foi acessado na porta 5001 porque 5000 estava ocupada.
Veja [reprodução e limites](../../observability-integration.md).
""")
    # Verificação de identificadores completos e sal antes de considerar o pacote pronto.
    texts = [
        p.read_text()
        for p in destination.rglob("*")
        if p.is_file() and p.suffix != ".png"
    ]
    candidate_ids = set()
    for text in texts:
        candidate_ids.update(re.findall(r"(?<![a-f0-9])[a-f0-9]{16}(?![a-f0-9])", text))
    for name in ("production_pool.parquet", "reference_sample.parquet"):
        ids = set(
            pd.read_parquet(
                ROOT / "data/processed" / name, columns=["customer_id"]
            ).customer_id.astype(str)
        )
        if ids & candidate_ids:
            raise ValueError("Identificador de cliente presente na seleção pública")
    for key, value in dotenv_values(ROOT / ".env").items():
        if "SALT" in key.upper() and value and any(value in text for text in texts):
            raise ValueError("Sal presente na seleção pública")
    for text in texts:
        if "/Users/" in text:
            raise ValueError("Caminho local absoluto presente na seleção pública")
    print(
        f"Evidências atualizadas: {len(artifacts)} artefatos, hashes e privacidade conferidos."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-date", required=True, help="Data local YYYY-MM-DD")
    freeze(parser.parse_args().execution_date)
