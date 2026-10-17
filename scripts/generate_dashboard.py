"""Gera o dashboard provisionado, usando o contrato de métricas do pipeline."""

import json
from pathlib import Path


def create_dashboard():
    panels = []
    specs = [
        (
            "Estado do lote: -1 executando / 0 falha / 1 sucesso / 2 bloqueado",
            "pipeline_batch_status",
            "stat",
        ),
        ("Gate: 0 OK / 1 WARN / 2 ALERT", "gate_status", "stat"),
        ("Avaliar retreinamento: 1 recomendado", "gate_retrain_recommended", "stat"),
        ("AUC do modelo e do controle", "model_roc_auc", "timeseries"),
        ("Queda de AUC vs. controle", "model_auc_drop", "timeseries"),
        ("Proporção de features com drift", "drift_share", "stat"),
        ("PSI por feature", "feature_psi", "timeseries"),
        ("KS D por feature", "feature_ks_d", "timeseries"),
        ("PSI do score", "score_psi", "timeseries"),
        ("Aprovação", "approval_rate", "stat"),
        ("Brier", "model_brier", "stat"),
        ("Registros processados", "pipeline_rows_processed", "stat"),
        ("Duração por estágio (s)", "pipeline_stage_duration_seconds", "timeseries"),
        ("CPU por estágio (s)", "stage_cpu_seconds", "timeseries"),
        (
            "RSS observado nos limites do estágio (bytes)",
            "stage_memory_rss_observed_bytes",
            "timeseries",
        ),
        ("Violações de contrato", "contract_violations", "timeseries"),
        (
            "Status do estágio: 0 falha / 1 sucesso / 2 bloqueado",
            "pipeline_stage_status",
            "stat",
        ),
        ("Último sucesso (Unix timestamp)", "pipeline_last_success_timestamp", "stat"),
    ]
    for index, (title, expression, kind) in enumerate(specs):
        targets = [
            {
                "refId": "A",
                "expr": expression,
                "legendFormat": "{{batch_id}} {{stage}} {{feature}}",
            }
        ]
        if expression == "model_roc_auc":
            targets.append(
                {
                    "refId": "B",
                    "expr": "model_reference_roc_auc",
                    "legendFormat": "Controle {{batch_id}}",
                }
            )
        if expression == "model_auc_drop":
            targets += [
                {"refId": ref, "expr": metric, "legendFormat": metric + " {{batch_id}}"}
                for ref, metric in [
                    ("B", "model_auc_drop_ci_low"),
                    ("C", "model_auc_drop_ci_high"),
                    ("D", "model_auc_drop_limit"),
                ]
            ]
        panels.append(
            {
                "id": index + 1,
                "title": title,
                "type": kind,
                "datasource": {"type": "prometheus", "uid": "prometheus"},
                "gridPos": {
                    "h": 7,
                    "w": 8,
                    "x": (index % 3) * 8,
                    "y": (index // 3) * 7,
                },
                "targets": targets,
                "options": {
                    "reduceOptions": {"calcs": ["lastNotNull"], "values": False}
                },
            }
        )
    panels.append(
        {
            "id": len(panels) + 1,
            "title": "Logs: trace_id abre o Tempo",
            "type": "logs",
            "datasource": {"type": "loki", "uid": "loki"},
            "gridPos": {"h": 10, "w": 24, "x": 0, "y": 42},
            "targets": [{"refId": "A", "expr": '{job="credit-scoring-pipeline"}'}],
        }
    )
    dashboard = {
        "uid": "credit-scoring-monitoring",
        "title": "Observabilidade - Credit Scoring",
        "schemaVersion": 38,
        "version": 1,
        "tags": ["mlops", "credit-scoring"],
        "timezone": "browser",
        "refresh": "15s",
        "time": {"from": "now-1h", "to": "now"},
        "panels": panels,
    }
    output = Path(__file__).resolve().parents[1] / "monitoring/grafana/dashboard.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dashboard, ensure_ascii=False, indent=2) + "\n")
    print(f"Dashboard gerado: {output.relative_to(output.parents[2])}")
    return dashboard


if __name__ == "__main__":
    create_dashboard()
