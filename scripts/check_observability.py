"""Verifica a execução mais recente nas APIs da stack local (após make monitor)."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import requests
from mlflow import MlflowClient


def get(url, **kwargs):
    response = requests.get(url, timeout=15, **kwargs)
    response.raise_for_status()
    return response


def check(reports: Path, mlflow_uri: str):
    outcomes = json.loads((reports / "monitoring/summary.json").read_text())["batches"]
    client = MlflowClient(tracking_uri=mlflow_uri)
    checks = []
    prometheus = "http://localhost:9090"
    targets = get(prometheus + "/api/v1/targets").json()["data"]["activeTargets"]
    assert targets and all(t["health"] == "up" for t in targets), (
        "Há alvos indisponíveis"
    )
    sources = get("http://localhost:3000/api/datasources").json()
    assert {"prometheus", "loki", "tempo"} <= {s["uid"] for s in sources}
    dashboard = get(
        "http://localhost:3000/api/dashboards/uid/credit-scoring-monitoring"
    ).json()
    assert dashboard["dashboard"]["panels"]

    def vector(metric):
        data = get(prometheus + "/api/v1/query", params={"query": metric}).json()
        assert data["status"] == "success"
        return {
            r["metric"]["batch_id"]: float(r["value"][1])
            for r in data["data"]["result"]
        }

    aucs, gates, statuses = (
        vector("model_roc_auc"),
        vector("gate_status"),
        vector("pipeline_batch_status"),
    )
    for row in outcomes:
        batch = row["batch_id"]
        assert row["telemetry_status"] == "success", batch
        assert (
            statuses[batch] == {"success": 1, "blocked": 2, "failed": 0}[row["status"]]
        )
        run = client.get_run(row["mlflow_run_id"])
        assert run.data.tags["analysis_status"] == row["status"]
        assert run.info.status == (
            "FINISHED" if row["status"] == "success" else "FAILED"
        )
        trace_id = run.data.tags["trace_id"]
        # Confirma entrega do trace e presença dos logs correlacionados.
        trace = get(
            "http://localhost:3200/api/traces/" + trace_id,
            headers={"Accept": "application/json"},
        ).json()
        assert trace.get("batches")
        logs = get(
            "http://localhost:3100/loki/api/v1/query_range",
            params={
                "query": '{job="credit-scoring-pipeline"} |= "' + trace_id + '"',
                "limit": 10,
            },
        ).json()
        assert logs["data"]["result"], batch
        folders = {a.path for a in client.list_artifacts(run.info.run_id)}
        assert (
            "monitoring" not in folders
        )  # Nenhum diretório de dados brutos é enviado.
        if row["status"] == "success":
            assert math.isclose(aucs[batch], row["roc_auc"], abs_tol=1e-12)
            assert gates[batch] == {"OK": 0, "WARN": 1, "ALERT": 2}[row["gate_status"]]
            assert {"drift", "performance", "gate", "intervention"} <= folders
            assert math.isclose(
                run.data.metrics["model_roc_auc"], row["roc_auc"], abs_tol=1e-12
            )
        else:
            assert batch not in aucs and batch not in gates
        checks.append(
            {
                "batch_id": batch,
                "status": row["status"],
                "metrics_match": True,
                "mlflow_verified": True,
                "trace_delivered": True,
                "logs_correlated": True,
            }
        )
    alerts = get(prometheus + "/api/v1/alerts").json()["data"]["alerts"]
    result = {
        "targets_up": len(targets),
        "grafana_provisioned": True,
        "batches": checks,
        "alerts": [
            {
                "name": a["labels"]["alertname"],
                "batch_id": a["labels"].get("batch_id"),
                "state": a["state"],
            }
            for a in alerts
        ],
    }
    output = reports / "monitoring/stack-check.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        f"Stack verificada: {len(checks)} lotes, {len(targets)} alvos; evidência em {output}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--mlflow-uri",
        default=os.getenv("MLFLOW_TRACKING_URI") or "http://localhost:5000",
    )
    args = parser.parse_args()
    check(args.reports_dir, args.mlflow_uri)
