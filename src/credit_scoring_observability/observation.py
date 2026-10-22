"""Publicação dos resultados reais, sem exportar linhas ou exemplos de clientes."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from mlflow import MlflowClient

from credit_scoring_observability.drift_report import write_json
from credit_scoring_observability.logger import get_logger
from credit_scoring_observability.metrics import MetricsPublisher
from credit_scoring_observability.stages import stage_scope

logger = get_logger(__name__)


class BatchObserver:
    def __init__(self, batch_id: str, reports_dir: Path, tracer):
        self.batch_id, self.reports_dir, self.tracer = batch_id, reports_dir, tracer
        self.publisher = MetricsPublisher(batch_id, metrics_dir=reports_dir / "metrics")
        self.stages = {}
        self.errors = []
        self.client = None
        self.run_id = None

    def delivery_error(self, channel, error):
        self.errors.append({"channel": channel, "error_type": type(error).__name__})
        logger.warning(
            "telemetria_indisponivel", channel=channel, error_type=type(error).__name__
        )

    def push(self):
        if not self.publisher.publish():
            self.errors.append({"channel": "pushgateway", "error_type": "PublishError"})

    def start(self):
        self.publisher.set("pipeline_batch_status", -1)
        self.push()
        try:
            uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
            if not uri:
                store = self.reports_dir.parent / "mlruns"
                store.mkdir(parents=True, exist_ok=True)
                uri = f"sqlite:///{(store / 'tracking.db').resolve()}"
            self.client = MlflowClient(tracking_uri=uri)
            name = "credit_scoring_monitoring"
            experiment = self.client.get_experiment_by_name(name)
            experiment_id = (
                experiment.experiment_id
                if experiment
                else self.client.create_experiment(
                    name,
                    artifact_location=(self.reports_dir.parent / "mlruns" / "artifacts")
                    .resolve()
                    .as_uri()
                    if uri.startswith("sqlite:")
                    else None,
                )
            )
            run = self.client.create_run(
                experiment_id,
                tags={
                    "mlflow.runName": f"batch_{self.batch_id}",
                    "batch_id": self.batch_id,
                },
            )
            self.run_id = run.info.run_id
        except Exception as error:
            self.delivery_error("mlflow", error)
            self.client = None

    def ml(self, method, *args, **kwargs):
        if self.client and self.run_id:
            try:
                getattr(self.client, method)(self.run_id, *args, **kwargs)
            except Exception as error:
                self.delivery_error("mlflow", error)

    @contextmanager
    def stage(self, name):
        with stage_scope(name, self.batch_id, self.record, self.tracer):
            yield

    def record(self, name, info):
        self.stages[name] = info
        for metric, key in [
            ("pipeline_stage_duration_seconds", "duration_seconds"),
            ("stage_cpu_seconds", "cpu_seconds"),
            ("stage_memory_rss_observed_bytes", "memory_rss_observed_bytes"),
        ]:
            self.publisher.set(metric, info[key], stage=name)
            self.ml("log_metric", f"{name}_{metric}", info[key])
        self.publisher.set(
            "pipeline_stage_status",
            {"success": 1, "blocked": 2, "failed": 0}[info["status"]],
            stage=name,
        )
        self.push()

    def finish(self, outcome):
        status = outcome["status"]
        self.publisher.set(
            "pipeline_batch_status", {"success": 1, "blocked": 2, "failed": 0}[status]
        )
        self.publisher.set("batch_quarantined", int(status == "blocked"))
        self.publisher.set("batch_rows", outcome.get("rows"))
        self.publisher.set(
            "pipeline_rows_processed", outcome.get("rows") if status == "success" else 0
        )
        work = self.reports_dir / "monitoring" / self.batch_id
        work.mkdir(parents=True, exist_ok=True)
        contract = self.reports_dir / "contracts" / f"{self.batch_id}.json"
        artifacts = []
        if contract.exists() and "validate" in self.stages:
            data = json.loads(contract.read_text(encoding="utf-8"))
            safe = {
                k: data[k]
                for k in ("batch_id", "n_rows", "passed", "n_blockers", "n_warnings")
            }
            safe["violations"] = [
                {
                    k: v
                    for k, v in row.items()
                    if k in {"rule", "column", "severity", "count", "layer"}
                }
                for row in data["violations"]
            ]
            public_contract = work / "contract.json"
            write_json(public_contract, safe)
            artifacts.append(public_contract)
            # Regras repetidas em colunas diferentes são agregadas.
            counts = {}
            for row in safe["violations"]:
                key = (row["rule"], row["severity"], row["layer"])
                counts[key] = counts.get(key, 0) + row["count"]
            for (rule, severity, layer), count in counts.items():
                self.publisher.set(
                    "contract_violations",
                    count,
                    rule=rule,
                    severity=severity,
                    layer=layer,
                )
        if status == "success":
            self.publisher.set("pipeline_last_success_timestamp", time.time())
            self.publish_analysis(outcome)
            for folder, extension in [
                ("drift", "html"),
                ("drift", "json"),
                ("performance", "json"),
                ("gate", "json"),
                ("intervention", "json"),
            ]:
                path = self.reports_dir / folder / f"{self.batch_id}.{extension}"
                if path.exists():
                    artifacts.append(path)
        self.ml("set_tag", "analysis_status", status)
        for path in artifacts:
            self.ml("log_artifact", str(path), artifact_path=path.parent.name)
        self.publisher.set("telemetry_delivery_status", int(not self.errors))
        self.push()
        telemetry_status = "degraded" if self.errors else "success"
        self.ml("set_tag", "telemetry_status", telemetry_status)
        self.ml(
            "set_terminated",
            status="FINISHED" if status == "success" and not self.errors else "FAILED",
        )
        summary = {
            **outcome,
            "mlflow_run_id": self.run_id,
            "stages": self.stages,
            "telemetry_status": "degraded" if self.errors else "success",
            "telemetry_errors": self.errors,
        }
        write_json(work / "run.json", summary)
        return summary

    def publish_analysis(self, outcome):
        drift = json.loads(
            (self.reports_dir / "drift" / f"{self.batch_id}.json").read_text(
                encoding="utf-8"
            )
        )
        gate = drift["gate"]
        for feature in drift["features"]:
            for field, metric in [
                ("psi", "feature_psi"),
                ("ks_d", "feature_ks_d"),
                ("categorical_distance", "feature_js"),
                ("missing_rate_delta", "feature_missing_rate_delta"),
            ]:
                self.publisher.set(metric, feature[field], feature=feature["feature"])
            self.publisher.set(
                "feature_drift_confirmed",
                int(feature["drifted"]),
                feature=feature["feature"],
            )
        mapping = {
            "drift_share": "drift_share",
            "score_psi": "score_psi",
            "roc_auc": "model_roc_auc",
            "reference_roc_auc": "model_reference_roc_auc",
            "auc_drop": "model_auc_drop",
            "auc_drop_ci_low": "model_auc_drop_ci_low",
            "auc_drop_ci_high": "model_auc_drop_ci_high",
            "pr_auc": "model_pr_auc",
            "brier": "model_brier",
            "approval_rate": "approval_rate",
            "default_rate": "model_default_rate",
            "label_z": "label_z",
        }
        for field, metric in mapping.items():
            value = outcome.get(field)
            self.publisher.set(metric, value)
            if value is not None:
                self.ml("log_metric", metric, value)
        context = drift["context"]
        self.publisher.set(
            "model_auc_drop_limit", context["performance_parameters"]["auc_drop"]
        )
        self.publisher.set(
            "gate_status", {"OK": 0, "WARN": 1, "ALERT": 2}[gate["status"]]
        )
        self.publisher.set("gate_retrain_recommended", int(gate["retrain_recommended"]))
        self.publisher.set(
            "gate_concept_suspected", int(gate["diagnosis"] == "concept_suspected")
        )
        self.publisher.set("label_z_limit", context["drift_parameters"]["label_z"])
        for key in (
            "model_version",
            "threshold",
            "model_sha256",
            "metadata_sha256",
            "batch_sha256",
            "reference_sha256",
            "control_sha256",
        ):
            if key in context:
                self.ml("log_param", key, context[key])
        self.ml("set_tag", "gate_status", gate["status"])
        self.ml("set_tag", "recommendation", gate["recommendation"])
