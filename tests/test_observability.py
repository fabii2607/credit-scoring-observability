import json
from unittest.mock import Mock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from structlog.contextvars import bind_contextvars, clear_contextvars, get_contextvars

from credit_scoring_observability import metrics
from credit_scoring_observability.metrics import MetricsPublisher
from credit_scoring_observability.observation import BatchObserver
from credit_scoring_observability.parameters import PerformanceParams, SimulationParams
from credit_scoring_observability.run_all import monitor
from credit_scoring_observability.run_batch import exit_code
from credit_scoring_observability.simulation import generate
from credit_scoring_observability.stages import stage_scope
from credit_scoring_observability.synthetic import make_model_batch


def test_metrics_isolation_and_replacement(tmp_path, monkeypatch):
    monkeypatch.setenv("PUSHGATEWAY_URL", "http://gateway")
    push = Mock()
    monkeypatch.setattr(metrics, "push_to_gateway", push)
    first = MetricsPublisher("month", metrics_dir=tmp_path)
    first.set("model_roc_auc", 0.7)
    first.publish()
    second = MetricsPublisher("month", metrics_dir=tmp_path)
    second.set("pipeline_batch_status", 2)
    second.set("model_roc_auc", None)
    second.publish()
    text = (tmp_path / "month.prom").read_text()
    assert "model_roc_auc" not in text
    assert "pipeline_batch_status" in text
    assert push.call_count == 2
    assert push.call_args.kwargs["grouping_key"] == {"batch_id": "month"}


def test_remote_failure_preserves_local_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("PUSHGATEWAY_URL", "http://gateway")
    monkeypatch.setattr(
        metrics, "push_to_gateway", Mock(side_effect=OSError("unavailable"))
    )
    publisher = MetricsPublisher("month", metrics_dir=tmp_path)
    publisher.set("pipeline_batch_status", 1)
    assert not publisher.publish()
    assert (tmp_path / "month.prom").exists()


def test_stage_restores_context_and_redacts_exception():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    bind_contextvars(execution_id="execution", stage="parent")
    try:
        with (
            pytest.raises(ValueError),
            stage_scope("score", "month", tracer=provider.get_tracer("test")),
        ):
            assert get_contextvars()["stage"] == "score"
            raise ValueError("sensitive_customer_value")
        assert get_contextvars()["stage"] == "parent"
        assert get_contextvars()["execution_id"] == "execution"
        span = exporter.get_finished_spans()[0]
        assert span.attributes["error.type"] == "ValueError"
        assert not span.events
        assert "sensitive_customer_value" not in str(span.to_json())
    finally:
        clear_contextvars()
        provider.shutdown()


@pytest.mark.parametrize(
    "statuses,code",
    [
        ([{"status": "success"}], 0),
        ([{"status": "blocked"}], 2),
        ([{"status": "blocked"}, {"status": "failed"}], 1),
        ([{"status": "success", "telemetry_status": "degraded"}], 1),
    ],
)
def test_exit_codes(statuses, code):
    assert exit_code(statuses) == code


@pytest.mark.integration
def test_real_batches_continue_after_block_and_failure(
    tmp_path, scoring_model, monkeypatch
):
    monkeypatch.setenv("PUSHGATEWAY_URL", "")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "")

    # Cliente local fake aqui; execução real SQLite é verificada no smoke test.
    def start(self):
        self.publisher.set("pipeline_batch_status", -1)
        self.push()

    monkeypatch.setattr(BatchObserver, "start", start)
    pool, ref = tmp_path / "pool.parquet", tmp_path / "reference.parquet"
    make_model_batch(6500, 71).to_parquet(pool, index=False)
    reference = make_model_batch(500, 72)
    reference.assign(split="train").to_parquet(ref, index=False)
    production, reports = tmp_path / "production", tmp_path / "reports"
    generate(
        pool,
        ref,
        production,
        SimulationParams(
            batch_rows=500, control_rows=500, calibration_rows=500, demo_rows=500
        ),
    )
    (production / "2026-02.parquet").unlink()
    outcomes = monitor(
        ["corrupted", "2026-02", "2026-01"],
        production_dir=production,
        reports_dir=reports,
        reference_path=ref,
        interim_dir=tmp_path / "interim",
        quarantine_dir=tmp_path / "quarantine",
        model=scoring_model,
        performance_params=PerformanceParams(bootstrap_samples=0),
    )
    assert [r["status"] for r in outcomes] == ["blocked", "failed", "success"]
    assert set(outcomes[0]["stages"]) == {"prepare", "ingest", "validate"}
    assert outcomes[0]["stages"]["validate"]["status"] == "blocked"
    assert outcomes[1]["stages"]["ingest"]["status"] == "failed"
    assert {"score", "drift", "performance", "gate", "reports"} <= outcomes[2][
        "stages"
    ].keys()
    assert not (tmp_path / "interim/corrupted/scored.parquet").exists()
    assert exit_code(outcomes) == 1
    summary = json.loads((reports / "performance/summary.json").read_text())
    assert len(summary["batches"]) == 3
    assert "batch_sha256" not in summary["context"]
    assert "original_sha256" not in summary["context"]
    text = (reports / "metrics/2026-01.prom").read_text()
    assert "model_roc_auc" in text
    assert "feature_psi" in text
    blocked = (reports / "metrics/corrupted.prom").read_text()
    assert "model_roc_auc" not in blocked
    contract = json.loads((reports / "monitoring/corrupted/contract.json").read_text())
    assert all("examples" not in v for v in contract["violations"])
    ids = set(reference.customer_id)
    for path in (tmp_path / "interim").glob("*/scored.parquet"):
        import pandas as pd

        ids.update(pd.read_parquet(path).customer_id)
    for path in [
        *(reports / "metrics").glob("*"),
        *(reports / "monitoring").rglob("*.json"),
    ]:
        content = path.read_text()
        assert not any(value in content for value in ids)
