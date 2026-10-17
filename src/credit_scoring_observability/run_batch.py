"""Executa um lote real com métricas, MLflow, logs e tracing."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from dotenv import load_dotenv
from structlog.contextvars import bound_contextvars

from credit_scoring_observability.analyze import run
from credit_scoring_observability.config import PROJECT_ROOT, REPORTS_DIR
from credit_scoring_observability.drift_report import safe_batch_id
from credit_scoring_observability.logger import get_logger, setup_logging
from credit_scoring_observability.observation import BatchObserver
from credit_scoring_observability.tracing import setup_tracing

logger = get_logger(__name__)


def exit_code(outcomes: list[dict]) -> int:
    if any(
        o["status"] == "failed" or o.get("telemetry_status") == "degraded"
        for o in outcomes
    ):
        return 1
    return 2 if any(o["status"] == "blocked" for o in outcomes) else 0


def execute_batch(
    batch_id: str, reports_dir: Path = REPORTS_DIR, provider=None, **analysis_options
) -> dict:
    safe_batch_id(batch_id)
    owned = provider is None
    provider = provider or setup_tracing()
    tracer = provider.get_tracer(__name__)
    observer = BatchObserver(batch_id, reports_dir, tracer)
    try:
        with tracer.start_as_current_span(
            "pipeline.batch", record_exception=False, set_status_on_exception=False
        ) as span:
            trace_id = format(span.get_span_context().trace_id, "032x")
            with bound_contextvars(
                batch_id=batch_id, execution_id=uuid.uuid4().hex, trace_id=trace_id
            ):
                observer.start()
                with bound_contextvars(mlflow_run_id=observer.run_id):
                    span.set_attribute("batch_id", batch_id)
                    if observer.run_id:
                        span.set_attribute("mlflow.run_id", observer.run_id)
                        observer.ml("set_tag", "trace_id", trace_id)
                    logger.info("iniciando_pipeline")
                    try:
                        outcomes = run(
                            batch_ids=[batch_id],
                            reports_dir=reports_dir,
                            observer=observer,
                            consolidate=False,
                            **analysis_options,
                        )
                        outcome = outcomes[0]
                    except Exception as error:
                        outcome = {
                            "batch_id": batch_id,
                            "status": "failed",
                            "error_type": type(error).__name__,
                        }
                        logger.error("pipeline_falhou", error_type=type(error).__name__)
                    result = observer.finish(outcome)
                    if exit_code([result]):
                        from opentelemetry import trace

                        span.set_status(trace.StatusCode.ERROR, result["status"])
                    logger.info(
                        "pipeline_concluido",
                        status=result["status"],
                        telemetry_status=result["telemetry_status"],
                    )
                    return result
    finally:
        if owned:
            provider.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    setup_logging(log_file=PROJECT_ROOT / "logs/pipeline.jsonl")
    raise SystemExit(exit_code([execute_batch(args.batch)]))


if __name__ == "__main__":
    main()
