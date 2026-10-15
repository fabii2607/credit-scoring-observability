import argparse
import sys
import pandas as pd
from pathlib import Path
import mlflow
import structlog
from structlog.contextvars import bind_contextvars
from opentelemetry import trace
from opentelemetry.trace import set_span_in_context, NonRecordingSpan, SpanContext, TraceFlags
from opentelemetry import context as otel_context

from credit_scoring_observability.metrics import MetricsPublisher
from credit_scoring_observability.stages import run_stage
from credit_scoring_observability.validate import validate_batch, enforce, ContractViolationError
from credit_scoring_observability.synthetic import make_model_batch
from credit_scoring_observability.logger import setup_logging
from credit_scoring_observability.tracing import setup_tracing

# Inicializa o logging (com gravação no JSONL para o Alloy) e o tracing no processo filho
setup_logging(log_file=Path("logs/pipeline.jsonl"))
setup_tracing()

logger = structlog.get_logger(__name__)

def execute_validation_stage(batch_id: str, df: pd.DataFrame, publisher: MetricsPublisher):
    result = validate_batch(df, batch_id)
    
    # Grava métricas no Prometheus/Pushgateway
    for violation in result.violations:
        publisher.set(
            "contract_violations", 
            violation.count, 
            rule=violation.rule, 
            severity=violation.severity, 
            layer=violation.layer
        )
        # Grava as mesmas métricas no MLflow
        mlflow.log_metric(f"violation_{violation.rule}_{violation.severity}", violation.count)
        
    publisher.set("batch_rows", len(df))
    mlflow.log_metric("batch_rows", len(df))
    publisher.publish()
    
    enforce(df, result, source_path=Path(f"data/production/{batch_id}.parquet"))
        
    publisher.set("batch_quarantined", 0)
    publisher.publish()
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, help="ID do lote (ex: 2026-03)")
    args = parser.parse_args()
    
    # 1. Configura o MLflow antes de iniciar os estágios
    mlflow.set_experiment("credit_scoring_pipeline")
    
    with mlflow.start_run(run_name=f"batch_{args.batch}") as run:
        run_id = run.info.run_id
        
        # Força o OpenTelemetry a adotar o run_id do MLflow como o Trace ID global deste lote
        trace_id_int = int(run_id[:32].ljust(32, '0'), 16)
        context = SpanContext(trace_id=trace_id_int, span_id=1, is_remote=False, trace_flags=TraceFlags(1))
        token = otel_context.attach(set_span_in_context(NonRecordingSpan(context)))
        
        # O trace_id exigido é injetado no log como mlflow_run_id
        bind_contextvars(trace_id=run_id, mlflow_run_id=run_id)
        logger.info("iniciando_pipeline_mlflow", batch_id=args.batch, mlflow_run_id=run_id)
        
        mlflow.log_param("batch_id", args.batch)
        mlflow.set_tag("trace_id", run_id)
        
        # 2. Ingestão
        df = make_model_batch(100)
        
        # 3. Validação
        validate_publisher = MetricsPublisher(batch_id=args.batch, stage="validate")
        
        def validate_fn():
            return execute_validation_stage(args.batch, df, validate_publisher)
            
        stage_info = run_stage("validate", args.batch, validate_fn)
        
        # 4. Métricas de infraestrutura
        validate_publisher.set("pipeline_stage_duration_seconds", stage_info["duration_seconds"])
        validate_publisher.set("stage_cpu_seconds", stage_info["cpu_seconds"])
        validate_publisher.set("stage_memory_peak_bytes", stage_info["memory_peak_bytes"])
        
        mlflow.log_metric("duration_seconds", stage_info["duration_seconds"])
        
        is_success = stage_info["status"] == "success"
        validate_publisher.set("pipeline_stage_status", 1 if is_success else 0)
        
        is_contract_violation = isinstance(stage_info.get("error"), ContractViolationError)
        if is_contract_violation:
            validate_publisher.set("batch_quarantined", 1)
            mlflow.log_metric("batch_quarantined", 1)
            
            # Submete o relatório de erro ao MLflow
            report_path = f"reports/contracts/{args.batch}.json"
            if Path(report_path).exists():
                mlflow.log_artifact(report_path, "contract_reports")
                
        validate_publisher.publish()

    # O exit deve ocorrer apenas FORA do bloco with mlflow para garantir que o run é fechado
    if is_contract_violation:
        # Encerramento forçado e síncrono do OpenTelemetry
        provider = trace.get_tracer_provider()
        if hasattr(provider, "shutdown"):
            provider.shutdown()
        sys.exit(2)

if __name__ == "__main__":
    main()