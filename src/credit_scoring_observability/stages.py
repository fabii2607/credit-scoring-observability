import time

import psutil
import structlog
from opentelemetry import trace
from structlog.contextvars import bind_contextvars, clear_contextvars

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)


def run_stage(name: str, batch_id: str, fn) -> dict:
    """
    Envolve a execução de um estágio para coletar métricas de infraestrutura,
    garantir logs estruturados com contexto isolado e rastrear a execução (traces).
    """
    bind_contextvars(stage=name, batch_id=batch_id)

    with tracer.start_as_current_span(f"stage.{name}") as span:
        span.set_attribute("batch_id", batch_id)
        logger.info("iniciando_estagio")

        process = psutil.Process()
        cpu_start = process.cpu_times().user + process.cpu_times().system
        mem_start = process.memory_info().rss
        start_time = time.time()

        status = "success"
        result = None
        error = None

        try:
            result = fn()
            span.set_status(trace.StatusCode.OK)
        except Exception as e:
            status = "failed"
            error = e
            span.set_status(trace.StatusCode.ERROR, str(e))
            span.record_exception(e)
            logger.error("falha_no_estagio", error=str(e))
        finally:
            end_time = time.time()
            cpu_end = process.cpu_times().user + process.cpu_times().system
            mem_peak = process.memory_info().rss  # Aproximação do pico

            duration = end_time - start_time
            cpu_seconds = max(0.0, cpu_end - cpu_start)

            span.set_attribute("duration_seconds", duration)
            span.set_attribute("cpu_seconds", cpu_seconds)
            span.set_attribute("memory_peak_bytes", max(mem_start, mem_peak))

            stage_info = {
                "status": status,
                "duration_seconds": duration,
                "cpu_seconds": cpu_seconds,
                "memory_peak_bytes": max(mem_start, mem_peak),
                "result": result,
                "error": error,
            }

            logger.info(
                "estagio_concluido", status=status, duration_seconds=round(duration, 2)
            )

            clear_contextvars()

        return stage_info
