"""Instrumentação de estágios sem capturar dados pessoais nas exceções."""

from __future__ import annotations

import time
from contextlib import contextmanager

import psutil
from opentelemetry import trace
from structlog.contextvars import bound_contextvars

from credit_scoring_observability.logger import get_logger
from credit_scoring_observability.validate import ContractViolationError

logger = get_logger(__name__)


@contextmanager
def stage_scope(name: str, batch_id: str, record=None, tracer=None):
    tracer = tracer or trace.get_tracer(__name__)
    with (
        bound_contextvars(stage=name, batch_id=batch_id),
        tracer.start_as_current_span(
            f"stage.{name}", record_exception=False, set_status_on_exception=False
        ) as span,
    ):
        span.set_attribute("batch_id", batch_id)
        process = psutil.Process()
        cpu = sum(process.cpu_times()[:2])
        rss = process.memory_info().rss
        start = time.perf_counter()
        status = "success"
        logger.info("iniciando_estagio")
        try:
            yield
        except Exception as error:
            status = (
                "blocked" if isinstance(error, ContractViolationError) else "failed"
            )
            # Mensagens/tracebacks de bibliotecas podem conter valores de clientes.
            span.set_attribute("error.type", type(error).__name__)
            span.set_status(trace.StatusCode.ERROR, type(error).__name__)
            logger.error(
                "falha_no_estagio", error_type=type(error).__name__, status=status
            )
            raise
        finally:
            info = {
                "status": status,
                "duration_seconds": time.perf_counter() - start,
                "cpu_seconds": max(0, sum(process.cpu_times()[:2]) - cpu),
                "memory_rss_observed_bytes": max(rss, process.memory_info().rss),
            }
            for key, value in info.items():
                span.set_attribute(key, value)
            logger.info("estagio_concluido", **info)
            if record:
                record(name, info)


def run_stage(name: str, batch_id: str, fn) -> dict:
    """Compatibilidade para chamadores que precisam receber o erro estruturado."""
    captured = {}
    result, error = None, None
    try:
        with stage_scope(name, batch_id, lambda _, info: captured.update(info)):
            result = fn()
    except Exception as exc:
        error = exc
    return {**captured, "result": result, "error": error}
