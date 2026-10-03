"""Logging estruturado com structlog.

Inicialize UMA vez no ponto de entrada (CLI, script, primeira célula do notebook)
com `setup_logging()`. Os outros módulos só chamam `get_logger(__name__)`.

    from credit_scoring_observability.logger import get_logger, setup_logging

    setup_logging(level="INFO")
    logger = get_logger(__name__)
    logger.info("dados carregados", linhas=1_348_099)

Modos:
    - json_logs=False (padrão): saída legível para desenvolvimento.
    - json_logs=True: uma linha JSON por evento (produção).
    - log_file: além do console, grava cada evento como JSON Lines.

O contexto do lote entra em todos os eventos com
`structlog.contextvars.bind_contextvars(batch_id=..., stage=...)`.

LGPD: `redact_identifiers` mascara qualquer campo de identificador antes de o
evento chegar ao console ou ao arquivo.

Etapa 3: estenda a lista de processadores (trace_id, Loki) sem mudar a
assinatura de `setup_logging` e `get_logger`.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import structlog

_CONFIGURED = False

IDENTIFIER_FIELDS = {"customer_id", "customer_ids", "id"}


def redact_identifiers(_, __, event_dict: dict) -> dict:
    """Mascara campos de identificador de cliente (nunca vão em claro para logs)."""
    for key in IDENTIFIER_FIELDS & event_dict.keys():
        event_dict[key] = "***"
    return event_dict


class JsonFileTee:
    """Processador que grava o evento como uma linha JSON e o repassa adiante."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def __call__(self, _, __, event_dict: dict) -> dict:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event_dict, default=str, ensure_ascii=False) + "\n")
        return event_dict


def build_processors(json_logs: bool, log_file: Path | None) -> list:
    """Cadeia de processadores (separada para a Etapa 3 poder estender e testar)."""
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        redact_identifiers,
    ]
    if log_file is not None:
        processors.append(JsonFileTee(log_file))
    processors.append(
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer()
    )
    return processors


def setup_logging(
    level: str = "INFO", json_logs: bool = False, log_file: Path | None = None
) -> None:
    """Configura o structlog globalmente; chamadas seguintes são ignoradas.

    Args:
        level: "DEBUG", "INFO", "WARNING", "ERROR" ou "CRITICAL" (inválido = INFO).
        json_logs: True emite JSON; False, formato legível.
        log_file: se informado, cada evento também vai para este arquivo (JSONL).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    structlog.configure(
        processors=build_processors(json_logs, log_file),
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str) -> structlog.BoundLogger:
    """Logger nomeado; use sempre `get_logger(__name__)`."""
    # Valores iniciais em vez de .bind(): o proxy continua lazy e respeita o
    # setup_logging() chamado depois dos imports.
    return structlog.get_logger(module=name)
