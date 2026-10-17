"""Snapshots isolados por lote, persistidos antes de qualquer envio remoto."""

from __future__ import annotations

import math
import os
from pathlib import Path

from prometheus_client import (
    CollectorRegistry,
    Gauge,
    push_to_gateway,
    write_to_textfile,
)

from credit_scoring_observability.drift_report import safe_batch_id
from credit_scoring_observability.logger import get_logger

logger = get_logger(__name__)


class MetricsPublisher:
    def __init__(
        self,
        batch_id: str,
        stage: str | None = None,
        metrics_dir: Path = Path("reports/metrics"),
    ):
        self.batch_id = safe_batch_id(batch_id)
        self.stage = stage
        self.metrics_dir = metrics_dir
        self.registry = CollectorRegistry()
        self._gauges: dict[str, Gauge] = {}
        self.grouping_key = {"batch_id": self.batch_id}
        if stage:
            self.grouping_key["stage"] = stage

    def set(self, name: str, value: float | None, **labels) -> None:
        if value is None or not math.isfinite(value):
            return
        all_labels = {"batch_id": self.batch_id, **labels}
        if self.stage:
            all_labels["stage"] = self.stage
        if name not in self._gauges:
            self._gauges[name] = Gauge(
                name, f"Métrica {name}", sorted(all_labels), registry=self.registry
            )
        self._gauges[name].labels(**all_labels).set(value)

    def publish(self) -> bool:
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{self.stage}" if self.stage else ""
        write_to_textfile(
            str(self.metrics_dir / f"{self.batch_id}{suffix}.prom"), self.registry
        )
        endpoint = os.getenv("PUSHGATEWAY_URL", "").strip()
        if not endpoint:
            return True
        try:
            # PUT substitui o snapshot do lote, eliminando métricas de execução anterior.
            push_to_gateway(
                endpoint,
                job="credit_scoring_pipeline",
                registry=self.registry,
                grouping_key=self.grouping_key,
                timeout=5,
            )
            return True
        except Exception as error:
            logger.warning(
                "publicação de métricas falhou", error_type=type(error).__name__
            )
            return False
