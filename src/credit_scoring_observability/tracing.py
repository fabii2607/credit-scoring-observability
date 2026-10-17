"""Provider explícito: spans locais e exportação OTLP opcional."""

import os

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


def setup_tracing() -> TracerProvider:
    provider = TracerProvider(
        resource=Resource.create({"service.name": "credit-scoring-pipeline"})
    )
    base = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    explicit = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "").strip()
    if base or explicit:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        endpoint = explicit or base.rstrip("/") + "/v1/traces"
        provider.add_span_processor(
            SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint, timeout=5))
        )
    return provider
