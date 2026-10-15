import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # <--- Mudança aqui


def setup_tracing() -> None:
    """Configura o OpenTelemetry para enviar traces via OTLP HTTP."""
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

    if not endpoint:
        return

    provider = TracerProvider()
    exporter = OTLPSpanExporter(endpoint=endpoint)
    # SimpleSpanProcessor envia imediatamente na mesma thread, ideal para scripts curtos
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
