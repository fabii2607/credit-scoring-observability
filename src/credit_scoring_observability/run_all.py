"""Monitora a seleção do manifesto e continua após bloqueios e falhas."""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from credit_scoring_observability.analyze import write_summary
from credit_scoring_observability.config import (
    PRODUCTION_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
)
from credit_scoring_observability.drift_report import write_json
from credit_scoring_observability.logger import get_logger, setup_logging
from credit_scoring_observability.run_batch import execute_batch, exit_code
from credit_scoring_observability.tracing import setup_tracing

logger = get_logger(__name__)


def monitor(
    batch_ids=None,
    include_corrupted=False,
    production_dir=PRODUCTION_DIR,
    reports_dir=REPORTS_DIR,
    **analysis_options,
):
    manifest = json.loads((production_dir / "manifest.json").read_text())
    batches = (
        list(batch_ids)
        if batch_ids is not None
        else [b["batch_id"] for b in manifest["batches"]]
    )
    if include_corrupted and "corrupted" not in batches:
        batches.append("corrupted")
    if not batches or len(set(batches)) != len(batches):
        raise ValueError("Seleção de lotes deve ser não vazia e sem duplicações")
    outcomes = []
    provider = setup_tracing()
    try:
        for batch in batches:
            try:
                outcome = execute_batch(
                    batch,
                    production_dir=production_dir,
                    reports_dir=reports_dir,
                    provider=provider,
                    **analysis_options,
                )
            except Exception as error:
                # Falha de preparação/telemetria também não interrompe os demais.
                outcome = {
                    "batch_id": batch,
                    "status": "failed",
                    "error_type": type(error).__name__,
                }
            outcomes.append(outcome)
            logger.info("lote_monitorado", batch_id=batch, status=outcome["status"])
    finally:
        provider.shutdown()
    context = {}
    for row in outcomes:
        if row["status"] == "success":
            context = json.loads(
                (reports_dir / "drift" / f"{row['batch_id']}.json").read_text()
            )["context"]
            context = {
                k: v
                for k, v in context.items()
                if k not in {"batch_sha256", "original_sha256"}
            }
            break
    # Stages e erros de entrega ficam no consolidado de monitoramento.
    write_json(reports_dir / "monitoring/summary.json", {"batches": outcomes})
    analysis = [
        {k: v for k, v in r.items() if k not in {"stages", "telemetry_errors"}}
        for r in outcomes
    ]
    write_summary(analysis, reports_dir, context)
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+")
    parser.add_argument("--include-corrupted", action="store_true")
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    setup_logging(log_file=PROJECT_ROOT / "logs/pipeline.jsonl")
    raise SystemExit(exit_code(monitor(args.batches, args.include_corrupted)))


if __name__ == "__main__":
    main()
