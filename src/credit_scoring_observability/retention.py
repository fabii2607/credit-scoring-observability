"""Descarte de lotes em quarentena após o prazo de retenção (LGPD, docs/lgpd.md).

A quarentena guarda o lote bloqueado como chegou, só pelo tempo de a origem
corrigir e reenviar. Passado o prazo (`retention.quarantine_days`, 30 dias), o
lote é apagado. O relatório do contrato fica: é agregado e os ids estão mascarados.

Uso (`make purge-quarantine`):
    uv run python -m credit_scoring_observability.retention [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from credit_scoring_observability.config import QUARANTINE_DIR
from credit_scoring_observability.logger import get_logger, setup_logging
from credit_scoring_observability.parameters import get_params

logger = get_logger(__name__)

BATCH_SUFFIXES = {".parquet", ".csv"}


def expired_batches(
    quarantine_dir: Path, max_age_days: int, now: float | None = None
) -> list[Path]:
    """Lotes da quarentena modificados há mais de `max_age_days` dias."""
    if not quarantine_dir.exists():
        return []
    limit = (time.time() if now is None else now) - max_age_days * 86_400
    return sorted(
        p
        for p in quarantine_dir.iterdir()
        if p.suffix in BATCH_SUFFIXES and p.stat().st_mtime < limit
    )


def purge(
    quarantine_dir: Path | None = None,
    max_age_days: int | None = None,
    dry_run: bool = False,
    now: float | None = None,
) -> list[str]:
    """Apaga os lotes vencidos e devolve os nomes (com `dry_run`, só lista)."""
    quarantine_dir = quarantine_dir or QUARANTINE_DIR
    days = (
        get_params().retention.quarantine_days if max_age_days is None else max_age_days
    )
    removed = []
    for path in expired_batches(quarantine_dir, days, now):
        if not dry_run:
            path.unlink()
        removed.append(path.name)
    logger.info(
        "descarte da quarentena",
        retencao_dias=days,
        removidos=removed,
        dry_run=dry_run,
    )
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Descarta lotes vencidos da quarentena"
    )
    parser.add_argument("--dry-run", action="store_true", help="só lista, não apaga")
    args = parser.parse_args(argv)
    setup_logging()
    removed = purge(dry_run=args.dry_run)
    verb = "Seriam removidos" if args.dry_run else "Removidos"
    print(f"{verb} {len(removed)} lote(s): {', '.join(removed) or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
