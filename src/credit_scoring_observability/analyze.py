"""Predições, relatórios de drift e performance dos meses simulados.

CLI: python -m credit_scoring_observability.analyze [--include-corrupted]
Não promove modelos nem recomenda retreino automaticamente.
"""

from __future__ import annotations

import argparse
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from credit_scoring_observability.config import (
    DATA_DIR,
    ID_COLUMN,
    MODEL_FILE,
    MODEL_METADATA_FILE,
    PRODUCTION_DIR,
    REFERENCE_SAMPLE_FILE,
    REPORTS_DIR,
    SPLIT_COLUMN,
)
from credit_scoring_observability.drift import detect_drift
from credit_scoring_observability.drift_report import (
    safe_batch_id,
    write_drift_report,
    write_json,
)
from credit_scoring_observability.logger import get_logger, setup_logging
from credit_scoring_observability.parameters import (
    DriftParams,
    PerformanceParams,
    get_params,
)
from credit_scoring_observability.performance import evaluate_performance
from credit_scoring_observability.registry import ScoringModel, load_model
from credit_scoring_observability.simulation import file_digest
from credit_scoring_observability.validate import (
    ContractViolationError,
    enforce,
    validate_batch,
)

logger = get_logger(__name__)


def score_validated(
    frame: pd.DataFrame,
    model: ScoringModel,
    batch_id: str,
    report_dir: Path,
    quarantine_dir: Path,
    source_path: Path | None = None,
    min_rows: int | None = None,
) -> pd.DataFrame:
    """Gate completo: nenhuma chamada ao modelo antes de enforce aprovar."""
    result = validate_batch(frame, batch_id, min_rows=min_rows)
    enforce(
        frame,
        result,
        source_path=source_path,
        report_dir=report_dir,
        quarantine_dir=quarantine_dir,
    )
    return model.score(frame)


def analyze_batch(
    path: Path,
    model: ScoringModel,
    reference_scored: pd.DataFrame,
    control_scored: pd.DataFrame,
    interim_dir: Path,
    reports_dir: Path,
    quarantine_dir: Path,
    drift_params: DriftParams | None = None,
    performance_params: PerformanceParams | None = None,
    context: dict | None = None,
    expected_sha256: str | None = None,
) -> dict:
    """Executa um lote, devolvendo blocked/failed sem interromper os seguintes.

    A leitura/validação é feita no arquivo real. Apenas arquivos gerados conhecidos
    daquele lote são invalidados ao reexecutar; dados de origem são preservados,
    exceto pelo movimento para quarentena previsto no contrato.
    """
    batch_id = safe_batch_id(path.stem)
    drift_params, performance_params = (
        drift_params or DriftParams(),
        performance_params or PerformanceParams(),
    )
    context = context or {}
    work = interim_dir / batch_id
    outputs = [
        work / "model_input.parquet",
        work / "scored.parquet",
        reports_dir / "drift" / f"{batch_id}.html",
        reports_dir / "drift" / f"{batch_id}.json",
        reports_dir / "performance" / f"{batch_id}.json",
    ]
    for output in outputs:
        output.unlink(missing_ok=True)
    status_path = work / "status.json"
    write_json(status_path, {"batch_id": batch_id, "status": "running"})
    rows = None
    try:
        frame = pd.read_parquet(path)
        rows = len(frame)
        # O contrato é aplicado antes da checagem de integridade e da inferência.
        validation = validate_batch(frame, batch_id)
        enforce(
            frame,
            validation,
            source_path=path,
            report_dir=reports_dir / "contracts",
            quarantine_dir=quarantine_dir,
        )
        source_sha = file_digest(path)
        if expected_sha256 is not None and source_sha != expected_sha256:
            raise ValueError("Lote difere do manifesto; gere ou verifique a simulação")
        ids = set(frame[ID_COLUMN])
        if ids & set(reference_scored[ID_COLUMN]) or ids & set(
            control_scored[ID_COLUMN]
        ):
            raise ValueError("Lote compartilha identificadores com referência/controle")
        # enforce já aprovou o lote completo; score aplica adicionalmente o schema de features.
        scored = model.score(frame)
        work.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(work / "model_input.parquet", index=False)
        scored.to_parquet(work / "scored.parquet", index=False)
        drift = detect_drift(reference_scored, scored, batch_id, drift_params)
        perf = evaluate_performance(
            control_scored,
            scored,
            model.threshold,
            performance_params,
            baseline_source="reserves/control.parquet (fora do treino)",
        )
        batch_context = {**context, "batch_sha256": source_sha}
        write_drift_report(
            reference_scored,
            scored,
            drift,
            reports_dir / "drift",
            drift_params,
            batch_context,
        )
        write_json(
            reports_dir / "performance" / f"{batch_id}.json",
            {
                "batch_id": batch_id,
                **perf.to_dict(),
                "parameters": performance_params.model_dump(mode="json"),
                "context": batch_context,
            },
        )
        outcome = {
            "batch_id": batch_id,
            "status": "success",
            **asdict(perf),
            "drift_share": drift.drift_share,
            "drifted_features": ",".join(drift.drifted_features),
            "score_psi": drift.score_psi,
            "label_z": drift.label_z,
        }
    except ContractViolationError:
        outcome = {
            "batch_id": batch_id,
            "status": "blocked",
            "rows": rows,
            "error_type": "ContractViolationError",
        }
    except Exception as error:
        logger.error(
            "análise do lote falhou", batch_id=batch_id, error_type=type(error).__name__
        )
        outcome = {
            "batch_id": batch_id,
            "status": "failed",
            "rows": rows,
            "error_type": type(error).__name__,
        }
    write_json(status_path, outcome)
    return outcome


def run(
    production_dir: Path = PRODUCTION_DIR,
    reference_path: Path = REFERENCE_SAMPLE_FILE,
    interim_dir: Path = DATA_DIR / "interim",
    reports_dir: Path = REPORTS_DIR,
    quarantine_dir: Path = DATA_DIR / "quarantine",
    batch_ids: list[str] | None = None,
    include_corrupted: bool = False,
    model: ScoringModel | None = None,
    performance_params: PerformanceParams | None = None,
) -> list[dict]:
    """Executa meses do manifesto com modelo congelado e controle separado."""
    import json

    params = get_params()
    manifest_path = production_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Execute make simulate antes da análise")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    batch_ids = (
        batch_ids
        if batch_ids is not None
        else [b["batch_id"] for b in manifest["batches"]]
    )
    if include_corrupted and "corrupted" not in batch_ids:
        batch_ids = [*batch_ids, "corrupted"]
    for batch_id in batch_ids:
        safe_batch_id(batch_id)
        if f"{batch_id}.parquet" not in manifest["artifacts"]:
            raise ValueError("Lote não consta no manifesto")
    if not batch_ids or len(set(batch_ids)) != len(batch_ids):
        raise ValueError("Seleção de lotes deve ser não vazia e sem duplicações")
    injected_model = model is not None
    model = model or load_model()
    reference = pd.read_parquet(reference_path).drop(
        columns=SPLIT_COLUMN, errors="ignore"
    )
    control_path = production_dir / "reserves" / "control.parquet"
    if (
        file_digest(control_path)
        != manifest["artifacts"]["reserves/control.parquet"]["sha256"]
    ):
        raise ValueError("Controle difere do manifesto")
    control = pd.read_parquet(control_path)
    if set(reference[ID_COLUMN]) & set(control[ID_COLUMN]):
        raise ValueError("Controle e referência não podem compartilhar registros")
    reference_scored = score_validated(
        reference, model, "reference_sample", reports_dir / "contracts", quarantine_dir
    )
    control_scored = score_validated(
        control, model, "performance_control", reports_dir / "contracts", quarantine_dir
    )
    context = {
        "model_version": model.version,
        "threshold": model.threshold,
        "reference_sha256": file_digest(reference_path),
        "control_sha256": file_digest(control_path),
        "simulation_manifest_sha256": file_digest(manifest_path),
        "versions": {
            name: version(name)
            for name in ("evidently", "scikit-learn", "numpy", "pandas", "scipy")
        },
        "python": platform.python_version(),
        "implementation_sha256": {
            name: file_digest(Path(__file__).with_name(name))
            for name in (
                "analyze.py",
                "drift.py",
                "drift_report.py",
                "performance.py",
                "stats.py",
                "results.py",
            )
        },
    }
    if not injected_model:
        context.update(
            model_sha256=file_digest(MODEL_FILE),
            metadata_sha256=file_digest(MODEL_METADATA_FILE),
        )
    for name, scored in [
        ("_reference", reference_scored),
        ("_control", control_scored),
    ]:
        destination = interim_dir / name
        destination.mkdir(parents=True, exist_ok=True)
        scored.to_parquet(destination / "scored.parquet", index=False)
        write_json(destination / "metadata.json", context)
    outcomes = []
    for batch_id in batch_ids:
        outcome = analyze_batch(
            production_dir / f"{batch_id}.parquet",
            model,
            reference_scored,
            control_scored,
            interim_dir,
            reports_dir,
            quarantine_dir,
            params.drift,
            performance_params or params.performance,
            context,
            manifest["artifacts"][f"{batch_id}.parquet"]["sha256"],
        )
        outcomes.append(outcome)
        logger.info("lote analisado", batch_id=batch_id, status=outcome["status"])
    write_json(
        reports_dir / "performance" / "summary.json",
        {"context": context, "batches": outcomes},
    )
    pd.DataFrame(outcomes).to_csv(
        reports_dir / "performance" / "summary.csv", index=False
    )
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batches", nargs="+")
    parser.add_argument("--include-corrupted", action="store_true")
    args = parser.parse_args()
    setup_logging()
    outcomes = run(batch_ids=args.batches, include_corrupted=args.include_corrupted)
    if any(b["status"] == "failed" for b in outcomes):
        raise SystemExit(1)
    if any(b["status"] == "blocked" for b in outcomes):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
