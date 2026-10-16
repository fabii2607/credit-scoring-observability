"""Simulação mensal reprodutível a partir do teste reservado da modelagem.

CLI: python -m credit_scoring_observability.simulation
Não treina nem pontua modelos. Registros originais e issue_date são preservados.
Datas de 2026 identificam a simulação, não novas observações históricas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from credit_scoring_observability.config import (
    ID_COLUMN,
    ISSUE_COLUMN,
    PRODUCTION_DIR,
    PRODUCTION_POOL_FILE,
    REFERENCE_FILE,
    REGION_COLUMN,
    TARGET,
)
from credit_scoring_observability.data_contract import MODEL_FEATURES
from credit_scoring_observability.logger import get_logger, setup_logging
from credit_scoring_observability.parameters import SimulationParams, get_params
from credit_scoring_observability.validate import make_corrupted, validate_batch

SIMULATION_VERSION = "1.0"
DATA_FEATURES = ["annual_inc", "dti", "revol_util", "fico_avg"]
logger = get_logger(__name__)


def _intensity(value: float) -> None:
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Intensidade deve ser finita e estar entre 0 e 1")


def apply_data_drift(
    batch: pd.DataFrame,
    intensity: float,
    rng: np.random.Generator | None = None,
    params: SimulationParams | None = None,
) -> pd.DataFrame:
    """Desloca quatro features, preservando Y, metadados e nulos.

    Transformações determinísticas: rng é aceito para manter a interface do
    simulador, mas não é consumido. FICO sofre deslocamento em múltiplos de 5;
    o corte em 300/850 pode reduzir esse deslocamento nos extremos.
    """
    _intensity(intensity)
    params = params or SimulationParams()
    out = batch.copy(deep=True)
    if intensity == 0:
        return out
    out["annual_inc"] = out["annual_inc"].astype(float) * (
        1 - params.income_reduction * intensity
    )
    out["dti"] = out["dti"].astype(float) * (1 + params.dti_increase * intensity)
    out["revol_util"] = (
        out["revol_util"].astype(float) + params.utilization_increase_pp * intensity
    )
    # Arredonda o DESLOCAMENTO, sem arredondar o FICO médio original.
    drop = 5 * int(np.floor(params.fico_drop_points * intensity / 5 + 0.5))
    out["fico_avg"] = (out["fico_avg"].astype(float) - drop).clip(300, 850)
    return out


def concept_segments(
    batch: pd.DataFrame, params: SimulationParams
) -> tuple[np.ndarray, np.ndarray]:
    """Grupos elegíveis definidos no X/Y original, sem depender do score.

    Novos defaults: bons pagadores com FICO >= 700 e prazo 36.
    Curas: inadimplentes com FICO <= 680 OU prazo 60.
    Limiares e probabilidades máximas são configuráveis, não inferência causal.
    """
    new_default = (
        batch[TARGET].eq(0)
        & batch["fico_avg"].ge(params.concept_high_fico)
        & batch["term"].eq(36)
    )
    cure = batch[TARGET].eq(1) & (
        batch["fico_avg"].le(params.concept_low_fico) | batch["term"].eq(60)
    )
    return new_default.to_numpy(dtype=bool), cure.to_numpy(dtype=bool)


def apply_concept_drift(
    batch: pd.DataFrame,
    intensity: float,
    rng: np.random.Generator,
    params: SimulationParams | None = None,
) -> pd.DataFrame:
    """Relabel probabilístico: somente target muda; features e ordem ficam iguais.

    Usa um sorteio por registro; com o mesmo estado inicial do RNG, aumentar a
    intensidade inclui as alterações menores. Lotes mensais têm empréstimos/ids
    distintos, portanto contagens observadas não precisam ser monotônicas.
    """
    _intensity(intensity)
    params = params or SimulationParams()
    out = batch.copy(deep=True)
    if intensity == 0:
        return out
    new_default, cure = concept_segments(batch, params)
    draws = rng.random(len(batch))
    out.loc[
        new_default & (draws < intensity * params.concept_default_probability), TARGET
    ] = 1
    out.loc[cure & (draws < intensity * params.concept_cure_probability), TARGET] = 0
    return out


def _validate(frame: pd.DataFrame, batch_id: str, min_rows: int) -> dict:
    result = validate_batch(frame, batch_id, min_rows=min_rows)
    if not result.passed:
        rules = sorted({v.rule for v in result.blockers})
        raise ValueError(f"Contrato inválido em {batch_id}: {', '.join(rules)}")
    return {"passed": True, "warnings": len(result.warnings)}


def frame_digest(frame: pd.DataFrame) -> str:
    """Hash de conteúdo/ordem/schema, sem expor linhas no manifesto."""
    digest = hashlib.sha256()
    schema = [(c, str(frame[c].dtype)) for c in frame.columns]
    digest.update(json.dumps(schema).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass
class SimulationData:
    frames: dict[str, pd.DataFrame]
    manifest: dict


def build_simulation(
    pool: pd.DataFrame,
    reference_ids: pd.Series,
    params: SimulationParams,
    min_rows: int = 500,
) -> SimulationData:
    """Audita, particiona sem reposição e gera os cenários em memória.

    Inválidos não são removidos silenciosamente: a geração falha antes de gravar.
    reference_ids deve representar a referência COMPLETA, não só sua amostra.
    """
    required = {ID_COLUMN, ISSUE_COLUMN, REGION_COLUMN, TARGET, *MODEL_FEATURES}
    if required - set(pool.columns):
        raise ValueError("Pool não contém todas as features, target e metadados")
    if pool[TARGET].isna().any() or not pool[TARGET].isin([0, 1]).all():
        raise ValueError("Pool precisa de target binário completo")
    if reference_ids.isna().any() or not reference_ids.is_unique:
        raise ValueError("Identificadores de referência ausentes ou duplicados")
    if pool[ID_COLUMN].isin(reference_ids).any():
        raise ValueError("Vazamento: pool e referência compartilham identificadores")
    if (
        min(
            params.batch_rows,
            params.control_rows,
            params.calibration_rows,
            params.demo_rows,
        )
        < min_rows
    ):
        raise ValueError("Volumes não atendem ao mínimo do contrato")
    audit = _validate(pool, "production_pool", min_rows)
    required_rows = (
        params.control_rows
        + params.calibration_rows
        + params.demo_rows
        + 7 * params.batch_rows
    )
    if len(pool) < required_rows:
        raise ValueError(
            f"Pool insuficiente: necessário {required_rows}, disponível {len(pool)}"
        )

    # Streams independentes para seleção, sete intervenções e corrupção.
    seeds = np.random.SeedSequence(params.seed).spawn(9)
    selected = np.random.default_rng(seeds[0]).permutation(len(pool))[:required_rows]
    cursor = 0
    frames: dict[str, pd.DataFrame] = {}
    reservations = {}

    def take(rows: int) -> pd.DataFrame:
        nonlocal cursor
        frame = (
            pool.iloc[selected[cursor : cursor + rows]].copy().reset_index(drop=True)
        )
        cursor += rows
        return frame

    for name, rows in [
        ("control", params.control_rows),
        ("calibration", params.calibration_rows),
        ("demo", params.demo_rows),
    ]:
        frame = take(rows)
        path = f"reserves/{name}.parquet"
        frames[path] = frame
        reservations[name] = {
            "path": path,
            "rows": rows,
            "content_sha256": frame_digest(frame),
        }

    batches = []
    for i, month in enumerate(params.months):
        original = take(params.batch_rows)
        rng = np.random.default_rng(seeds[i + 1])
        scenario, intensity = "stable", 0.0
        current = original.copy(deep=True)
        if 2 <= i <= 3:
            scenario, intensity = "data_drift", params.data_intensities[i - 2]
            current = apply_data_drift(original, intensity, rng, params)
        elif i >= 4:
            scenario, intensity = "concept_drift", params.concept_intensities[i - 4]
            current = apply_concept_drift(original, intensity, rng, params)
        contract = _validate(current, month, min_rows)
        unchanged = [
            c
            for c in original
            if c
            not in (
                DATA_FEATURES
                if scenario == "data_drift"
                else [TARGET]
                if scenario == "concept_drift"
                else []
            )
        ]
        pd.testing.assert_frame_equal(original[unchanged], current[unchanged])
        path, original_path = f"{month}.parquet", f"originals/{month}.parquet"
        frames[path], frames[original_path] = current, original
        new_default, cure = concept_segments(original, params)
        batches.append(
            {
                "batch_id": month,
                "scenario": scenario,
                "intensity": intensity,
                "path": path,
                "original_path": original_path,
                "rows": len(current),
                "content_sha256": frame_digest(current),
                "original_content_sha256": frame_digest(original),
                "original_default_rate": float(original[TARGET].mean()),
                "default_rate": float(current[TARGET].mean()),
                "labels_0_to_1": int(
                    (original[TARGET].eq(0) & current[TARGET].eq(1)).sum()
                ),
                "labels_1_to_0": int(
                    (original[TARGET].eq(1) & current[TARGET].eq(0)).sum()
                ),
                "eligible_new_defaults": int(new_default.sum())
                if scenario == "concept_drift"
                else 0,
                "eligible_cures": int(cure.sum()) if scenario == "concept_drift" else 0,
                "contract": contract,
            }
        )

    corrupted = make_corrupted(
        frames["reserves/demo.parquet"], np.random.default_rng(seeds[8])
    )
    blocked = validate_batch(corrupted, "corrupted", min_rows=min_rows)
    if blocked.passed:
        raise ValueError("Lote de demonstração deveria falhar no contrato")
    frames["corrupted.parquet"] = corrupted
    # Assertiva sobre partições; originals são cópias pareadas, não novas partições.
    partition_paths = [item["path"] for item in reservations.values()] + [
        item["path"] for item in batches
    ]
    identifiers = pd.concat(
        [frames[p][ID_COLUMN] for p in partition_paths], ignore_index=True
    )
    if not identifiers.is_unique:
        raise ValueError("Partições da simulação se sobrepõem")
    return SimulationData(
        frames,
        {
            "simulation_version": SIMULATION_VERSION,
            "params": params.model_dump(mode="json"),
            "contract_min_rows": min_rows,
            "rng": "numpy.default_rng/PCG64; SeedSequence.spawn: selection, M1-M7, corrupted",
            "audit": {
                "pool_rows": len(pool),
                "selected_rows": required_rows,
                "excluded_rows": 0,
                "unused_rows": len(pool) - required_rows,
                "reference_overlap": 0,
                "partition_overlap": 0,
                "pool_contract": audit,
            },
            "reservations": reservations,
            "batches": batches,
            "corrupted": {
                "path": "corrupted.parquet",
                "source_path": "reserves/demo.parquet",
                "rows": len(corrupted),
                "expected_blocked": True,
                "content_sha256": frame_digest(corrupted),
                "blocker_rules": sorted({v.rule for v in blocked.blockers}),
            },
            "interpretation": "Synthetic timeline; issue_date preserved. X changes do not guarantee pure covariate shift. Relabel is simulated, not an observed economic cause.",
        },
    )


def generate(
    pool_path: Path = PRODUCTION_POOL_FILE,
    reference_path: Path = REFERENCE_FILE,
    output_dir: Path = PRODUCTION_DIR,
    params: SimulationParams | None = None,
    min_rows: int | None = None,
) -> dict:
    """Grava parquets e manifesto (por último), após auditar toda a simulação.

    Reexecução substitui somente os arquivos conhecidos. Não execute simultâneo
    ao monitoramento. Use outro output-dir se alterar o calendário de meses.
    """
    params = params or get_params().simulation
    min_rows = get_params().contract.min_rows if min_rows is None else min_rows
    for path in (pool_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Arquivo ausente: {path.name}. Execute make prepare."
            )
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["params"]["months"] != list(params.months):
            raise ValueError(
                "Calendário diferente: use outro output-dir para evitar lotes antigos"
            )
    if pool_path.resolve().is_relative_to(
        output_dir.resolve()
    ) or reference_path.resolve().is_relative_to(output_dir.resolve()):
        raise ValueError("Fontes não podem estar dentro do diretório de saída")
    simulation = build_simulation(
        pd.read_parquet(pool_path),
        pd.read_parquet(reference_path, columns=[ID_COLUMN])[ID_COLUMN],
        params,
        min_rows,
    )
    manifest = simulation.manifest
    manifest["sources"] = {
        "pool": {"name": pool_path.name, "sha256": file_digest(pool_path)},
        "reference": {
            "name": reference_path.name,
            "sha256": file_digest(reference_path),
        },
    }
    manifest["implementation_sha256"] = {
        name: file_digest(Path(__file__).with_name(name))
        for name in (
            "simulation.py",
            "parameters.py",
            "validate.py",
            "data_contract.py",
        )
    }
    manifest["versions"] = {
        name: version(name) for name in ("numpy", "pandas", "pyarrow", "pandera")
    }
    manifest["versions"]["python"] = platform.python_version()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix="simulation-", dir=output_dir.parent
    ) as staging_name:
        staging = Path(staging_name)
        artifacts = {}
        for relative, frame in simulation.frames.items():
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            artifacts[relative] = {"sha256": file_digest(path), "rows": len(frame)}
        manifest["artifacts"] = artifacts
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        for relative in [*artifacts, "manifest.json"]:
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / relative, destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pool", type=Path, default=PRODUCTION_POOL_FILE)
    parser.add_argument("--reference", type=Path, default=REFERENCE_FILE)
    parser.add_argument("--output-dir", type=Path, default=PRODUCTION_DIR)
    args = parser.parse_args()
    setup_logging()
    manifest = generate(args.pool, args.reference, args.output_dir)
    logger.info(
        "simulação gerada",
        meses=len(manifest["batches"]),
        linhas_mensais=sum(b["rows"] for b in manifest["batches"]),
        linhas_reservadas=sum(r["rows"] for r in manifest["reservations"].values()),
        manifesto=str(args.output_dir / "manifest.json"),
    )


if __name__ == "__main__":
    main()
