"""Carregamento e validação dos parâmetros do pipeline (params.yaml).

Cada etapa acrescenta a sua seção como um modelo pydantic e um campo em
`PipelineParams`. `extra="forbid"` faz um typo no YAML falhar na hora.
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from credit_scoring_observability.config import PARAMS_PATH


class ParamsModel(BaseModel):
    """Modelo base que rejeita parâmetros desconhecidos."""

    model_config = ConfigDict(extra="forbid")


class SplitParams(ParamsModel):
    test_size: float = Field(gt=0.0, lt=1.0)
    reference_sample_rows: int = Field(gt=0)
    chunksize: int = Field(gt=0)


class ThresholdGrid(ParamsModel):
    start: float = Field(gt=0.0, lt=1.0)
    stop: float = Field(gt=0.0, lt=1.0)
    step: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_order(self) -> ThresholdGrid:
        if self.start >= self.stop:
            raise ValueError("start deve ser menor que stop")
        return self

    def values(self) -> np.ndarray:
        """Thresholds candidatos (mesma grade do notebook 04)."""
        return np.round(np.arange(self.start, self.stop + 1e-3, self.step), 3)


class TrainParams(ParamsModel):
    train_sample_rows: int = Field(gt=0)
    validation_fraction: float = Field(gt=0.0, lt=1.0)
    min_precision: float = Field(gt=0.0, lt=1.0)
    threshold_grid: ThresholdGrid
    class_weights: list[Literal["standard", "balanced"]] = Field(min_length=1)
    max_iter: int = Field(gt=0)
    tol: float = Field(gt=0.0)


class ContractParams(ParamsModel):
    min_rows: int = Field(gt=0)


class RetentionParams(ParamsModel):
    quarantine_days: int = Field(gt=0)


Intensity = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class SimulationParams(ParamsModel):
    """Sorteios e intervenções; independentes do split/treino da Etapa 1."""

    seed: int = Field(default=42, ge=0, strict=True)
    batch_rows: int = Field(default=10_000, ge=500, strict=True)
    control_rows: int = Field(default=30_000, ge=500, strict=True)
    calibration_rows: int = Field(default=10_000, ge=500, strict=True)
    demo_rows: int = Field(default=2_000, ge=500, strict=True)
    months: tuple[str, ...] = (
        "2026-01",
        "2026-02",
        "2026-03",
        "2026-04",
        "2026-05",
        "2026-06",
        "2026-07",
    )
    data_intensities: tuple[Intensity, Intensity] = (0.6, 1.0)
    concept_intensities: tuple[Intensity, Intensity, Intensity] = (0.35, 0.55, 0.75)
    income_reduction: Intensity = 0.35
    dti_increase: float = Field(default=0.8, ge=0, allow_inf_nan=False)
    utilization_increase_pp: float = Field(default=25.0, ge=0, allow_inf_nan=False)
    fico_drop_points: int = Field(default=30, ge=0, multiple_of=5, strict=True)
    concept_low_fico: float = Field(default=680, ge=300, le=850, allow_inf_nan=False)
    concept_high_fico: float = Field(default=700, ge=300, le=850, allow_inf_nan=False)
    concept_default_probability: Intensity = 0.25
    concept_cure_probability: Intensity = 0.2

    @model_validator(mode="after")
    def validate_timeline(self) -> SimulationParams:
        if self.concept_low_fico >= self.concept_high_fico:
            raise ValueError("concept_low_fico deve ser menor que concept_high_fico")
        if len(self.months) != 7 or len(set(self.months)) != 7:
            raise ValueError("São necessários sete meses distintos")
        for month in self.months:
            parsed = datetime.strptime(month, "%Y-%m")
            if parsed.strftime("%Y-%m") != month:
                raise ValueError("Meses devem usar YYYY-MM")
        ordinals = [int(m[:4]) * 12 + int(m[5:]) for m in self.months]
        if any(b - a != 1 for a, b in pairwise(ordinals)):
            raise ValueError("Meses devem ser consecutivos e crescentes")
        for values in (self.data_intensities, self.concept_intensities):
            if tuple(sorted(values)) != values:
                raise ValueError("Intensidades devem ser não decrescentes")
        return self


class DriftParams(ParamsModel):
    """Limiares operacionais; não representam significância estatística universal."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    bins: int = Field(default=10, ge=2, le=100, strict=True)
    epsilon: float = Field(default=1e-6, gt=0, lt=0.01)
    psi_warning: float = Field(default=0.10, gt=0)
    psi_alert: float = Field(default=0.25, gt=0)
    ks_d: float = Field(default=0.10, gt=0, le=1)
    categorical_js: float = Field(default=0.10, gt=0, le=1)
    missing_rate_delta: float = Field(default=0.05, gt=0, le=1)
    label_z: float = Field(default=3.0, gt=0)

    @model_validator(mode="after")
    def validate_thresholds(self) -> DriftParams:
        if self.psi_warning >= self.psi_alert:
            raise ValueError("psi_warning deve ser menor que psi_alert")
        return self


class PerformanceParams(ParamsModel):
    """Avaliação com rótulos; bootstrap zero desabilita intervalos futuros."""

    auc_drop: float = Field(default=0.05, gt=0, lt=1, allow_inf_nan=False)
    bootstrap_samples: int = Field(default=200, ge=0, le=2000, strict=True)
    seed: int = Field(default=42, ge=0, strict=True)


class PipelineParams(ParamsModel):
    split: SplitParams
    train: TrainParams
    contract: ContractParams
    retention: RetentionParams
    simulation: SimulationParams = Field(default_factory=SimulationParams)
    drift: DriftParams = Field(default_factory=DriftParams)
    performance: PerformanceParams = Field(default_factory=PerformanceParams)

    @model_validator(mode="after")
    def validate_batch_volume(self) -> PipelineParams:
        if (
            min(
                self.simulation.batch_rows,
                self.simulation.control_rows,
                self.simulation.calibration_rows,
                self.simulation.demo_rows,
            )
            < self.contract.min_rows
        ):
            raise ValueError(
                "Lotes, controle e calibração devem atender contract.min_rows"
            )
        return self


def load_params(path: Path = PARAMS_PATH) -> PipelineParams:
    """Lê e valida um params.yaml."""
    with Path(path).open(encoding="utf-8") as f:
        return PipelineParams.model_validate(yaml.safe_load(f))


@lru_cache
def get_params() -> PipelineParams:
    """Retorna os parâmetros do params.yaml padrão (carregados uma única vez)."""
    return load_params()
