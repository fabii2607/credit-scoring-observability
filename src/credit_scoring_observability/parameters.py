"""Carregamento e validação dos parâmetros do pipeline (params.yaml).

Cada etapa acrescenta a sua seção como um modelo pydantic e um campo em
`PipelineParams`. `extra="forbid"` faz um typo no YAML falhar na hora.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

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


class PipelineParams(ParamsModel):
    split: SplitParams
    train: TrainParams
    contract: ContractParams
    retention: RetentionParams


def load_params(path: Path = PARAMS_PATH) -> PipelineParams:
    """Lê e valida um params.yaml."""
    with Path(path).open(encoding="utf-8") as f:
        return PipelineParams.model_validate(yaml.safe_load(f))


@lru_cache
def get_params() -> PipelineParams:
    """Retorna os parâmetros do params.yaml padrão (carregados uma única vez)."""
    return load_params()
