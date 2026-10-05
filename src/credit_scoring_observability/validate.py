"""Validação de lotes de produção: o "script de validação operante" da Etapa 1.

Duas camadas de regras, todas registradas num `ValidationResult`:

- `model_input` (Pandera, `data_contract.MODEL_INPUT_SCHEMA`): as 23 features,
  com tipos, faixas, nulos e coerência entre colunas. Toda falha é blocker.
- `batch` (regras do lote): `customer_id` presente e sem reenvio (duplicado),
  volume mínimo, colunas inesperadas (ex.: `loan_status`, vazamento), target
  binário quando presente — blockers; perfis idênticos com ids diferentes e
  região desconhecida — warnings.

`enforce()` grava `reports/contracts/<lote>.json` (exemplos com o `customer_id`
mascarado: 4 caracteres + `***`) e, havendo blocker, move o lote para
`data/quarantine/` (Dead Letter Channel) e lança `ContractViolationError`.

Uso:
    uv run python -m credit_scoring_observability.validate --batch arquivo.parquet
    uv run python -m credit_scoring_observability.validate --demo-corrupted

Exit 0 = lote aprovado; exit 2 = lote bloqueado.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pandera.errors import SchemaErrors

from credit_scoring_observability.config import (
    CONTRACT_REPORTS_DIR,
    ID_COLUMN,
    ISSUE_COLUMN,
    PREDICTION_COLUMN,
    PRODUCTION_DIR,
    PRODUCTION_POOL_FILE,
    QUARANTINE_DIR,
    REGION_COLUMN,
    REGIONS,
    SCORE_COLUMN,
    TARGET,
)
from credit_scoring_observability.data_contract import (
    MODEL_FEATURES,
    validate_model_input,
)
from credit_scoring_observability.logger import get_logger, setup_logging
from credit_scoring_observability.parameters import get_params

logger = get_logger(__name__)

Severity = Literal["blocker", "warning"]
Layer = Literal["batch", "model_input"]

EXIT_OK = 0
EXIT_BLOCKED = 2
MAX_EXAMPLES = 5
# Colunas que um lote pode trazer além das features.
ALLOWED_EXTRA_COLUMNS = {
    ID_COLUMN,
    ISSUE_COLUMN,
    REGION_COLUMN,
    TARGET,
    SCORE_COLUMN,
    PREDICTION_COLUMN,
}


class ContractViolationError(RuntimeError):
    """O lote violou ao menos uma regra blocker e não pode ser pontuado."""

    def __init__(self, result: ValidationResult):
        self.result = result
        rules = ", ".join(sorted({v.rule for v in result.blockers}))
        super().__init__(f"Lote {result.batch_id} bloqueado pelo contrato: {rules}")


@dataclass
class Violation:
    rule: str
    column: str | None
    severity: Severity
    count: int
    layer: Layer
    examples: list[str] = field(default_factory=list)  # customer_id mascarados


@dataclass
class ValidationResult:
    batch_id: str
    n_rows: int
    violations: list[Violation] = field(default_factory=list)
    validated_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )

    @property
    def blockers(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "blocker"]

    @property
    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "warning"]

    @property
    def passed(self) -> bool:
        return not self.blockers

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "passed": self.passed,
            "n_blockers": len(self.blockers),
            "n_warnings": len(self.warnings),
        }


def mask_id(value: object) -> str:
    """Identificador mascarado para relatórios: 4 caracteres + `***`."""
    return f"{str(value)[:4]}***"


def _examples(batch: pd.DataFrame, mask: pd.Series | np.ndarray) -> list[str]:
    if ID_COLUMN not in batch:
        return []
    ids = batch.loc[np.asarray(mask), ID_COLUMN].head(MAX_EXAMPLES)
    return [mask_id(v) for v in ids]


def _batch_rules(batch: pd.DataFrame, min_rows: int) -> list[Violation]:
    violations: list[Violation] = []

    def add(rule, column, severity, mask=None, count=None):
        n = int(np.asarray(mask).sum()) if mask is not None else int(count)
        if n:
            violations.append(
                Violation(
                    rule=rule,
                    column=column,
                    severity=severity,
                    count=n,
                    layer="batch",
                    examples=_examples(batch, mask) if mask is not None else [],
                )
            )

    add("min_volume", None, "blocker", count=len(batch) < min_rows)

    unexpected = sorted(
        set(batch.columns) - set(MODEL_FEATURES) - ALLOWED_EXTRA_COLUMNS
    )
    for column in unexpected:
        add("unexpected_column", column, "blocker", count=1)

    if ID_COLUMN not in batch:
        add("missing_customer_id", ID_COLUMN, "blocker", count=1)
    else:
        add("customer_id_not_null", ID_COLUMN, "blocker", batch[ID_COLUMN].isna())
        # Mesmo cliente reenviado no lote: blocker (dupla pontuação).
        add(
            "duplicate_customer_id",
            ID_COLUMN,
            "blocker",
            batch[ID_COLUMN].duplicated(keep=False) & batch[ID_COLUMN].notna(),
        )

    present = [c for c in MODEL_FEATURES if c in batch]
    if present:
        # Perfis idênticos com ids diferentes acontecem naturalmente: warning.
        same_content = batch.duplicated(present, keep=False)
        if ID_COLUMN in batch:
            same_content &= ~batch[ID_COLUMN].duplicated(keep=False)
        add("duplicate_content", None, "warning", same_content)

    if TARGET in batch:
        add(
            "target_binary",
            TARGET,
            "blocker",
            batch[TARGET].notna() & ~batch[TARGET].isin([0, 1]),
        )
    if REGION_COLUMN in batch:
        add(
            "unknown_region",
            REGION_COLUMN,
            "warning",
            ~batch[REGION_COLUMN].isin(REGIONS),
        )
    return violations


def _rule_name(check: object) -> str:
    """Nome estável da regra a partir da descrição do check do Pandera.

    `greater_than(0)` -> `greater_than`; checks de tabela usam o prefixo da
    mensagem (`emp_length_consistency: ...` -> `emp_length_consistency`).
    """
    return str(check).split("(")[0].split(":")[0].strip()


def _model_input_rules(batch: pd.DataFrame) -> list[Violation]:
    features = batch.loc[:, [c for c in batch.columns if c in MODEL_FEATURES]]
    try:
        validate_model_input(features)
    except SchemaErrors as error:
        cases = error.failure_cases.copy()
    except ValueError as error:  # lote vazio
        return [Violation(str(error), None, "blocker", 1, "model_input")]
    else:
        return []

    cases["rule"] = cases["check"].map(_rule_name)
    # Checks de tabela aparecem repetidos por coluna: viram uma regra sem coluna.
    table_level = cases["schema_context"].eq("DataFrameSchema") | cases["column"].isna()
    cases["column"] = cases["column"].astype("object").where(~table_level, None)
    # Coluna ausente/inesperada: o nome da coluna vem em failure_case.
    presence = cases["rule"].isin(["column_in_dataframe", "column_in_schema"])
    cases.loc[presence, "column"] = cases.loc[presence, "failure_case"].astype(str)
    violations = []
    for (rule, column), group in cases.groupby(
        ["rule", "column"], dropna=False, sort=True
    ):
        rows = pd.to_numeric(group["index"], errors="coerce").dropna().astype(int)
        mask = batch.index.isin(rows.unique())
        violations.append(
            Violation(
                rule=str(rule),
                column=None if pd.isna(column) else str(column),
                severity="blocker",
                count=int(rows.nunique()) or len(group),
                layer="model_input",
                examples=_examples(batch, mask),
            )
        )
    return violations


def validate_batch(
    batch: pd.DataFrame, batch_id: str, min_rows: int | None = None
) -> ValidationResult:
    """Aplica as duas camadas e devolve todas as violações (não lança exceção)."""
    min_rows = get_params().contract.min_rows if min_rows is None else min_rows
    batch = batch.reset_index(drop=True)
    return ValidationResult(
        batch_id=batch_id,
        n_rows=len(batch),
        violations=_batch_rules(batch, min_rows) + _model_input_rules(batch),
    )


def enforce(
    batch: pd.DataFrame,
    result: ValidationResult,
    source_path: Path | None = None,
    report_dir: Path | None = None,
    quarantine_dir: Path | None = None,
) -> Path:
    """Grava o relatório; com blocker, põe o lote em quarentena e lança exceção.

    O arquivo de origem (se houver) é movido para a quarentena; sem arquivo, o
    lote é gravado lá em parquet. Devolve o caminho do relatório. Diretórios
    padrão: `reports/contracts/` e `data/quarantine/`.
    """
    report_dir = report_dir or CONTRACT_REPORTS_DIR
    quarantine_dir = quarantine_dir or QUARANTINE_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{result.batch_id}.json"
    report_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",  # LF também no Windows: sem diff falso no git
    )
    if result.passed:
        logger.info(
            "lote aprovado no contrato",
            batch_id=result.batch_id,
            linhas=result.n_rows,
            warnings=len(result.warnings),
        )
        return report_path

    quarantine_dir.mkdir(parents=True, exist_ok=True)
    if source_path is not None and source_path.exists():
        destination = quarantine_dir / source_path.name
        shutil.move(source_path, destination)
    else:
        destination = quarantine_dir / f"{result.batch_id}.parquet"
        batch.to_parquet(destination, index=False)
    logger.error(
        "lote bloqueado pelo contrato",
        batch_id=result.batch_id,
        blockers=len(result.blockers),
        regras=sorted({v.rule for v in result.blockers}),
        quarentena=str(destination),
        relatorio=str(report_path),
    )
    raise ContractViolationError(result)


def make_corrupted(
    df: pd.DataFrame, rng: np.random.Generator, fraction: float = 0.05
) -> pd.DataFrame:
    """Lote corrompido de demonstração a partir de um lote válido.

    Mesmos erros do Bad Batch do notebook 05 (valor negativo, prazo inválido,
    renda ausente, FICO fora da faixa, flag de emprego incoerente), aplicados a
    ~`fraction` das linhas cada, mais: tipo de aplicação inválido, clientes
    reenviados (customer_id duplicado) e a coluna de desfecho `loan_status`
    (vazamento). A Etapa 2 importa esta função para gerar o lote `corrupted`.
    """
    bad = df.copy().reset_index(drop=True)
    n = len(bad)
    k = max(1, int(n * fraction))

    def rows() -> np.ndarray:
        return rng.choice(n, size=k, replace=False)

    bad.loc[rows(), "loan_amnt"] = -500.0
    bad.loc[rows(), "term"] = 48
    bad.loc[rows(), "annual_inc"] = np.nan
    bad.loc[rows(), "fico_avg"] = 910.0
    flag_rows = rows()
    bad.loc[flag_rows, "emp_length_missing"] = 1
    bad.loc[flag_rows, "emp_length_years"] = 5
    bad["application_type"] = bad["application_type"].astype("object")
    bad.loc[rows(), "application_type"] = "Invalid App"
    if ID_COLUMN in bad:
        resent = bad.loc[rows()]
        bad = pd.concat([bad, resent], ignore_index=True)
    bad["loan_status"] = "Fully Paid"
    return bad


def read_batch(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def run(path: Path, batch_id: str | None = None) -> int:
    """Valida um arquivo de lote; devolve o exit code (0 aprovado, 2 bloqueado)."""
    batch = read_batch(path)
    result = validate_batch(batch, batch_id or path.stem)
    try:
        report = enforce(batch, result, source_path=path)
    except ContractViolationError as error:
        print(f"BLOQUEADO: {error}")
        for v in error.result.blockers:
            column = f" [{v.column}]" if v.column else ""
            print(f"  - {v.layer}: {v.rule}{column}: {v.count} linha(s)")
        print(f"Relatório: {CONTRACT_REPORTS_DIR / (result.batch_id + '.json')}")
        return EXIT_BLOCKED
    print(f"APROVADO: {result.n_rows} linhas, {len(result.warnings)} warning(s).")
    print(f"Relatório: {report}")
    return EXIT_OK


def write_demo_corrupted(seed: int = 42, n: int = 2_000) -> Path:
    """Gera `data/production/corrupted.parquet` a partir do pool (ou sintético)."""
    rng = np.random.default_rng(seed)
    if PRODUCTION_POOL_FILE.exists():
        pool = pd.read_parquet(PRODUCTION_POOL_FILE)
        base = pool.sample(n=min(n, len(pool)), random_state=seed)
    else:
        from credit_scoring_observability.synthetic import make_model_batch

        logger.warning("pool de produção ausente: usando lote sintético")
        base = make_model_batch(n=n, seed=seed)
    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    path = PRODUCTION_DIR / "corrupted.parquet"
    make_corrupted(base, rng).to_parquet(path, index=False)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Valida um lote pelo contrato de dados."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--batch", type=Path, help="arquivo .parquet ou .csv do lote")
    group.add_argument(
        "--demo-corrupted",
        action="store_true",
        help="gera o lote corrompido de demonstração e valida (esperado: exit 2)",
    )
    parser.add_argument("--batch-id", help="nome do lote (padrão: nome do arquivo)")
    args = parser.parse_args(argv)
    setup_logging()
    path = write_demo_corrupted() if args.demo_corrupted else args.batch
    return run(path, args.batch_id)


if __name__ == "__main__":
    sys.exit(main())
