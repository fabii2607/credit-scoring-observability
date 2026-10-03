import json

import numpy as np
import pandas as pd
import pytest

from credit_scoring_observability.config import ID_COLUMN
from credit_scoring_observability.validate import (
    EXIT_BLOCKED,
    EXIT_OK,
    ContractViolationError,
    enforce,
    make_corrupted,
    mask_id,
    run,
    validate_batch,
)


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    quarantine = tmp_path / "quarantine"
    # A CLI usa os diretórios padrão: aponta para o tmp.
    monkeypatch.setattr(
        "credit_scoring_observability.validate.CONTRACT_REPORTS_DIR", reports
    )
    monkeypatch.setattr(
        "credit_scoring_observability.validate.QUARANTINE_DIR", quarantine
    )
    return reports, quarantine


def rules(result, severity="blocker"):
    return {v.rule for v in result.violations if v.severity == severity}


def test_valid_batch_passes(model_batch):
    result = validate_batch(model_batch, "ok", min_rows=100)
    assert result.passed
    assert not result.blockers


def test_corrupted_batch_lists_every_rule(model_batch, rng):
    result = validate_batch(make_corrupted(model_batch, rng), "bad", min_rows=100)
    assert not result.passed
    assert {
        "duplicate_customer_id",
        "unexpected_column",
        "greater_than",
        "isin",
        "not_nullable",
        "in_range",
    } <= rules(result)
    columns = {v.column for v in result.blockers}
    assert {
        "loan_amnt",
        "term",
        "annual_inc",
        "fico_avg",
        "application_type",
    } <= columns
    consistency = [v for v in result.blockers if v.rule == "emp_length_consistency"]
    assert len(consistency) == 1
    assert consistency[0].column is None
    assert consistency[0].count > 1  # linhas apontadas, não só "a tabela falhou"
    assert consistency[0].examples
    assert {v.layer for v in result.blockers} == {"batch", "model_input"}


@pytest.mark.parametrize(
    ("change", "rule"),
    [
        (lambda b: b.drop(columns=ID_COLUMN), "missing_customer_id"),
        (lambda b: b.drop(columns="fico_avg"), "column_in_dataframe"),
        (lambda b: b.assign(loan_status="Charged Off"), "unexpected_column"),
        (lambda b: b.head(10), "min_volume"),
        (lambda b: b.assign(target=7), "target_binary"),
        (
            lambda b: pd.concat([b, b.head(3)], ignore_index=True),
            "duplicate_customer_id",
        ),
    ],
)
def test_batch_rules_block(model_batch, change, rule):
    result = validate_batch(change(model_batch), "x", min_rows=100)
    assert rule in rules(result)


def test_same_profile_with_other_id_is_only_a_warning(model_batch):
    twin = model_batch.head(1).assign(**{ID_COLUMN: "outro-cliente-000"})
    result = validate_batch(pd.concat([model_batch, twin]), "x", min_rows=100)
    assert result.passed
    assert "duplicate_content" in rules(result, "warning")


def test_examples_are_masked(model_batch, rng):
    result = validate_batch(make_corrupted(model_batch, rng), "bad", min_rows=100)
    examples = [e for v in result.violations for e in v.examples]
    assert examples
    assert all(e.endswith("***") and len(e) == 7 for e in examples)
    raw_ids = set(model_batch[ID_COLUMN])
    assert not raw_ids & set(examples)
    assert mask_id("abcdef123") == "abcd***"


def test_enforce_quarantines_and_raises(model_batch, rng, tmp_path):
    source = tmp_path / "2026-03.parquet"
    bad = make_corrupted(model_batch, rng)
    bad.to_parquet(source)
    result = validate_batch(bad, "2026-03", min_rows=100)
    with pytest.raises(ContractViolationError, match="2026-03"):
        enforce(bad, result, source, tmp_path / "rep", tmp_path / "q")
    assert not source.exists()
    assert (tmp_path / "q" / "2026-03.parquet").exists()
    report = json.loads((tmp_path / "rep" / "2026-03.json").read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["n_blockers"] == len(result.blockers)
    for violation in report["violations"]:
        assert {"rule", "column", "severity", "count", "layer", "examples"} <= set(
            violation
        )
    assert not set(model_batch[ID_COLUMN]) & set(json.dumps(report).split('"'))


def test_enforce_passes_and_writes_report(model_batch, tmp_path):
    result = validate_batch(model_batch, "ok", min_rows=100)
    report = enforce(model_batch, result, None, tmp_path / "rep", tmp_path / "q")
    assert json.loads(report.read_text(encoding="utf-8"))["passed"] is True
    assert not (tmp_path / "q").exists()


def test_cli_exit_codes(model_batch, rng, tmp_path, dirs):
    good = tmp_path / "good.parquet"
    model_batch.to_parquet(good)
    assert run(good) == EXIT_OK

    bad = tmp_path / "bad.csv"
    make_corrupted(model_batch, rng).to_csv(bad, index=False)
    assert run(bad) == EXIT_BLOCKED
    _, quarantine = dirs
    assert (quarantine / "bad.csv").exists()


def test_make_corrupted_keeps_original_untouched(model_batch):
    before = model_batch.copy()
    make_corrupted(model_batch, np.random.default_rng(0))
    pd.testing.assert_frame_equal(model_batch, before)
