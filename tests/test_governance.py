import os
import time

import numpy as np
import pandas as pd
import pytest

from credit_scoring_observability.causal import (
    analyze_batch,
    recovery,
    restore_group,
    score_psi,
)
from credit_scoring_observability.config import REGION_COLUMN, SCORE_COLUMN, TARGET
from credit_scoring_observability.fairness import (
    approval_disparity,
    fairness_by_region,
)
from credit_scoring_observability.mitigation import (
    decide,
    evaluate_mitigation,
    fpr_parity_thresholds,
)
from credit_scoring_observability.retention import expired_batches, purge
from credit_scoring_observability.synthetic import make_model_batch

# ── Fairness ─────────────────────────────────────────────────────────────────


def test_fairness_by_region_known_case():
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    score = np.array([0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.9, 0.1])
    region = pd.Series(["A"] * 4 + ["B"] * 4)
    result = fairness_by_region(y, score, 0.5, region)
    assert result["approval_rate"] == {"A": 0.25, "B": 0.75}
    assert result["fpr"] == {"A": 0.5, "B": 0.0}
    assert result["tpr"] == {"A": 1.0, "B": 0.5}
    assert result["disparate_impact"] == pytest.approx(1 / 3, abs=1e-4)
    assert result["passes_four_fifths"] is False
    assert result["n"] == {"A": 4, "B": 4}


def test_approval_disparity_without_labels():
    result = approval_disparity(np.array([1, 0, 0, 0]), pd.Series(["A", "A", "B", "B"]))
    assert result["approval_rate"] == {"A": 0.5, "B": 1.0}
    assert result["disparate_impact"] == 0.5


# ── Mitigação ────────────────────────────────────────────────────────────────


@pytest.fixture
def scored(scoring_model):
    return scoring_model.score(make_model_batch(n=6_000, seed=3))


def test_fpr_parity_thresholds_equalize_fpr(scored):
    y, s, g = scored[TARGET], scored[SCORE_COLUMN], scored[REGION_COLUMN]
    thresholds = fpr_parity_thresholds(y, s, g, target_fpr=0.3)
    denied = decide(s, g, thresholds)
    fpr = pd.Series(denied[y.to_numpy() == 0]).groupby(g[y == 0].to_numpy()).mean()
    assert np.allclose(fpr, 0.3, atol=0.02)
    assert thresholds == fpr_parity_thresholds(y, s, g, target_fpr=0.3)


def test_mitigation_is_evaluated_but_not_adopted(scoring_model):
    scored = scoring_model.score(make_model_batch(n=20_000, seed=3))
    report = evaluate_mitigation(scored, scoring_model.threshold)
    assert report["adopted_in_production"] is False
    # No holdout, o FPR de cada região fica perto da meta (ruído amostral).
    after = report["after"]["fpr_by_group"]
    assert max(abs(v - report["target_fpr"]) for v in after.values()) < 0.03
    assert set(report["group_thresholds"]) == set(scored[REGION_COLUMN])
    assert report["fit_rows"] + report["holdout_rows"] == len(scored)


# ── Retenção ─────────────────────────────────────────────────────────────────


def test_purge_removes_only_expired(tmp_path):
    now = time.time()
    old = tmp_path / "2026-01.parquet"
    recent = tmp_path / "2026-03.csv"
    other = tmp_path / "notes.txt"
    for path in (old, recent, other):
        path.write_text("x")
    os.utime(old, (now - 40 * 86_400, now - 40 * 86_400))
    os.utime(other, (now - 40 * 86_400, now - 40 * 86_400))

    assert expired_batches(tmp_path, 30, now) == [old]
    assert purge(tmp_path, 30, dry_run=True, now=now) == ["2026-01.parquet"]
    assert old.exists()
    assert purge(tmp_path, 30, now=now) == ["2026-01.parquet"]
    assert not old.exists()
    assert recent.exists()
    assert other.exists()


def test_purge_without_quarantine_dir(tmp_path):
    assert purge(tmp_path / "missing", 30) == []


# ── Causal ───────────────────────────────────────────────────────────────────


@pytest.fixture
def reference():
    return make_model_batch(n=5_000, seed=4)


def covariate_batch(seed=5):
    batch = make_model_batch(n=4_000, seed=seed)
    batch["annual_inc"] *= 0.5
    batch["dti"] *= 1.6
    batch["fico_avg"] = (batch["fico_avg"] - 30).clip(lower=300)
    return batch  # rótulos originais: só X mudou


def concept_batch(seed=6):
    batch = make_model_batch(n=4_000, seed=seed)
    cure = (batch[TARGET] == 1) & (batch["fico_avg"] < 720)
    new_default = (batch[TARGET] == 0) & (batch["fico_avg"] > 740)
    batch.loc[cure, TARGET] = 0
    batch.loc[new_default, TARGET] = 1
    return batch  # X igual: só P(Y|X) mudou


def test_restore_group_preserves_order_and_correlation(reference):
    batch = covariate_batch()
    restored = restore_group(batch, reference, ["annual_inc", "dti"])
    for col in ("annual_inc", "dti"):
        present = batch[col].notna()
        assert batch.loc[present, col].corr(
            restored.loc[present, col], method="spearman"
        ) == pytest.approx(1.0, abs=1e-6)
        assert restored[col].isna().equals(batch[col].isna())
        assert set(restored.loc[present, col]) <= set(reference[col].dropna())
    before = batch["annual_inc"].corr(batch["dti"], method="spearman")
    after = restored["annual_inc"].corr(restored["dti"], method="spearman")
    assert after == pytest.approx(before, abs=0.01)
    # Distribuição volta à da Referência.
    assert restored["annual_inc"].median() == pytest.approx(
        reference["annual_inc"].median(), rel=0.05
    )


def test_score_psi_and_recovery():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=5_000)
    assert score_psi(ref, rng.normal(size=5_000)) < 0.02
    assert score_psi(ref, rng.normal(loc=1, size=5_000)) > 0.25
    assert recovery(0.3, 0.03, 0.0) == pytest.approx(0.9)
    assert recovery(0.5, 0.5, 0.5) is None


def test_covariate_shift_is_recovered_by_restoring_x(reference, scoring_model):
    result = analyze_batch(covariate_batch(), reference, scoring_model, "cov")
    assert result["drifted"]["score_psi"] > 0.10
    all_groups = result["interventions"]["renda+dívida + rotativo + score"]
    assert all_groups["recovery_score_psi"] > 0.8
    assert result["diagnosis"].startswith("covariate shift")
    assert len(result["interventions"]) == 7  # todas as combinações dos 3 grupos


def test_concept_drift_is_not_recovered_by_restoring_x(reference, scoring_model):
    result = analyze_batch(concept_batch(), reference, scoring_model, "concept")
    assert result["drifted"]["score_psi"] < 0.05
    assert result["reference"]["roc_auc"] - result["drifted"]["roc_auc"] > 0.05
    assert result["diagnosis"].startswith("concept drift")
