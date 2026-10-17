"""Estatística pura de drift numérico: PSI por quantis e KS de duas amostras.

Quantis são ajustados SOMENTE na referência. Nulos saem dos cálculos de
distribuição e são monitorados separadamente. Amostras vazias/sem valores
observados retornam indisponibilidade. Infinity e dados não numéricos falham
explicitamente. KS p-valor é informativo, nunca o gate operacional.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from credit_scoring_observability.parameters import DriftParams
from credit_scoring_observability.results import DriftStatus, JsonResult


def _sample(values) -> tuple[np.ndarray, int, float | None]:
    """Aceita vetor numérico com nulos, sem converter texto em número."""
    series = pd.Series(values)
    present = series.dropna()
    if any(isinstance(v, (str, bytes, complex, bool)) for v in present):
        raise ValueError("Amostra deve conter somente números reais ou nulos")
    try:
        array = present.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Amostra deve ser numérica") from error
    if not np.isfinite(array).all():
        raise ValueError("Amostra contém Infinity ou outro valor não finito")
    total = len(series)
    return array, total, float(series.isna().mean()) if total else None


def _cuts(reference: np.ndarray, bins: int) -> np.ndarray:
    if np.min(reference) == np.max(reference):
        # Três grupos: abaixo, igual e acima da constante. Sem esse caso,
        # uma referência constante teria um único bin e esconderia mudanças.
        value = reference[0]
        return np.array([np.nextafter(value, -np.inf), value])
    return np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))


def _psi(ref: np.ndarray, cur: np.ndarray, bins: int, epsilon: float) -> float:
    cuts = _cuts(ref, bins)
    # side=left: intervalos fechados à direita, incluindo empates em quantis.
    # Grupos externos ilimitados incluem TODOS os valores fora da referência.
    ref_counts = np.bincount(
        np.searchsorted(cuts, ref, side="left"), minlength=len(cuts) + 1
    )
    cur_counts = np.bincount(
        np.searchsorted(cuts, cur, side="left"), minlength=len(cuts) + 1
    )
    p = np.maximum(ref_counts / len(ref), epsilon)
    q = np.maximum(cur_counts / len(cur), epsilon)
    # Renormalizar depois do piso mantém duas distribuições de probabilidade.
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum((q - p) * np.log(q / p)))


def psi_quantile(
    reference, current, bins: int = 10, epsilon: float = 1e-6
) -> float | None:
    """PSI com faixas da referência; None se alguma amostra não tem observações.

    Valores da produção nunca alteram os limites. Quantis repetidos são
    deduplicados; o número efetivo de faixas pode ser menor que bins.
    """
    params = DriftParams(bins=bins, epsilon=epsilon)
    ref, _, _ = _sample(reference)
    cur, _, _ = _sample(current)
    if not len(ref) or not len(cur):
        return None
    return _psi(ref, cur, params.bins, params.epsilon)


@dataclass
class KSResult(JsonResult):
    ks_d: float | None
    p_value: float | None


def ks_two_sample(reference, current) -> KSResult:
    """KS bilateral, método assintótico explícito para execução reproduzível.

    O p-valor clássico pressupõe distribuições contínuas; empates/variáveis
    discretas limitam sua interpretação. O D continua uma distância descritiva.
    """
    ref, _, _ = _sample(reference)
    cur, _, _ = _sample(current)
    if not len(ref) or not len(cur):
        return KSResult(None, None)
    # Evitar a aproximação assintótica degenerada quando há uma observação.
    method = "exact" if min(len(ref), len(cur)) == 1 else "asymp"
    result = ks_2samp(ref, cur, alternative="two-sided", method=method)
    if not np.isfinite(result.pvalue):
        result = ks_2samp(ref, cur, alternative="two-sided", method="exact")
    return KSResult(float(result.statistic), float(result.pvalue))


def psi_status(psi: float | None, params: DriftParams | None = None) -> DriftStatus:
    """Faixas inclusivas: WARN em [warning, alert]; ALERT estritamente acima."""
    params = params or DriftParams()
    if psi is None:
        return "UNAVAILABLE"
    if not np.isfinite(psi) or psi < 0:
        raise ValueError("PSI deve ser finito e não negativo")
    if psi > params.psi_alert:
        return "ALERT"
    if psi >= params.psi_warning:
        return "WARN"
    return "OK"


@dataclass
class NumericComparison(JsonResult):
    psi: float | None
    ks_d: float | None
    ks_p_value: float | None
    status: DriftStatus
    ks_confirmed: bool
    critical: bool
    reference_rows: int
    current_rows: int
    reference_valid_rows: int
    current_valid_rows: int
    reference_missing_rate: float | None
    current_missing_rate: float | None
    missing_rate_delta: float | None
    missing_drift: bool | None
    effective_bins: int | None
    unavailable_reason: str | None


def compare_numeric(
    reference, current, params: DriftParams | None = None
) -> NumericComparison:
    """Distribuição e ausências independentes; status representa SOMENTE o PSI.

    critical exige PSI ALERT e KS D > limiar; missing_drift pode ser True mesmo
    se o PSI estiver OK ou indisponível. Delta de nulos é produção - referência.
    """
    params = params or DriftParams()
    ref, ref_total, ref_missing = _sample(reference)
    cur, cur_total, cur_missing = _sample(current)
    available = bool(len(ref) and len(cur))
    psi = _psi(ref, cur, params.bins, params.epsilon) if available else None
    ks = ks_two_sample(ref, cur)
    status = psi_status(psi, params)
    confirmed = ks.ks_d is not None and ks.ks_d > params.ks_d
    delta = (
        cur_missing - ref_missing
        if ref_missing is not None and cur_missing is not None
        else None
    )
    return NumericComparison(
        psi=psi,
        ks_d=ks.ks_d,
        ks_p_value=ks.p_value,
        status=status,
        ks_confirmed=confirmed,
        critical=status == "ALERT" and confirmed,
        reference_rows=ref_total,
        current_rows=cur_total,
        reference_valid_rows=len(ref),
        current_valid_rows=len(cur),
        reference_missing_rate=ref_missing,
        current_missing_rate=cur_missing,
        missing_rate_delta=delta,
        missing_drift=abs(delta) >= params.missing_rate_delta
        if delta is not None
        else None,
        effective_bins=len(_cuts(ref, params.bins)) + 1 if len(ref) else None,
        unavailable_reason=None if available else "empty_or_all_missing_sample",
    )
