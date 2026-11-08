"""Censoring-robust concordance: Uno's IPCW C-statistic and Gönen-Heller CPE.

Harrell's C (:func:`survival_kit.concordance.concordance_index`) depends on
the censoring distribution: with heavier censoring, early event pairs carry
more of the weight and the estimate drifts even when the model is unchanged.
Two remedies:

* :func:`uno_concordance_index` – Uno, Cai, Pencina, D'Agostino & Wei
  (Stat. Med. 2011). Each comparable pair ``(i, j)`` with ``T_i < T_j``,
  ``delta_i = 1`` and ``T_i < tau`` is weighted by ``G(T_i-)^-2`` where ``G``
  is the Kaplan-Meier estimate of the censoring survival::

      C_tau = sum_ij delta_i G(T_i-)^-2 1(T_i < T_j, T_i < tau) [1(eta_i > eta_j) + 0.5 1(eta_i = eta_j)]
              -------------------------------------------------------------------------------------
                           sum_ij delta_i G(T_i-)^-2 1(T_i < T_j, T_i < tau)

  which consistently estimates ``P(eta_i > eta_j | T_i < T_j, T_i < tau)``,
  a quantity free of the censoring distribution. ``G`` can come from a
  training sample. A nonparametric bootstrap gives a standard error and
  percentile interval.

* :func:`gonen_heller_cpe` – Gönen & Heller (Biometrika 2005) concordance
  probability estimate for a proportional-hazards model. It is a function of
  the linear predictors only (no observed times), hence unaffected by
  censoring when the PH model holds::

      CPE = 2 / (n (n-1)) sum_{i<j} [ 1(b_ji < 0) / (1 + e^{b_ji}) + 1(b_ij < 0) / (1 + e^{b_ij}) ]

  with ``b_ij = eta_i - eta_j`` (risk orientation: larger ``eta`` = higher
  hazard). Ties in ``eta`` contribute ``1/2`` (the limit of the smoothed
  version).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kaplan_meier import fit_kaplan_meier
from .utils import validate_durations_events

__all__ = ["UnoConcordanceResult", "gonen_heller_cpe", "uno_concordance_index"]


@dataclass
class UnoConcordanceResult:
    """Uno's truncated IPCW concordance and its pair accounting."""

    c_index: float
    tau: float
    comparable_pairs: int
    weighted_pairs: float
    std_err: float | None
    ci_lower: float | None
    ci_upper: float | None
    n_bootstrap: int


def _censoring_left_limit(train_d, train_e, t):
    if np.all(train_e):
        return np.ones_like(t, dtype=float)
    curve = fit_kaplan_meier(train_d, ~train_e)
    return np.asarray(curve.at(np.nextafter(t, -np.inf)), dtype=float)


def _uno_core(d, e, s, tau, g_left, tied_tol):
    order = np.argsort(d, kind="mergesort")
    d, e, s, g_left = d[order], e[order], s[order], g_left[order]
    num = 0.0
    den = 0.0
    pairs = 0
    n = d.size
    for i in range(n):
        if not e[i] or d[i] >= tau:
            continue
        # strictly later observed times
        k = int(np.searchsorted(d, d[i], side="right"))
        if k >= n:
            continue
        w = 1.0 / (g_left[i] ** 2)
        diff = s[i] - s[k:]
        conc = np.count_nonzero(diff > tied_tol)
        ties = np.count_nonzero(np.abs(diff) <= tied_tol)
        m = n - k
        num += w * (conc + 0.5 * ties)
        den += w * m
        pairs += m
    return num, den, pairs


def uno_concordance_index(
    durations,
    events,
    risk_scores,
    *,
    tau: float | None = None,
    train_durations=None,
    train_events=None,
    tied_tol: float = 1e-8,
    n_bootstrap: int = 0,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> UnoConcordanceResult:
    """Uno's IPCW concordance truncated at ``tau``.

    ``risk_scores``: larger = earlier expected event. ``tau`` defaults to the
    largest observed event time (pairs need ``T_i < tau``); choose a ``tau``
    where the censoring survival ``G`` is still clearly positive. ``G`` is
    estimated on ``train_durations`` / ``train_events`` when given (and is
    then held fixed in the bootstrap), otherwise on the evaluation data.
    """
    d, e = validate_durations_events(durations, events)
    s = np.asarray(risk_scores, dtype=float)
    if s.ndim != 1 or s.shape[0] != d.shape[0]:
        raise ValueError("risk_scores must be one-dimensional and match the durations")
    if not np.all(np.isfinite(s)):
        raise ValueError("risk_scores must contain only finite values")
    if (train_durations is None) != (train_events is None):
        raise ValueError("pass both train_durations and train_events, or neither")
    if not e.any():
        raise ValueError("no events: concordance is undefined")
    if tau is None:
        tau = float(np.max(d[e])) + 1e-12 * max(1.0, float(np.max(d[e])))
    tau = float(tau)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be a positive finite number")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be non-negative")

    external = train_durations is not None
    if external:
        tr_d, tr_e = validate_durations_events(train_durations, train_events)

    def g_for(dd, ee):
        if external:
            return _censoring_left_limit(tr_d, tr_e, dd)
        return _censoring_left_limit(dd, ee, dd)

    g_left = g_for(d, e)
    if np.any((g_left <= 0.0) & e & (d < tau)):
        raise ValueError("censoring survival G(T-) is 0 before tau; choose a smaller tau")
    num, den, pairs = _uno_core(d, e, s, tau, g_left, tied_tol)
    if den <= 0:
        raise ValueError("no comparable pairs before tau: concordance is undefined")
    c = num / den

    se = lo = hi = None
    if n_bootstrap:
        rng = np.random.default_rng(seed)
        boots = []
        n = d.size
        for _ in range(int(n_bootstrap)):
            idx = rng.integers(0, n, n)
            bd, be, bs = d[idx], e[idx], s[idx]
            bg = g_for(bd, be)
            if np.any((bg <= 0.0) & be & (bd < tau)):
                continue
            bn, bden, _ = _uno_core(bd, be, bs, tau, bg, tied_tol)
            if bden > 0:
                boots.append(bn / bden)
        if len(boots) >= 2:
            boots = np.asarray(boots)
            a = (1.0 - confidence_level) / 2.0
            se = float(np.std(boots, ddof=1))
            lo, hi = (float(v) for v in np.quantile(boots, [a, 1.0 - a]))
    return UnoConcordanceResult(
        c_index=float(c),
        tau=tau,
        comparable_pairs=int(pairs),
        weighted_pairs=float(den),
        std_err=se,
        ci_lower=lo,
        ci_upper=hi,
        n_bootstrap=int(n_bootstrap),
    )


def gonen_heller_cpe(linear_predictor, *, chunk_size: int = 2048) -> float:
    """Gönen-Heller concordance probability estimate for a PH linear predictor.

    ``linear_predictor`` is ``x @ beta`` from a Cox model (larger = higher
    hazard). Uses only the predictors, so it is unaffected by censoring
    under proportional hazards. ``O(n^2)`` time, ``O(n * chunk_size)`` memory.
    """
    eta = np.asarray(linear_predictor, dtype=float).ravel()
    if eta.size < 2:
        raise ValueError("need at least two linear predictors")
    if not np.all(np.isfinite(eta)):
        raise ValueError("linear_predictor must contain only finite values")
    n = eta.size
    total = 0.0
    for start in range(0, n, chunk_size):
        block = eta[start : start + chunk_size]
        diff = np.abs(block[:, None] - eta[None, :])  # |eta_i - eta_j|
        # for each unordered pair exactly one of b_ij, b_ji is negative:
        # contribution 1 / (1 + exp(-|b|)); ties give 1/2
        contrib = 1.0 / (1.0 + np.exp(-diff))
        # remove self-pairs (diff = 0 -> 1/2)
        rows = np.arange(block.size)
        contrib[rows, start + rows] = 0.0
        total += float(contrib.sum())
    # each unordered pair counted twice
    return total / (n * (n - 1))
